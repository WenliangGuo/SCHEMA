import torch.nn as nn
import torch

from models.state_encoder import StateEncoder
from models.state_decoder import StateDecoder
from models.action_decoder import ActionDecoder
from models.utils import viterbi_path

class ProcedureModel(nn.Module):
    def __init__(
            self, 
            vis_input_dim,
            lang_input_dim,
            embed_dim, 
            time_horz, 
            num_classes,
            num_tasks,
            args
        ):
        '''Procedure model initialization

        This class defines the Procedure Model. It consists of a state encoder,
        a state decoder, a task decoder, and an action decoder.

        Args:
            vis_input_dim:  dimension of visual features.
            lang_input_dim: dimension of language features.
            embed_dim:      dimension of embedding features.
            time_horz:      time horizon.
            num_classes:    number of action classes.
            num_tasks:      number of tasks.
            args:           arguments from parser.
        '''
        super().__init__()

        self.att_heads = args.attn_heads
        self.mlp_ratio = args.mlp_ratio
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.uncertainty = args.uncertain
        self.dataset = args.dataset
        self.time_horz = time_horz
        self.embed_dim = embed_dim

        self.state_encoder = StateEncoder(
            vis_input_dim,
            lang_input_dim,
            embed_dim, 
            dropout = 0.4
        )

        self.state_decoder = StateDecoder(
            embed_dim = embed_dim, 
            time_horz = time_horz, 
            att_heads = self.att_heads,
            mlp_ratio = self.mlp_ratio,
            num_layers = self.num_layers,
            dropout = self.dropout, 
            num_classes = num_classes,
            num_tasks = num_tasks,
            img_input_dim = vis_input_dim,
            uncertainty = self.uncertainty,
            dataset = self.dataset
        )

        self.action_decoder = ActionDecoder(
            embed_dim = embed_dim,
            time_horz = time_horz,
            att_heads = self.att_heads,
            mlp_ratio = self.mlp_ratio,
            num_layers = self.num_layers,
            dropout = self.dropout, 
            num_classes = num_classes,
            img_input_dim = vis_input_dim,
            num_tasks = num_tasks,
            uncertainty = self.uncertainty
        )

        self.task_decoder = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(embed_dim, num_tasks)
        )

        self.dropout = nn.Dropout(self.dropout)

        self.loss_action = nn.CrossEntropyLoss()
        self.loss_state = nn.CrossEntropyLoss()
        self.loss_state_pred = nn.MSELoss()
        self.loss_task = nn.CrossEntropyLoss()
        

    def forward(
            self, 
            visual_features, 
            state_prompt_features, 
            actions, 
            tasks, 
            transition_matrix=None
        ):
        '''Forward pass and loss calculation

        This function calls forward_once() to get the outputs, and then calls 
        forward_loss() to get processed labels and losses.

        Args:
            visual_features:        Visual observations of procedures.  [batch_size, time_horizon, 2, vis_input_dim]
            state_prompt_features:  Descriptions of before and after state of all actions. [num_action, num_prompts, lang_input_dim]
            actions:                Ground truth actions.     [batch_size, time_horizon]
            tasks:                  Ground truth tasks.       [batch_size]

        Returns:
            outputs: Dictionary of outputs.
            labels:  Dictionary of labels.
            losses:  Dictionary of losses.
        '''

        # forward network
        outputs = self.forward_once(
            visual_features, 
            state_prompt_features, 
            tasks,
        )

        batch_size, T = actions.shape
        action_logits = outputs["action"].reshape(batch_size, T, -1)
        action_logits = torch.softmax(action_logits, -1)

        # viterbi decoding
        if transition_matrix is not None:
            pred_viterbi = []
            for i in range(batch_size):
                viterbi_rst = viterbi_path(transition_matrix, action_logits[i].permute(1, 0).detach().cpu().numpy())
                pred_viterbi.append(torch.from_numpy(viterbi_rst))
            pred_viterbi = torch.stack(pred_viterbi).cuda()
        else:
            pred_viterbi = None
        outputs["pred_viterbi"] = pred_viterbi

        # loss calculation
        labels, losses = self.forward_loss(outputs, actions, tasks)

        return outputs, labels, losses
    
    def forward_once(
            self, 
            visual_features, 
            state_prompt_features, 
            tasks,
        ):
        '''Forward pass

        This function calls the state encoder, state decoder, task decoder, and
        action decoder to get the outputs.

        Args:
            visual_features:        Visual observations of procedures.  [batch_size, time_horizon, 2, vis_input_dim]
            state_prompt_features:  Descriptions of before and after state of all actions.     [num_action, num_prompts, lang_input_dim]
            tasks:                  Ground truth tasks.      [batch_size]
        
        Returns:
            outputs:                Dictionary of outputs.
        '''

        batch_size = visual_features.shape[0]

        # Step 1: state encoding
        state_feat_encode, inter_state_feat_encode, state_logits, state_prompt_features = \
            self.state_encoder(visual_features, state_prompt_features)

        # Step 2.1: task prediction
        state_feat_concat = state_feat_encode.reshape(batch_size, -1)
        task_pred = self.task_decoder(state_feat_concat)
        if self.training is False:
            tasks_input = task_pred.argmax(-1)
        else:
            tasks_input = tasks

        # Step 2.2: state decoding
        state_feat_decode = self.state_decoder(
            state_feat_encode,
            state_prompt_features, 
            task_pred
        )
        state_feat_input = state_feat_decode

        # Step 3: action decoding
        prompt_features = state_prompt_features
        action_logits = self.action_decoder(
            state_feat_input, 
            prompt_features, 
            tasks_input
        )

        # Collect outputs
        outputs = self.process_outputs(state_prompt_features, 
                                       state_logits, 
                                       state_feat_decode, 
                                       action_logits,
                                       task_pred)

        return outputs


    def forward_loss(self, outputs, actions, tasks):
        '''Loss calculation

        This function calculates the losses for state encoding, state decoding,
        action decoding, and task decoding.

        Args:
            outputs:    Dictionary of outputs.
            actions:    Ground truth actions.
            tasks:      Ground truth tasks.
        
        Returns:
            labels:     Dictionary of processed labels.
            losses:     Dictionary of losses.
        '''

        _, num_action = outputs["action"].shape
        embed_dim = self.embed_dim

        labels = self.process_labels(outputs, actions, tasks)

        losses = {}
        losses["state_encode"] = self.loss_state(
            outputs["state_encode"].reshape(-1, num_action), 
            labels["state"]
        )
        losses["state_decode"] = self.loss_state_pred(
            outputs["state_decode"].reshape(-1, embed_dim), 
            labels["state_decode"]
        )
        losses["action"] = self.loss_action(outputs["action"].reshape(-1, num_action), labels["action"])
        losses["task"] = self.loss_task(outputs["task"], labels["task"])

        return labels, losses


    def process_outputs(
            self, 
            state_prompt_features,
            state_logits, 
            state_feat_decode,
            action_logits,
            task_pred,
            pred_viterbi = None,
        ):
        '''Process outputs

        This function processes the outputs from the forward pass.

        Args:
            state_prompt_features: Descriptions of before and after state of all actions.   [num_action, num_prompts, embed_dim]
            state_logits:          Similarity between visual and linguistic features for start and end states.  [batch_size, 2, num_action]
            state_feat_decode:     Decoded features of all states.  [batch_size, time_horizon+1, embed_dim]
            action_logits:         Predicted action logits.  [batch_size, time_horizon, num_action]
            task_pred:             Predicted tasks.     [batch_size, num_tasks]
            pred_viterbi:          Predicted actions using viterbi decoding.    [batch_size, time_horizon]

        Returns:
            outputs: Dictionary of processed outputs.
        '''

        batch_size, _, num_action = state_logits.shape

        outputs = {}
        outputs["state_encode"] = state_logits.reshape(-1, num_action)
        outputs["state_decode"] = state_feat_decode[:, 1:-1, :]
        outputs["action"] = action_logits.reshape(-1, num_action)
        outputs["task"] = task_pred
        outputs["state_prompt_features"] = state_prompt_features
        outputs["pred_viterbi"] = pred_viterbi

        return outputs


    def process_labels(self, outputs, actions, tasks):
        labels = {}
        labels["state"] = actions[:, [0, -1]].reshape(-1)
        labels["action"] = actions.reshape(-1)
        labels["task"] = tasks
        labels["state_decode"] = self.process_state_prompts(outputs["state_prompt_features"], actions)

        return labels


    def process_state_prompts(self, state_prompt_features, actions):
        '''Process state prompts

        This function combines the language descriptions after the current action with
        the descriptions before the next action to get consistent descriptions for 
        each state.

`       Args:
            state_prompt_features: Descriptions of before and after state of all actions.   [num_action, num_prompts, embed_dim]
            actions:               Ground truth actions.    [batch_size, time_horizon]
        
        Returns:
            target_state_decode:   Reduced descriptions for each state.     [batch_size*(time_horizon-1), embed_dim]
        '''

        batch_size, time_horizon = actions.shape
        num_action, num_desc, embed_dim = state_prompt_features.shape
        actions = actions.reshape(-1)   # [batch_size*time_horizon]
        state_prompt_features = state_prompt_features[actions, :, :].reshape(batch_size, time_horizon, num_desc, embed_dim)

        before_state_prompt_feat = torch.cat([state_prompt_features[:, :, :num_desc//2, :], 
                                              state_prompt_features[:, -1:, num_desc//2:, : ]], 1)  # [batch_size, time_horizon+1, 3, embed_dim]
        
        after_state_prompt_feat  = torch.cat([state_prompt_features[:, :1, :num_desc//2, :], 
                                              state_prompt_features[:, :, num_desc//2:, :]], 1)     # [batch_size, time_horizon+1, 3, embed_dim]
    
        target_state_decode = torch.cat([before_state_prompt_feat, after_state_prompt_feat], 2)     # [batch_size, time_horizon+1, 6, embed_dim]
        target_state_decode = target_state_decode.mean(2)[:, 1:-1, :].reshape(-1, embed_dim)        # [batch_size*(time_horizon-1), embed_dim]

        return target_state_decode.clone().detach()

import torch.nn as nn
import torch

from models.state_encoder import StateEncoder
from models.state_decoder import StateDecoder
from models.action_decoder import ActionDecoder
from models.utils import viterbi_path, img_text_similarlity

class ProcedureModel(nn.Module):
    def __init__(
            self, 
            vis_input_dim,
            lang_input_dim,
            embed_dim, 
            time_horz, 
            num_classes,
            num_tasks,
            args,
            vis_mode = False,  
        ):
        super().__init__()

        self.att_heads = args.attn_heads
        self.mlp_ratio = args.mlp_ratio
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.use_task = not args.no_task
        self.use_state_pred = not args.no_state_pred
        self.use_state_memory = not args.no_state_memory
        self.use_action_memory = args.use_action_memory
        self.use_observ_memory = args.use_observ_memory
        self.use_action_proj_loss = not args.no_action_proj_loss
        self.use_random = args.use_random
        self.use_state_decoder = not args.no_state_decoder
        self.mid_state_align = args.fully_supervised
        self.uncertainty = args.uncertain
        self.vis_mode = vis_mode
        self.dataset = args.dataset

        self.time_horz = time_horz
        self.embed_dim = embed_dim

        self.state_encoder = StateEncoder(
            vis_input_dim,
            lang_input_dim,
            embed_dim, 
            dropout = 0.4,
            vis_mode = vis_mode,
            mid_state_align=self.mid_state_align
        )

        if self.use_state_pred & self.use_state_decoder:
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
                use_state_memory = self.use_state_memory,
                use_random = self.use_random,
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
            use_task = self.use_task,
            use_state_pred = self.use_state_pred,
            use_state_memory = self.use_state_memory,
            use_action_memory = self.use_action_memory,
            use_random = self.use_random,
            use_state_decoder = self.use_state_decoder,
            uncertainty = self.uncertainty
        )

        if self.use_task:
            self.task_decoder = nn.Sequential(
                nn.Linear(embed_dim*2, embed_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(embed_dim, num_tasks)
            )

        self.dropout = nn.Dropout(self.dropout)

        self.loss_action = nn.CrossEntropyLoss()
        self.loss_state = nn.CrossEntropyLoss()

        if self.use_state_pred:
            self.loss_state_pred = nn.MSELoss()
        if self.use_task:
            self.loss_task = nn.CrossEntropyLoss()
        
        self.loss_recon = nn.MSELoss()

    def forward(
            self, 
            visual_features, 
            state_prompt_features, 
            action_prompt_features, 
            observation_features,
            actions, 
            tasks, 
            transition_matrix=None
        ):
        '''
        visual_features: [batch_size, time_horizon, 2, embed_dim]
        state_prompt_features: [batch_size, num_action, num_prompts, embed_dim]
        action_prompt_features: [batch_size, num_action, embed_dim]
        observation_features: [num_observ, embed_dim]
        actions: [batch, time_horizon]
        tasks:   [batch]
        '''

        if self.use_random:
            outputs_action = 0
            outputs_action_proj = 0
            Times = 3
            for i in range(Times):
                outputs = self.forward_once(
                    visual_features, 
                    state_prompt_features, 
                    action_prompt_features,
                    observation_features, 
                    actions, 
                    tasks,
                )
                outputs_action += outputs["action"]
            outputs["action"] = outputs_action / Times
        else:
            outputs = self.forward_once(
                visual_features, 
                state_prompt_features, 
                action_prompt_features, 
                observation_features,
                actions, 
                tasks,
            )

        batch_size, T = actions.shape
        action_logits = outputs["action"].reshape(batch_size, T, -1)
        action_logits = torch.softmax(action_logits, -1)

        # viterbi
        if transition_matrix is not None:
            pred_viterbi = []
            for i in range(batch_size):
                viterbi_rst = viterbi_path(transition_matrix, action_logits[i].permute(1, 0).detach().cpu().numpy())
                pred_viterbi.append(torch.from_numpy(viterbi_rst))
            pred_viterbi = torch.stack(pred_viterbi).cuda()
        else:
            pred_viterbi = None

        if pred_viterbi is not None:
            outputs["pred_viterbi"] = pred_viterbi

        labels, losses = self.forward_loss(outputs, actions, tasks)

        return outputs, labels, losses
    
    def forward_once(
            self, 
            visual_features, 
            state_prompt_features, 
            action_prompt_features, 
            observation_features,
            actions, 
            tasks,
        ):
        '''
        input:
        state_feature: [batch_size, time_horizon, 2, embed_dim]
        actions: [batch, time_horizon]
        state_prompt_features: [batch_size, num_action, num_prompts, embed_dim]
        action_prompt_features: [batch_size, num_action, embed_dim]
        '''
        batch_size = visual_features.shape[0]

        # Step 1: state encoding
        if self.vis_mode is False:
            state_feat_encode, inter_state_feat_encode, state_logits, state_prompt_features = \
                self.state_encoder(visual_features, state_prompt_features)
            state_desc_logits = None
        else:
            state_feat_encode, inter_state_feat_encode, state_logits, state_desc_logits, state_prompt_features = \
                self.state_encoder(visual_features, state_prompt_features)

        action_prompt_features = self.state_encoder.desc_encoder(action_prompt_features)

        # replace visual feature with language feature
        # start_desc_feature = state_prompt_features[actions[:, 0], :3, :].mean(1, keepdim=True)
        # end_desc_feature = state_prompt_features[actions[:, -1], 3:, :].mean(1, keepdim=True)
        # state_feat_encode = torch.cat([start_desc_feature, state_feat_encode[:, 1:-1, :], end_desc_feature], 1)

        # Step 2.1: task prediction
        if self.use_task:
            # task prediction
            state_feat_concat = state_feat_encode.reshape(batch_size, -1)
            task_pred = self.task_decoder(state_feat_concat)
        else:
            task_pred = None

        if not self.training and self.use_task:
            tasks_input = task_pred.argmax(-1)
        else:
            tasks_input = tasks

        # Step 2.2: state decoding
        if self.use_state_pred & self.use_state_decoder:
            if self.use_action_memory:
                state_feat_decode = self.state_decoder(
                    state_feat_encode,
                    action_prompt_features, 
                    tasks_input
                )
            else:
                ## Using predicted task logits for crosstask increases performance
                if self.dataset == "crosstask":
                    state_feat_decode = self.state_decoder(
                        state_feat_encode,
                        state_prompt_features, 
                        task_pred
                    )
                else:
                    state_feat_decode = self.state_decoder(
                        state_feat_encode,
                        state_prompt_features, 
                        tasks_input
                    )
            state_feat_input = state_feat_decode

        else:
            state_feat_decode = None
            state_feat_input = state_feat_concat.reshape(batch_size, 2, self.embed_dim)

        # Step 3: action decoding
        if self.use_action_memory:
            prompt_features = action_prompt_features
        else:
            prompt_features = state_prompt_features
        
        action_outputs = self.action_decoder(
            state_feat_input, 
            prompt_features, 
            tasks_input
        )
        if self.use_state_decoder:
            action_logits, action_decode = action_outputs
        else:
            state_feat_decode, action_logits, action_decode = action_outputs

        action_logits, action_proj_logits = \
            self.process_action_outputs(
                action_logits, 
                action_decode, 
                action_prompt_features
            )

        outputs = self.process_outputs(state_prompt_features, 
                                       state_logits, 
                                       state_desc_logits,
                                       inter_state_feat_encode,
                                       state_feat_decode, 
                                       action_logits, 
                                       action_proj_logits, 
                                       task_pred)

        return outputs


    def forward_loss(self, outputs, actions, tasks):
        '''
        input:
        actions: [batch_size*time_horizon, num_action]
        tasks:   [batch]
        '''

        _, num_action = outputs["action"].shape
        embed_dim = self.embed_dim

        labels = self.process_labels(outputs, actions, tasks)

        losses = {}
        losses["state_encode"] = self.loss_state(
            outputs["state_encode"].reshape(-1, num_action), 
            labels["state"]
        )
        if self.use_state_pred:
            losses["state_decode"] = self.loss_state_pred(
                outputs["state_decode"].reshape(-1, embed_dim), 
                labels["state_decode"]
            )
            losses["state_recon"] = self.loss_recon(
                outputs["inter_states"].reshape(-1, embed_dim), 
                outputs["state_decode"].reshape(-1, embed_dim)
            )

        losses["action"] = self.loss_action(outputs["action"].reshape(-1, num_action), labels["action"])
        losses["action_proj"] = self.loss_action(outputs["action_proj"].reshape(-1, num_action), labels["action"])

        if self.use_task:
            losses["task"] = self.loss_task(outputs["task"], labels["task"])

        return labels, losses


    def process_action_outputs(
            self,
            action_logits,
            action_decode,
            action_prompt_features,
        ):
        if self.use_state_pred:
            # index of action/state
            action_idx = list(range(1, self.time_horz*2, 2))
        else:
            action_idx = list(range(1, self.time_horz+1, 1))

        action_logits = action_logits[:, action_idx, :]
        # action contrastive logits
        action_proj = self.action_decoder.proj(action_decode)
        action_proj = action_proj[:, action_idx, :]
        action_proj_logits = img_text_similarlity(
            action_proj, 
            action_prompt_features.clone().detach().unsqueeze(0), 
            self.action_decoder.logit_scale.exp()
        )
        return action_logits, action_proj_logits


    def process_outputs(
            self, 
            state_prompt_features,
            state_logits, 
            state_desc_logits,
            inter_state_feat_encode,
            state_feat_decode,
            action_logits, 
            action_proj_logits,
            task_pred,
            pred_viterbi = None,
        ):

        batch_size, _, num_action = state_logits.shape

        outputs = {}
        outputs["state_encode"] = state_logits.reshape(-1, num_action)
        outputs["state_desc"] = state_desc_logits
        if self.use_state_pred:
            outputs["state_decode"] = state_feat_decode[:, 1:-1, :]
            outputs["inter_states"] = inter_state_feat_encode

        outputs["action"] = action_logits.reshape(-1, num_action)
        outputs["action_proj"] = action_proj_logits.reshape(-1, num_action)

        if self.use_task:
            outputs["task"] = task_pred

        outputs["state_prompt_features"] = state_prompt_features

        if pred_viterbi is not None:
            outputs["pred_viterbi"] = pred_viterbi

        return outputs


    def process_labels(self, outputs, actions, tasks):
        labels = {}
        if self.mid_state_align is False:   #default
            labels["state"] = actions[:, [0, -1]].reshape(-1)
        else:
            labels["state"] = torch.repeat_interleave(actions, 2, dim=1).reshape(-1)
            
        labels["action"] = actions.reshape(-1)
        labels["task"] = tasks

        state_prompt_features = outputs["state_prompt_features"]
        labels["state_decode"] = self.process_state_prompts(state_prompt_features, actions)

        return labels


    def process_state_prompts(self, state_prompt_features, actions):

        '''
        input:
        state_prompt_features: [num_action, num_desc, embed_dim]
        actions:   [batch_size, time_horizon]
        output:
        target_state_decode: [batch_size*(time_horizon+1), embed_dim]
        '''

        batch_size, time_horizon = actions.shape
        num_action, num_desc, embed_dim = state_prompt_features.shape
        actions = actions.reshape(-1) # [batch_size*time_horizon]
        state_prompt_features = state_prompt_features[actions, :, :].reshape(batch_size, time_horizon, num_desc, embed_dim)

        before_state_prompt_feat = torch.cat([state_prompt_features[:, :, :num_desc//2, :], 
                                              state_prompt_features[:, -1:, num_desc//2:, : ]], 1) # [batch_size, time_horizon+1, 3, embed_dim]
        
        after_state_prompt_feat  = torch.cat([state_prompt_features[:, :1, :num_desc//2, :], 
                                              state_prompt_features[:, :, num_desc//2:, :]], 1) # [batch_size, time_horizon+1, 3, embed_dim]
    
        target_state_decode = torch.cat([before_state_prompt_feat, after_state_prompt_feat], 2) # [batch_size, time_horizon+1, 6, embed_dim]
        target_state_decode = target_state_decode.mean(2)[:, 1:-1, :].reshape(-1, embed_dim) # [batch_size*(time_horizon-1), embed_dim]

        return target_state_decode.clone().detach()

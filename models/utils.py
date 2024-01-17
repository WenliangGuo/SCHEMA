import torch
import torch.nn.functional as F
import numpy as np

def process_state_feat(state_feat):
    '''
    input:
    state_feat: (batch, time_horz, 2, input_dim)
    output:
    state_feat_tmp: (batch, time_horz+1, input_dim)
    '''
    b, horizon, num_obser, dim = state_feat.shape

    state_feat_tmp_1 = torch.cat([state_feat[:, :, 0:1, :], state_feat[:, -1:, -1:, :]], 1) # [batch, time_horz+1, 1, dim]
    state_feat_tmp_2 = torch.cat([state_feat[:, 0:1, 0:1, :], state_feat[:, :, -1:, :]], 1)
    state_feat_tmp = torch.cat([state_feat_tmp_1, state_feat_tmp_2], 2) # [batch, time_horz+1, 2, dim]
    state_feat_tmp = state_feat_tmp.mean(2) # [batch, time_horz+1, dim]

    return state_feat_tmp

def img_text_similarlity(state_features, prompt_features, scale):
        '''
        input:
        state_features: (batch, length, embedding_dim)
        prompt_features: (batch, length, embedding_dim)
        scale: float
        output:
        logits_per_image: (batch, batch)
        logits_per_text: (batch, batch)
        '''
        embedding_dim = state_features.shape[-1]
        
        # flatten features
        state_features = state_features.reshape(-1, embedding_dim)
        prompt_features = prompt_features.reshape(-1, embedding_dim)

        # normalized features
        image_features = state_features / state_features.norm(dim=1, keepdim=True) #.detach().clone()
        text_features = prompt_features / prompt_features.norm(dim=1, keepdim=True) #.detach().clone()
        # image_features = F.normalize(state_features)
        # text_features = F.normalize(prompt_features)

        # cosine similarity as logits
        logits = scale * image_features @ text_features.t()

        return logits

def viterbi_path(transition, emission, prior=None, observation=None, return_likelihood=False):
    '''
    Args:
        transition:     the transition matrix, where A[i][j] is the probability of transitioning from state i to state j.
        emission:       the emission matrix, where B[i][j] is the probability of emitting observation j from state i.
        prior:          the prior probabilities, where pi[i] is the probability of starting in state i.
        observation:    the sequence of observations.
        best_path:      the most likely action sequence
    
    Shape:
        transition:     (num_action, num_action)
        emission:       (num_action, horizon)
        prior:          (num_action)
        observation:    (horizon)
        best_path:      (horizon)
    '''

    # Initialize trellis
    T = emission.shape[1]                       # time horizon
    N = transition.shape[0]                     # number of actions

    if observation is None:
        observation = np.arange(T)
    
    if prior is None:
        prior = np.ones((N,), dtype=np.float32) / N

    trellis = np.zeros((T, N), dtype=np.float32)       # store the probabilities of each state at each time step
    backpointers = np.zeros((T, N), dtype=np.int32)    # store the indices of the most likely previous state at each time step
    
    # Calculate probabilities for first time step
    trellis[0] = prior * emission[:, observation[0]]
    
    # Calculate probabilities for subsequent time steps
    for t in range(1, T):
        temp = trellis[t-1].reshape((N, 1)) * transition
        trellis[t] = emission[:, observation[t]] * np.max(temp, axis=0)
        backpointers[t] = np.argmax(temp, axis=0)
    
    # Backtrack to find most likely sequence of hidden states
    best_path_prob = np.max(trellis[-1])
    best_path_pointer = np.argmax(trellis[-1])
    best_path = [best_path_pointer]
    for t in range(T-1, 0, -1):
        best_path_pointer = backpointers[t][best_path_pointer]
        best_path.insert(0, best_path_pointer)
    
    best_path = np.array(best_path)
    
    if return_likelihood:
        return best_path, best_path_prob
    else:
        return best_path
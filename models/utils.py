import numpy as np

def img_text_similarlity(state_features, prompt_features, scale):
        ''' Compute the similarity between visual and linguistic features

        Args:
            state_features:     Input visual feature.   (batch, length, embedding_dim)
            prompt_features:    Input language feature. (batch, length, embedding_dim)
            scale:              Scale parameter.

        Returns:
            logits:             Similarity matrix.      (batch, length, length)
        '''

        embedding_dim = state_features.shape[-1]
        
        # flatten features
        state_features = state_features.reshape(-1, embedding_dim)
        prompt_features = prompt_features.reshape(-1, embedding_dim)

        # normalized features
        image_features = state_features / state_features.norm(dim=1, keepdim=True)
        text_features = prompt_features / prompt_features.norm(dim=1, keepdim=True)

        # similarity as logits
        logits = scale * image_features @ text_features.t()
        return logits


def viterbi_path(transition, emission, prior=None, observation=None, return_likelihood=False):
    ''' Viterbi algorithm

    Search the most likely sequence of hidden states given the observations.

    Args:
        transition:     Transition matrix, where A[i][j] is the probability of 
                        transitioning from state i to state j.  (num_action, num_action)
        emission:       Emission matrix, where B[i][j] is the probability of 
                        emitting observation j from state i.    (num_action, horizon)
        prior:          Prior probabilities, where pi[i] is the probability of 
                        starting in state i.    (num_action)
        observation:    Sequence of observations.   (horizon)
        return_likelihood:  Whether to return the likelihood of the best path.  (default: False)
    
    Returns:
        best_path:      The most likely action sequence.    (horizon)
        best_path_prob: The likelihood of the best path.
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


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
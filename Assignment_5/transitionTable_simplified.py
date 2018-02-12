import numpy as np
from scipy.stats import rv_discrete

class TransitionTable:

    # basic funcs

    def __init__(self, state_siz, act_num, hist_len,
                       minibatch_size, max_transitions, alpha):
        self.state_siz = state_siz
        self.act_num = act_num
        self.hist_len  = hist_len
        self.batch_size = minibatch_size
        self.max_transitions = max_transitions
        
        self.alpha = alpha

        # memory for state transitions
        self.states  = np.zeros((max_transitions, state_siz*hist_len))
        self.actions = np.zeros((max_transitions, act_num))
        self.next_states = np.zeros((max_transitions, state_siz*hist_len))
        self.rewards = np.zeros((max_transitions, 1))
        self.terminal = np.zeros((max_transitions, 1))
        self.tds = np.zeros((max_transitions, 1))
        self.size = 0

        x_vals = np.arange(1,len(self.tds)+1)
        rank_based_probs = (1./x_vals)**alpha/sum((1./x_vals)**alpha)
        self.power_law_model = rv_discrete(name='power_law_model', values=(x_vals,rank_based_probs))

    # helper funcs
    def add(self, state, action, next_state, reward, terminal, td_err):
        if self.size == self.max_transitions:
            # find transition with lowest probability and delete it
            idx_min = np.argmin(self.tds)
            self.states[idx_min] = state
            self.actions[idx_min] = action
            self.next_states[idx_min] = next_state
            self.rewards[idx_min] = reward
            self.terminal[idx_min] = terminal
            self.tds[idx_min] = td_err
        else:
            self.states[self.size] = state
            self.actions[self.size] = action
            self.next_states[self.size] = next_state
            self.rewards[self.size] = reward
            self.terminal[self.size] = terminal
            self.tds[self.size] = td_err     
            self.size += 1

    def one_hot_action(self, actions):
        actions = np.atleast_1d(actions)
        one_hot_actions = np.zeros((actions.shape[0], self.act_num))
        for i in range(len(actions)):
            one_hot_actions[i, int(actions[i])] = 1
        return one_hot_actions

    def sample_minibatch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
            
        x_vals = np.arange(1,self.size+1)
        rank_based_probs = (1./x_vals)**self.alpha/sum((1./x_vals)**self.alpha)
        power_law_model = rv_discrete(name='power_law_model', values=(x_vals,rank_based_probs))
        
        sort_idx = np.argsort(self.tds[:self.size])[-1::-1]
        idx = power_law_model.rvs(size=batch_size)-1
        batch_idx = sort_idx[idx]
        
        state = self.states[batch_idx]
        action = self.actions[batch_idx]
        next_state = self.next_states[batch_idx]
        reward = self.rewards[batch_idx]
        terminal = self.terminal[batch_idx]
        
        return state, action, next_state, reward, terminal



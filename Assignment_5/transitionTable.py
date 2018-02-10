import numpy as np

class Transition():
    def __init__(self, state, action, reward, next_state, terminal, weight, td):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.terminal = terminal
        self.weight = weight
        self.td = td

class TransitionTable:

    # basic funcs

    def __init__(self, state_siz, act_num, hist_len,
                       minibatch_size, max_transitions):
        self.state_siz = state_siz
        self.act_num = act_num
        self.hist_len  = hist_len
        self.batch_size = minibatch_size
        self.max_transitions = max_transitions
        self.alpha = 1.
        self.beta = 1.

        #transition memory
        self.transitions = []
        self.max_td = 1e6
        self.top = 0
        self.bottom = 0
        self.size = 0

    # Updates weights
    def update_weights(self):
        priority_sum = sum((1./i) ** self.alpha for i in range(1, self.size+1))
        assert priority_sum != 0
        self.transitions = sorted(self.transitions[self.bottom:self.top], key=lambda transition: -transition.td)

        for i in range(len(self.transitions)):
            self.transitions[i].weight = (1./self.size * 1./(((1./(i+1.)) ** self.alpha) / priority_sum)) ** self.beta

    # New transitions get maximal priority (=td error)
    def add(self, state, action, next_state, reward, terminal):
        if self.size == self.max_transitions:
            self.bottom = (self.bottom + 1) % self.max_transitions
        else:
            self.size += 1

        weight = (1./self.size * sum((1./i) ** self.alpha for i in range(1, self.size+1))) ** self.beta
        self.transitions.append(Transition(state=state, action=action, reward=reward, next_state=next_state, terminal=terminal, weight=weight, td = self.max_td))
        self.top = (self.top + 1) % self.max_transitions
        self.update_weights()

    def one_hot_action(self, actions):
        actions = np.atleast_2d(actions)
        one_hot_actions = np.zeros((actions.shape[0], self.act_num))
        for i in range(len(actions)):
            one_hot_actions[i, int(actions[i])] = 1
        return one_hot_actions

    # Minibatch from prioritized replay
    # We priorize rank-based
    def sample_minibatch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        # Sample a batch with the priorities as probabilities
        priority_sum = sum((1./i) ** self.alpha for i in range(1, self.size+1))
        probabilities = np.zeros(self.size)

        # P(i) = ((1/rank(i)) ** \alpha) / (sum_i((1/rank(i)) ** \alpha))
        for i in range(len(probabilities)):
            probabilities[i] = ((1./(i+1.)) ** self.alpha) / priority_sum

        # Sort transitions according to rank
        self.transitions = sorted(self.transitions[self.bottom:self.top], key=lambda transition: -transition.td)

        # Sample batch
        batch_indices = np.random.choice(self.size, batch_size, p=probabilities)
        batch = [None] * batch_size
        weight_vector = np.zeros(batch_size)

        # Get weights and normalize
        for i in range(batch_size):
            if self.bottom + batch_indices[i] > self.top:
                print("Index {} is too big for size {}. Highest index will be used." .format(self.bottom + batch_indices[i], self.top))
                batch[i] = self.transitions[self.top]
            else:
                batch[i] = self.transitions[self.bottom + batch_indices[i]]
            weight_vector[i] = batch[i].weight

        normalized_weights = 1/max(weight_vector) * weight_vector

        return batch, normalized_weights

    # Update td error values
    def update_td(self, batch, tds):
        for i, t in enumerate(batch):
            index = self.transitions.index(t)
            self.transitions[index].td = tds[i]

            if tds[i] > self.max_td:
                self.max_td = tds[i]

        self.update_weights()

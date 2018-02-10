import numpy as np
import matplotlib
#matplotlib.use('Agg')
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randrange
from collections import namedtuple
import argparse
import os
import time
import datetime

tf.reset_default_graph() 

# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator
from transitionTable import TransitionTable

#Statistics = namedtuple("Stats",["loss","loss_stamp","avg_episode_len","avg_episode_len_stamp"])
"""
# Parse arguments from the command line
parser = argparse.ArgumentParser(description='Uses DQN to navigate a maze.')
parser.add_argument('-d' , help="set discount factor (default = 0.9)")
parser.add_argument('-eps' , help="set starting epsilon (default = 0.3)")
parser.add_argument('-noise' , help="use noisy nets (default False)")
parser.add_argument('-units' , help="set number of units per layer (default 256)")
parser.add_argument('-filters1' , help="set number of filters of convlayer1 (default 16)")
parser.add_argument('-filter_size1' , help="set filter size of convlayer1 (default 8)")
parser.add_argument('-filters2' , help="set number of filters of convlayer2 (default 32)")
parser.add_argument('-filter_size2' , help="set filter size of convlayer2 (default 4)")
parser.add_argument('-pool_size' , help="set pool size (default 2)")
parser.add_argument('-keep_prob' , help="keep probability used in dropout layer (default 0.5)")
parser.add_argument('-steps' , help="set number of episodes (default 1e5)")
parser.add_argument('-lr' , help="set learning rate (default 0.0005)")
parser.add_argument('-path' , help="output path for plots (default .)")
parser.add_argument('-plt_show' , help="show plots or not (default False)")
args = parser.parse_args()

# Set parameters from command line or default
DISCOUNT_FACTOR = args.d if args.d != None else 0.9
EPSILON = args.eps if args.eps != None else 0.3
NOISY = args.noise if args.noise != None else False
UNITS = args.units if args.units != None else 256
FILTERS1 = args.filters1 if args.filters1 != None else 16
FILTER_SIZE1 = args.filter_size1 if args.filter_size1 != None else 8
FILTERS2 = args.filters2 if args.filters2 != None else 32
FILTER_SIZE2 = args.filter_size2 if args.filter_size2 != None else 4
POOL_SIZE = args.pool_size if args.pool_size != None else 2
KEEP_PROB = args.keep_prob if args.keep_prob != None else 0.5
STEPS = args.steps if args.steps != None else 1e4
LEARNING_RATE = args.lr if args.lr != None else 5e-4
PATH = args.path if args.path != None else os.path.join(os.getcwd(),"data")
PLT_SHOW = args.plt_show if args.plt_show != None else False
"""

""" Testing """
DISCOUNT_FACTOR = 0.9
EPSILON = 0.3
NOISY = False
UNITS = 256
FILTERS1 = 16
FILTER_SIZE1 = 8
FILTERS2 = 32
FILTER_SIZE2 = 4
POOL_SIZE = 2
KEEP_PROB = 0.5
STEPS = 2e5
LEARNING_RATE = 5e-4
PATH = os.path.join(os.getcwd(),"data")
PLT_SHOW = False
""" """

class Agent:
    "Neural Network agent in tensorflow"
    def __init__(self, opt, scope="agent"):
        # Identify the Network
        self.scope = scope
        with tf.variable_scope(scope):
            self.model = self.build_model(opt)

    "Build the model"
    def build_model(self, opt):
        self.x = tf.placeholder(tf.float32, shape=[None, opt.cub_siz*opt.pob_siz, opt.cub_siz*opt.pob_siz, opt.hist_len])
        
        # Placeholder for keep probability (= 1- droput rate)
        self.keep_prob = tf.placeholder(tf.float32)

        # Placeholders for loss
        self.targets = tf.placeholder(tf.float32, shape = [None])
        self.action_onehot = tf.placeholder(tf.float32, shape=[None, opt.act_num])
        self.normalized_is_weights = tf.placeholder(tf.float32, shape=[None])

        # L2 Regularizer
        self.l2 = tf.contrib.layers.l2_regularizer(0.1)

        # Convolutional layers (padded to size)
        self.conv1 = tf.contrib.layers.conv2d(self.x, FILTERS1, FILTER_SIZE1)
        self.pool1 = tf.contrib.layers.max_pool2d(self.conv1, POOL_SIZE)
        self.conv2 = tf.contrib.layers.conv2d(self.pool1, FILTERS2, FILTER_SIZE2)
        self.pool2 = tf.contrib.layers.max_pool2d(self.conv2, POOL_SIZE)

        # Flatten before hidden layers
        self.flatten = tf.contrib.layers.flatten(self.pool2)

        # Hidden layers
        #TODO: This is super clunky... Can we make it prettier?
        if NOISY == True:
            # Noisy layers: y = Wx + b + (W_noise \times \eps_w)x + b_noise \times \eps_b
            self.w1 = tf.get_variable('weights_hidden1', shape=[self.flatten.shape[1], UNITS])
            self.b1 = tf.get_variable('bias_hidden1', shape=[UNITS])

            self.noise_w1 = tf.get_variable('noise_weights_hidden1', shape=[self.flatten.shape[1], UNITS])
            self.noise_b1 = tf.get_variable('noise_bias_hidden1', shape=[UNITS])

            # Independent Gaussian noise
            self.eps_w1 = tf.random_normal(self.w1.shape, 0., 0.5)
            self.eps_b1 = tf.random_normal([UNITS], 0, 0.5)

            self.hidden1 = tf.nn.relu(tf.matmul(self.flatten, self.w1) + self.b1 + tf.matmul(self.flatten, np.multiply(self.noise_w1, self.eps_w1)) + np.multiply(self.noise_b1, self.eps_b1))

            self.dropout1 = tf.contrib.layers.dropout(self.hidden1)

            self.w2 = tf.get_variable('weights_hidden2', shape=[self.dropout1.shape[1], UNITS])
            self.b2 = tf.get_variable('bias_hidden2', shape=[UNITS])

            self.noise_w2 = tf.get_variable('noise_weights_hidden2', shape=[self.dropout1.shape[1], UNITS])
            self.noise_b2 = tf.get_variable('noise_bias_hidden2', shape=[UNITS])

            # Independent Gaussian noise
            self.eps_w2 = tf.random_normal(self.w2.shape, 0., 0.5)
            self.eps_b2 = tf.random_normal([UNITS], 0, 0.5)

            self.hidden2 = tf.nn.relu(tf.matmul(self.dropout1, self.w2) + self.b2 + tf.matmul(self.dropout1, np.multiply(self.noise_w2, self.eps_w2)) + np.multiply(self.noise_b2, self.eps_b2))

            self.dropout2 = tf.contrib.layers.dropout(self.hidden2)

        else:
            self.hidden1 = tf.contrib.layers.fully_connected(self.flatten, UNITS, weights_regularizer=self.l2, activation_fn=tf.nn.relu)
            self.dropout2 = tf.nn.dropout(self.hidden1,self.keep_prob)

        # Linear output
        self.out = tf.contrib.layers.fully_connected(self.dropout2, opt.act_num, activation_fn=None)

        self.selected_q = tf.reduce_sum(self.action_onehot * self.out, 1)

        # We need to factor in the importance sampling weights here to avoid a sampling bias
        # The weight for a transition i is w_i = 1/(N*P(i))^\beta with P(i) being the priority of i
        # Source: Schaul, Quan, Antonoglou, Silver - Prioritized Experience Replay (2016)
        self.tds = self.targets - self.selected_q
        self.loss = tf.reduce_mean(np.square(np.multiply(self.normalized_is_weights, self.tds)))

        #Adam optimizer
        self.train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

        pass

    def predict(self, sess, states):
        """
        Returns agent prediction
        """
        return sess.run([self.out], feed_dict = {self.x: states, self.keep_prob: 1.})

    def update(self, sess, state_batch, targets, action_batch, is_weights, keep_p):
        """
        Trains the network
        """
        self.train_step.run(feed_dict={self.x: state_batch, self.targets: targets, self.action_onehot: action_batch, self.normalized_is_weights: is_weights, self.keep_prob: keep_p}, session = sess)
        return sess.run([self.loss, self.tds], feed_dict={self.x: state_batch, self.targets: targets, self.action_onehot: action_batch, self.normalized_is_weights: is_weights,self.keep_prob:1.})

class TargetNetwork(Agent):
    """
    Slowly updated target network. Tau indicates the speed of adjustment. If 1,
    it is always set to the values of its associate.
    """
    def __init__(self, opt, tau=0.001):
        self.scope = "target"
        Agent.__init__(self, opt, self.scope)
        self.tau = tau
        self._associate = self._register_associate()

    def _register_associate(self):
      op_holder = []
      # All variables outside the target scope, sorted by name
      agent_vars = [var for var in tf.trainable_variables() if not var.name.startswith(self.scope)]
      agent_vars = sorted(agent_vars, key=lambda var: var.name)

      # All target network variables, sorted by name
      target_vars = [var for var in tf.trainable_variables() if var.name.startswith(self.scope)]
      target_vars = sorted(target_vars, key=lambda var: var.name)

      # Go tau sized step towards the agent variable values
      for agent_var, target_var in zip(agent_vars, target_vars):
          op_holder.append(target_var.assign(self.tau * agent_var.value() + (1-self.tau) * target_var.value()))

      return op_holder

    def update(self, sess):
      for op in self._associate:
        sess.run(op)

def append_to_hist(state, obs):
    """
    Add observation to the state.
    """
    for i in range(state.shape[0]-1):
        state[i, :] = state[i+1, :]
    state[-1, :] = obs

def plot_stats(stats):
    # Plot loss over time
    fig1 = plt.figure(figsize=(10,10))
    plt.plot(stats["loss_stamp"],stats["loss"])
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Loss over step")

    # Plot successes in in-between tests
    fig2 = plt.figure(figsize=(10,10))
    plt.plot(stats["avg_episode_len_stamp"],stats["avg_episode_len"])
    plt.xlabel("Training episode")
    plt.ylabel("Average number of steps per episode")
    plt.title("Average episode length during training")

    """
    # Plot test steps
    fig3 = plt.figure(figsize=(10,10))
    plt.plot(stats.tests)
    plt.xlabel("Test Number")
    plt.ylabel("Steps used")
    plt.title("Steps per test")
    """
    if PATH != None:
        if not os.path.exists(PATH):
            os.makedirs(PATH)

        fig1.savefig(PATH + "/loss.png")
        fig2.savefig(PATH + "/avg_episode_len.png")
        #fig3.savefig(PATH + "/test_steps.png")
    else:
        fig1.savefig('loss.png')
        fig2.savefig('avg_episode_len.png')
        #fig3.savefig('test_steps.png')

    if PLT_SHOW:
        plt.show(fig1)
        plt.show(fig2)
        #plt.show(fig3)


# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
test_sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)

"""
Setup a large transitiontable that is filled during training
It will save the last third of the total number of episodes
This is a relatively small number and might be increased to a more complete representation of possible transitions
The reason it's relatively small here is that we prioritize large errors anyway and don't want to train on the same ones the whole time
This could be considered a hyperparameter that needs to be tuned
"""

#maxlen = int(EPISODES) // 3
maxlen = 100000

trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                        opt.minibatch_size, maxlen)
agent = Agent(opt)
target = TargetNetwork(opt)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#opt.disp_on = True

if opt.disp_on:
    win_all = None
    win_pob = None

steps = int(STEPS)

# Parameter for epsilon schedule (greedy choice of action)
epsilon_decay = 0.999

show_image = False
show_image_interval = [20,5] #every 20th episode, show 5 episodes
show_image_count = show_image_interval[0]
numbers_games_in_bins = 10
step_num_vec_temp = np.array([])

epi_step = 0
nepisodes = 0
episode_ended = False
show_avg_steps = False
stats = {"loss": np.array([]), "loss_stamp": np.array([]), "avg_episode_len": np.array([]), "avg_episode_len_stamp": np.array([])}

state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)

n_step_return = np.zeros(opt.early_stop)
save_states = np.zeros((opt.early_stop, opt.hist_len*opt.state_siz))
save_next_states = np.zeros((opt.early_stop, opt.hist_len*opt.state_siz))
save_actions = np.zeros((opt.early_stop, opt.act_num))
save_terminals = np.zeros_like(n_step_return)

start_time = time.time()
loss = 0

for step in range(steps):
    if state.terminal or epi_step >= opt.early_stop:
        # add to the transition table
        for i in range(epi_step):
            #sum backwards
            trans.add(save_states[i], save_actions[i], save_next_states[i], n_step_return[i:].sum(), save_terminals[i])
        #reset
        step_num_vec_temp= np.append(step_num_vec_temp,epi_step)
        epi_step = 0
        nepisodes += 1
        episode_ended = True
        # reset the game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)
        # and reset the history
        state_with_history[:] = 0
        append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
        next_state_with_history = np.copy(state_with_history)
        # Start with returns of 0 and empty sets
        n_step_return = np.zeros(opt.early_stop)
        save_states = np.zeros((opt.early_stop, opt.hist_len*opt.state_siz))
        save_next_states = np.zeros((opt.early_stop, opt.hist_len*opt.state_siz))
        save_actions = np.zeros((opt.early_stop, opt.act_num))
        save_terminals = np.zeros_like(n_step_return)
        # check if image should be shown or not
        if nepisodes == show_image_count:
            show_image = not(show_image)
            if show_image:
                show_image_count += show_image_interval[1]
            else:
                show_image_count += show_image_interval[0]
        # check if it's time to update our step numbers
        if len(step_num_vec_temp) == numbers_games_in_bins:
            show_avg_steps = True
            stats["avg_episode_len"] = np.append(stats["avg_episode_len"],np.mean(step_num_vec_temp))
            stats["avg_episode_len_stamp"] = np.append(stats["avg_episode_len_stamp"],nepisodes)
            step_num_vec_temp = np.array([])
    
    # Print the loss
    if episode_ended:
        print("Episode {}. Loss in step {}: {:.2f}. Training time: {:.2f} min" .format(nepisodes,step,loss,(time.time()-start_time)/60.))
        stats["loss"] = np.append(stats["loss"],loss)
        stats["loss_stamp"] = np.append(stats["loss_stamp"],step)
        episode_ended = False
        
    if show_avg_steps:
        print('Average numbers of steps in last {} episodes: {:.2f}'.format(numbers_games_in_bins,stats["avg_episode_len"][-1]))
        show_avg_steps = False
    # Take best action
    best_action = np.argmax(agent.predict(sess, np.reshape(state_with_history, (1,opt.cub_siz*opt.pob_siz,opt.cub_siz*opt.pob_siz,opt.hist_len))))
    epsilon_action = randrange(opt.act_num)
    best_action = epsilon_action if np.random.rand() <= EPSILON*epsilon_decay**step else best_action
    action = best_action if not NOISY else best_action
        
    action_onehot = trans.one_hot_action(action)
    next_state = sim.step(action)
    # Update reward
    n_step_return[epi_step] = next_state.reward
    # append to history
    append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
    # Save transition
    save_states[epi_step] = state_with_history.reshape(-1)
    save_next_states[epi_step] = next_state_with_history.reshape(-1)
    save_actions[epi_step] = action_onehot
    save_terminals[epi_step] = next_state.terminal
    # mark next state as current state
    state_with_history = np.copy(next_state_with_history)
    state = next_state
    
    if trans.size >= opt.minibatch_size:
        #Training
        #Get batch
        batch, batch_weights = trans.sample_minibatch()
        states = np.zeros((len(batch), opt.cub_siz*opt.pob_siz, opt.cub_siz*opt.pob_siz, opt.hist_len))
        targets = np.zeros(len(batch))
        action_batch = np.zeros((len(batch), opt.act_num))

        for i in range(len(batch)):
            # Double Q-learning to eliminate maximation bias
            # Q(s, a) <- r + \gamma * Q_t(s', max(Q(s', a)))
            next_state = np.reshape(batch[i].next_state, (1, opt.cub_siz*opt.pob_siz, opt.cub_siz*opt.pob_siz, opt.hist_len))
            targets[i] = batch[i].reward + DISCOUNT_FACTOR * target.predict(sess, next_state)[0][0][np.argmax(agent.predict(sess, next_state))] * (1. - batch[i].terminal)

            # Shape & check if actions are one-hotted
            states[i] = np.reshape(batch[i].state, (1, opt.cub_siz*opt.pob_siz, opt.cub_siz*opt.pob_siz, opt.hist_len))
            if len(batch[i].action) == 1: #not np.array_equal([1, opt.act_num], batch[i].action.shape):
                action_batch[i] = trans.one_hot_action(batch[i].action)
            else:
                action_batch[i] = batch[i].action

        # Update Target Net and Agent
        target.update(sess)
        loss, tds = agent.update(sess, states.astype(np.float32), targets.astype(np.float32), action_batch.astype(np.float32), batch_weights.astype(np.float32), KEEP_PROB)
        trans.update_td(batch, tds)
        #stats.loss[step] = loss
        
    """
    if step % 250 == 0:
        test_state = test_sim.newGame(opt.tgt_y, opt.tgt_x)
        for i in range(100):
            # check if episode ended
            if state.terminal == True:
                stats.val_success[step // 250] = 1
                break
            else:
                #test run
                test_state_with_history = np.zeros((opt.hist_len, opt.state_siz))
                next_test_state = np.copy(test_state_with_history)
                append_to_hist(test_state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
                # Take action with highest Q-value
                action = np.argmax(agent.predict(sess, test_state_with_history.T[np.newaxis, ..., np.newaxis]))
                action_onehot = trans.one_hot_action(action)
                next_state = test_sim.step(action)
                # append to history
                append_to_hist(next_test_state, rgb2gray(next_state.pob).reshape(opt.state_siz))
                # add to the transition table
                trans.add(state_with_history.reshape(-1), action_onehot, next_test_state.reshape(-1), next_state.reward, next_state.terminal)
                # mark next state as current state
                test_state_with_history = np.copy(next_test_state)
                test_state = next_state 
    """

    if show_image and opt.disp_on:
        if win_all is None:
            plt.subplot(121)
            win_all = plt.imshow(state.screen)
            plt.subplot(122)
            win_pob = plt.imshow(state.pob)
        else:
            win_all.set_data(state.screen)
            win_pob.set_data(state.pob)
        plt.pause(opt.disp_interval)
        plt.draw()
        
    epi_step += 1

# 2. perform a final test of your model and save it
if not os.path.exists(PATH):
    os.mkdir(PATH)
saver = tf.train.Saver()
now = datetime.datetime.now()
save_path = saver.save(sess, os.path.join(PATH,"my-model_"+now.strftime("%y_%m_%d_%H_%M_%S")+".ckpt"))

# save losses and average episodal steps
np.savez(os.path.join(PATH,"my-data_"+now.strftime("%y_%m_%d_%H_%M_%S")+".npz"), loss=stats["loss"], loss_stamp =stats["loss_stamp"],\
         avg_episode_len=stats["avg_episode_len"],avg_episode_len_stamp=stats["avg_episode_len_stamp"] )

#plot_stats(stats)

"""
# Test on 500 steps in total
opt.disp_on = True

if opt.disp_on:
    win_all = None
    win_pob = None

epi_step = 0
nepisodes_test = 0
nepisodes_solved = 0
action = 0

# Restart game
state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)

for step in range(500):

    # Check if episode ended and if yes start new game
    if state.terminal or epi_step >= opt.early_stop:
        stats.tests[step] = epi_step
        epi_step = 0
        nepisodes += 1
        if state.terminal:
            nepisodes_solved += 1
        state = sim.newGame(opt.tgt_y, opt.tgt_x)
        state_with_history[:] = 0
        append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
        next_state_with_history = np.copy(state_with_history)
    else:
        action = np.argmax(agent.predict(sess, state_with_history.T[np.newaxis, ..., np.newaxis]))
        state = sim.step(action)
        epi_step += 1

    if opt.disp_on:
        if win_all is None:
            plt.subplot(121)
            win_all = plt.imshow(state.screen)
            plt.subplot(122)
            win_pob = plt.imshow(state.pob)
        else:
            win_all.set_data(state.screen)
            win_pob.set_data(state.pob)
        plt.pause(opt.disp_interval)
        plt.draw()

print("Solved {} of {} episodes" .format(nepisodes_solved, nepisodes))
print("Success rate of {}" .format(float(nepisodes_solved) / float(nepisodes)))
"""
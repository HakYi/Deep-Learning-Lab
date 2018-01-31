import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randrange
import tensorflow as tf
from collections import namedtuple
import argparse
import os

# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator
from transitionTable import TransitionTable

Statistics = namedtuple("Stats",["loss", "val_success", "tests"])

# Parse arguments from the command line
parser = argparse.ArgumentParser(description='Uses DQN to navigate a maze.')
parser.add_argument('-d' , help="set discount factor (default = 0.9)")
parser.add_argument('-eps' , help="set starting epsilon (default = 0.9)")
parser.add_argument('-noise' , help="use noisy nets (default True)")
parser.add_argument('-units' , help="set number of units per layer (default 128)")
parser.add_argument('-filters' , help="set number of filters (default 32)")
parser.add_argument('-filter_size' , help="set filter size (default 2)")
parser.add_argument('-pool_size' , help="set pool size (default 2)")
parser.add_argument('-episodes' , help="set number of episodes (default 3000)")
parser.add_argument('-lr' , help="set learning rate (default 0.0005)")
parser.add_argument('-gpu' , help="Use GPU (default False)")
parser.add_argument('-path' , help="output path for plots (default .)")
parser.add_argument('-plt_show' , help="show plots or not (default False)")
args = parser.parse_args()

# Set parameters from command line or default
# TODO: those are bad defaults
DISCOUNT_FACTOR = args.d if args.d != None else 0.9
EPSILON = args.eps if args.eps != None else 0.9
NOISY = args.noise if args.noise != None else True
UNITS = args.units if args.units != None else 128
FILTERS = args.filters if args.filters != None else 32
FILTER_SIZE = args.filter_size if args.filter_size != None else 2
POOL_SIZE = args.pool_size if args.pool_size != None else 2
EPISODES = args.episodes if args.episodes != None else 3 * 10 ** 3
LEARNING_RATE = args.lr if args.lr != None else 5e-4
GPU = args.gpu if args.gpu != None else False
PATH = args.path
PLT_SHOW = args.plt_show if args.plt_show != None else False

if GPU:
    device = "/gpu:0"
else:
    device = "/cpu:0"

def schedule(epsilon, step, max_steps):
    if step != 0:
        return epsilon * (1 - step/max_steps)
    else:
        return epsilon

class Agent:
    "Neural Network agent in tensorflow"
    def __init__(self, opt, scope="agent"):
        # Identify the Network
        self.scope = scope
        with tf.variable_scope(scope):
            self.model = self.build_model(opt)

    "Build the model"
    def build_model(self, opt):
        with tf.device(device):
            self.x = tf.placeholder(tf.float32, shape=[None, opt.state_siz, opt.hist_len, 1])

            # Placeholders for loss
            self.targets = tf.placeholder(tf.float32, shape = [None])
            self.action_onehot = tf.placeholder(tf.float32, shape=[None, opt.act_num])
            self.normalized_is_weights = tf.placeholder(tf.float32, shape=[None])

            # L2 Regularizer
            self.l2 = tf.contrib.layers.l2_regularizer(scale=0.1)

            # Convolutional layers (padded to size)
            self.conv1 = tf.contrib.layers.conv2d(self.x, FILTERS, FILTER_SIZE)
            self.conv2 = tf.contrib.layers.conv2d(self.conv1, FILTERS, FILTER_SIZE)
            self.pool1 = tf.contrib.layers.max_pool2d(self.conv2, POOL_SIZE)

            # Flatten before hidden layers
            self.flatten = tf.contrib.layers.flatten(self.pool1)

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
                self.dropout1 = tf.contrib.layers.dropout(self.hidden1)
                self.hidden2 = tf.contrib.layers.fully_connected(self.dropout1, UNITS, weights_regularizer=self.l2, activation_fn=tf.nn.relu)
                self.dropout2 = tf.contrib.layers.dropout(self.hidden2)

            # Linear output
            self.out = tf.contrib.layers.fully_connected(self.dropout2, opt.act_num, activation_fn=None)

            selected_q = tf.reduce_sum(self.action_onehot * self.out, 1)

            # We need to factor in the importance sampling weights here to avoid a sampling bias
            # The weight for a transition i is w_i = 1/(N*P(i))^\beta with P(i) being the priority of i
            # Source: Schaul, Quan, Antonoglou, Silver - Prioritized Experience Replay (2016)
            self.tds = self.targets - selected_q
            self.loss = tf.reduce_mean(np.square(np.multiply(self.normalized_is_weights, self.tds)))

            #Adam optimizer
            self.train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

            pass

    def predict(self, sess, states):
        """
        Returns agent prediction
        """
        return sess.run([self.out], feed_dict = {self.x: states})

    def update(self, sess, state_batch, targets, action_batch, is_weights):
        """
        Trains the network
        """
        self.train_step.run(feed_dict={self.x: state_batch, self.targets: targets, self.action_onehot: action_batch, self.normalized_is_weights: is_weights}, session = sess)
        return sess.run([self.loss, self.tds], feed_dict={self.x: state_batch, self.targets: targets, self.action_onehot: action_batch, self.normalized_is_weights: is_weights})

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
    plt.plot(stats.loss)
    plt.xlabel("Timestep")
    plt.ylabel("Loss")
    plt.title("Loss per step")

    # Plot successes in in-between tests
    fig2 = plt.figure(figsize=(10,10))
    plt.plot(stats.val_success)
    plt.xlabel("Validation Intervall")
    plt.ylabel("Success (yes/no)")
    plt.title("Successes in intermediate testing")

    # Plot test steps
    fig3 = plt.figure(figsize=(10,10))
    plt.plot(stats.tests)
    plt.xlabel("Test Number")
    plt.ylabel("Steps used")
    plt.title("Steps per test")

    if PATH != None:
        if not os.path.exists(PATH):
            os.makedirs(PATH)

        fig1.savefig(PATH + "/loss.png")
        fig2.savefig(PATH + "/inter_tests.png")
        fig3.savefig(PATH + "/test_steps.png")
    else:
        fig1.savefig('loss.png')
        fig1.savefig('inter_tests.png')
        fig1.savefig('test_steps.png')

    if PLT_SHOW:
        plt.show(fig1)
        plt.show(fig2)
        plt.show(fig3)

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
test_sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
# setup a large transitiontable that is filled during training
maxlen = 100000
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                        opt.minibatch_size, maxlen)
agent = Agent(opt)
target = TargetNetwork(opt)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

opt.disp_on = False

if opt.disp_on:
    win_all = None
    win_pob = None

# lets assume we will train for a total of 1 million steps
# this is just an example and you might want to change it
steps = int(EPISODES)
epi_step = 0
nepisodes = 0
stats = Statistics(loss = np.zeros(steps), val_success = np.zeros(steps //250), tests = np.ones(500)*100)
epsilon = EPSILON

state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)
for step in range(steps):
    if state.terminal or epi_step >= opt.early_stop:
        epi_step = 0
        nepisodes += 1
        # reset the game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)
        # and reset the history
        state_with_history[:] = 0
        append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
        next_state_with_history = np.copy(state_with_history)

    # Take action epslion-greedily
    epsilon = schedule(epsilon, step, steps)
    best_action = np.random.choice([0, 1], p=[epsilon, 1-epsilon])
    if best_action == True:
        action = np.argmax(agent.predict(sess, np.reshape(state_with_history, (900, 4))[np.newaxis, ..., np.newaxis]))
    else:
        action = np.random.choice(opt.act_num)

    action_onehot = trans.one_hot_action(action)
    next_state = sim.step(action)
    # append to history
    append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
    # add to the transition table
    trans.add(state_with_history.reshape(-1), action_onehot, next_state_with_history.reshape(-1), next_state.reward, next_state.terminal)
    # mark next state as current state
    state_with_history = np.copy(next_state_with_history)
    state = next_state

    #Training
    #Get batch
    batch, batch_weights = trans.sample_minibatch()
    states = np.zeros((len(batch), 900, 4, 1))
    targets = np.zeros(len(batch))
    action_batch = np.zeros((len(batch), opt.act_num))

    for i in range(len(batch)):
        # Double Q-learning to eliminate maximation bias
        # Q(s, a) <- r + \gamma * Q_t(s', max(Q(s', a)))
        next_state = np.reshape(batch[i].next_state, (900, 4))[np.newaxis, ..., np.newaxis]
        targets[i] = batch[i].reward + DISCOUNT_FACTOR * target.predict(sess, next_state)[0][0][np.argmax(agent.predict(sess, next_state))] * (1. - batch[i].terminal)

        # Shape & check it actions are one-hotted
        states[i] = np.reshape(batch[i].state, (900, 4, 1))
        if not np.array_equal([1, opt.act_num], batch[i].action.shape):
            action_batch[i] = trans.one_hot_action(batch[i].action)
        else:
            action_batch[i] = batch[i].action

    # Update Target Net and Agent
    target.update(sess)
    loss, tds = agent.update(sess, states, targets, action_batch, batch_weights)
    trans.update_td(batch, tds)
    stats.loss[step] = loss

    # Print the loss
    print("Loss in step {}: {}" .format(step, loss))

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


# 2. perform a final test of your model and save it
saver = tf.train.Saver()
save_path = saver.save(sess, "./model.ckpt")
plot_stats(stats)

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

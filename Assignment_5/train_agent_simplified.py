import numpy as np
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randrange
import argparse
import os
import time
import datetime

# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator
from transitionTable_simplified import TransitionTable

# Parse arguments from the command line
parser = argparse.ArgumentParser(description='Uses DQN to navigate a maze.')
parser.add_argument('-gpu' , help="use GPU? (default 1)", type=int,                            default=1)
parser.add_argument('-d' , help="set discount factor (default = 0.99)", type=float,                         default=0.99)
parser.add_argument('-peps' , help="part of steps after which epsilon decreases to 0.01 (default 0.025)", type=float,       default=0.025)
parser.add_argument('-psteps' , help="part of steps before training (default 0.08)", type=float,                           default=0.08)
parser.add_argument('-alpha' , help="parameter for replay prioritization (default = 0.7)",type=float,       default=0.7)
parser.add_argument('-noise' , help="use noisy nets (default 0)", type=int,                                 default=0)
parser.add_argument('-units' , help="set number of units per layer (default 512)", type=int,                default=512)
parser.add_argument('-filt1' , help="set number of filters of convlayer1 (default 32)", type=int,           default=32)
parser.add_argument('-fs1' , help="set filter size of convlayer1 (default 8)", type=int,                    default=8)
parser.add_argument('-filt2' , help="set number of filters of convlayer2 (default 64)",type=int,            default=64)
parser.add_argument('-fs2' , help="set filter size of convlayer2 (default 4)",type=int,                     default=4)
parser.add_argument('-filt3' , help="set number of filters of convlayer2 (default 64)",type=int,            default=64)
parser.add_argument('-fs3' , help="set filter size of convlayer3 (default 3)",type=int,                     default=3)
parser.add_argument('-ps' , help="set pool size (default 1)", type=int,                                     default=1)
parser.add_argument('-steps' , help="set number of training steps (default 1e6)",type=float,                default=1e6)
parser.add_argument('-ti' , help="training interval (default 1)",type=int,                                  default=1)
parser.add_argument('-lr' , help="set learning rate (default 0.0000625)",type=float,                        default=6.25e-5)
parser.add_argument('-path' , help="output path for data, ckpts and plots (default 'workingdir/data')",     default=os.path.join(os.getcwd(),"data"))
parser.add_argument('-fn', help="folder number for default workingdir",type=int,                            default=0)
parser.add_argument('-ckpt_i' , help="checkpoints interval (default 1e3)", type=float,                      default=1e3)
parser.add_argument('-plt_show' , help="show plots or not (default 0)", type=int,                           default=0)
args = parser.parse_args()

""" Variables """
GPU = args.gpu
DISCOUNT_FACTOR = args.d
ALPHA = args.alpha
NOISY = args.noise
UNITS = args.units
FILTERS1 = args.filt1
FILTER_SIZE1 = args.fs1
FILTERS2 = args.filt2
FILTER_SIZE2 = args.fs2
FILTERS3 = args.filt3
FILTER_SIZE3 = args.fs3
POOL_SIZE = args.ps
STEPS = int(args.steps)
STEPS_BEFORE_TRAIN = int(args.peps*STEPS)
STEPS_TO_NULL_EPSILON = int(args.peps*STEPS)
TRAINING_INTERVAL = args.ti
LEARNING_RATE = args.lr
FOLDER_NUM = args.fn
PATH = args.path+str(FOLDER_NUM)
CKPT_INTERVAL = args.ckpt_i
PLT_SHOW = args.plt_show
EPSILON = 0.9 if not NOISY else 0.

""" Initializations """
tf.reset_default_graph() 

""" Agent class (implements DQN with choosable extensions) """
class Agent:
    "Neural Network agent in tensorflow"
    def __init__(self, opt, scope="agent"):
        # Identify the Network
        self.scope = scope
        with tf.variable_scope(scope):
            self.model = self.build_model(opt)

    "Build the model"
    def build_model(self, opt):
        with tf.device("/gpu:0" if GPU else "/cpu:0"):
            self.x = tf.placeholder(tf.float32, shape=[None, opt.cub_siz*opt.pob_siz, opt.cub_siz*opt.pob_siz, opt.hist_len])
            
            # Placeholder for keep probability (= 1- droput rate)
            self.is_training = tf.placeholder(tf.bool)
    
            # Placeholders for loss
            self.targets = tf.placeholder(tf.float32, shape = [None])
            self.action_onehot = tf.placeholder(tf.float32, shape=[None, opt.act_num])
    
            # L2 Regularizer
            #self.l2 = tf.contrib.layers.l2_regularizer(scale=0.1)
    
            # Convolutional layers (padded to size)
            self.conv1 = tf.contrib.layers.conv2d(self.x, FILTERS1, FILTER_SIZE1,stride=4)
            self.pool1 = tf.contrib.layers.max_pool2d(self.conv1, POOL_SIZE)
            self.conv2 = tf.contrib.layers.conv2d(self.pool1, FILTERS2, FILTER_SIZE2,stride=2)
            self.pool2 = tf.contrib.layers.max_pool2d(self.conv2, POOL_SIZE)
            self.conv3 = tf.contrib.layers.conv2d(self.pool2, FILTERS3, FILTER_SIZE3,stride=1)
            self.pool3 = tf.contrib.layers.max_pool2d(self.conv3, POOL_SIZE)
    
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
    
                self.dropout1 = tf.contrib.layers.dropout(self.hidden1,self.is_training)
    
                self.w2 = tf.get_variable('weights_hidden2', shape=[self.dropout1.shape[1], UNITS])
                self.b2 = tf.get_variable('bias_hidden2', shape=[UNITS])
    
                self.noise_w2 = tf.get_variable('noise_weights_hidden2', shape=[self.dropout1.shape[1], UNITS])
                self.noise_b2 = tf.get_variable('noise_bias_hidden2', shape=[UNITS])
    
                # Independent Gaussian noise
                self.eps_w2 = tf.random_normal(self.w2.shape, 0., 0.5)
                self.eps_b2 = tf.random_normal([UNITS], 0, 0.5)
    
                self.hidden2 = tf.nn.relu(tf.matmul(self.dropout1, self.w2) + self.b2 + tf.matmul(self.dropout1, np.multiply(self.noise_w2, self.eps_w2)) + np.multiply(self.noise_b2, self.eps_b2))
    
                self.dropout2 = tf.contrib.layers.dropout(self.hidden2,self.is_training)
    
            else:
                #self.hidden1 = tf.contrib.layers.fully_connected(self.dropout1, UNITS, weights_regularizer=self.l2, activation_fn=tf.nn.relu)
                self.hidden1 = tf.contrib.layers.fully_connected(self.flatten, UNITS, activation_fn=tf.nn.relu)
                self.dropout2 = tf.contrib.layers.dropout(self.hidden1,is_training=self.is_training)
        
            # Linear output
            self.out = tf.contrib.layers.fully_connected(self.dropout2, opt.act_num, activation_fn=None)
    
            self.selected_q = tf.reduce_sum(self.action_onehot * self.out, 1)
    
            # We need to factor in the importance sampling weights here to avoid a sampling bias
            # The weight for a transition i is w_i = 1/(N*P(i))^\beta with P(i) being the priority of i
            # Source: Schaul, Quan, Antonoglou, Silver - Prioritized Experience Replay (2016)
            self.tds = self.targets - self.selected_q
            self.loss = tf.reduce_mean(np.square(self.tds))
        
            #Adam optimizer
            self.opt = tf.train.AdamOptimizer(LEARNING_RATE,epsilon=1.5e-4)
            #tf.get_variable_scope().reuse_variables()
            self.train_step = self.opt.minimize(self.loss,colocate_gradients_with_ops=True)
    
        pass

    def predict(self, sess, states):
        """
        Returns agent prediction
        """
        return sess.run([self.out], feed_dict = {self.x: states, self.is_training: False})

    def update(self, sess, state_batch, targets, action_batch):
        """
        Trains the network
        """
        sess.run(self.train_step,feed_dict={self.x: state_batch, self.targets: targets, self.action_onehot: action_batch, self.is_training: True})
        return sess.run([self.loss, self.tds], feed_dict={self.x: state_batch, self.targets: targets, self.action_onehot: action_batch, self.is_training: False})
""" End of class definition """

class TargetNetwork(Agent):
    """
    Slowly updated target network. Tau indicates the speed of adjustment. If 1,
    it is always set to the values of its associate.
    """
    def __init__(self, opt):
        self.scope = "target"
        Agent.__init__(self, opt, self.scope)
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
          op_holder.append(target_var.assign(agent_var.value()))
      return op_holder

    def update(self, sess):
      for op in self._associate:
        sess.run(op)
""" End of class definition """

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

    fig1.savefig(os.path.join(PATH,"loss.png"))
    fig2.savefig(os.path.join(PATH,"avg_episode_len.png"))

    if PLT_SHOW:
        plt.show(fig1)
        plt.show(fig2)


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
                        opt.minibatch_size, maxlen, ALPHA)
agent = Agent(opt)
target = TargetNetwork(opt)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

if opt.disp_on:
    win_all = None
    win_pob = None

# Parameter for epsilon schedule (greedy choice of action)
epsilon_decay = (0.01/EPSILON)**(1./STEPS_TO_NULL_EPSILON)

show_image = False
show_image_interval = [20,5] #every 20th episode, show 5 episodes
show_image_count = show_image_interval[0]
numbers_games_in_bins = 10
step_num_vec_temp = np.array([])

epi_step = 0
nepisodes = 0
episode_ended = False
show_avg_steps = False

input_shape = (-1,opt.cub_siz*opt.pob_siz,opt.cub_siz*opt.pob_siz,opt.hist_len)
stats = {"loss": np.array([]), "loss_stamp": np.array([]), "avg_episode_len": np.array([]), "avg_episode_len_stamp": np.array([])}

# Prepare saver and directory
if not os.path.exists(PATH):
    os.mkdir(PATH)
saver = tf.train.Saver()

start_time = time.time()
loss = 0

"""Collecting samples"""
print("First, collect some data in {} steps ...".format(STEPS_BEFORE_TRAIN))
state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)

for step in range(STEPS_BEFORE_TRAIN):
    if state.terminal or epi_step >= opt.early_stop:
        # reset the game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)
        # and reset the history
        state_with_history[:] = 0
        append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
        next_state_with_history = np.copy(state_with_history)

    # Take random action
    action = randrange(opt.act_num)
        
    action_onehot = trans.one_hot_action(action)
    next_state = sim.step(action)
    # append to history
    append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
    # Save transition
    trans.add(state_with_history.reshape(-1), action_onehot, next_state_with_history.reshape(-1), next_state.reward, next_state.terminal, np.nan)
    # mark next state as current state
    state_with_history = np.copy(next_state_with_history)
    state = next_state
""" Collecting samples ended """

print("Collecting data was successful. Now start training with {} steps ...".format(STEPS))

""" TRAINING """
for step in range(STEPS):
    if state.terminal or epi_step >= opt.early_stop:
        #reset
        step_num_vec_temp = np.append(step_num_vec_temp,epi_step)
        stats["avg_episode_len"] = np.append(stats["avg_episode_len"],epi_step)
        stats["avg_episode_len_stamp"] = np.append(stats["avg_episode_len_stamp"],nepisodes)
        epi_step = 0
        nepisodes += 1
        episode_ended = True
        # reset the game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)
        # and reset the history
        state_with_history[:] = 0
        append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
        next_state_with_history = np.copy(state_with_history)
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
            avg_step_num = np.mean(step_num_vec_temp)
            step_num_vec_temp = np.array([])
    
    # Print the loss
    if episode_ended:
        print("Episode {}. Loss in step {}: {:.2f}. Training time: {:.2f} min" .format(nepisodes,step,loss,(time.time()-start_time)/60.))
        stats["loss"] = np.append(stats["loss"],loss)
        stats["loss_stamp"] = np.append(stats["loss_stamp"],step)
        episode_ended = False
        
    if show_avg_steps:
        print('Average numbers of steps in last {} episodes: {:.2f}'.format(numbers_games_in_bins,avg_step_num))
        show_avg_steps = False
    # Take best action
    best_action = np.argmax(agent.predict(sess, np.reshape(state_with_history, input_shape)))
    epsilon_action = randrange(opt.act_num)
    action = epsilon_action if np.random.rand() <= EPSILON*epsilon_decay**step else best_action
    action = action if not NOISY else best_action
        
    action_onehot = trans.one_hot_action(action)
    next_state = sim.step(action)
    # append to history
    append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
    # Save transition
    curr_state = np.reshape(state_with_history,input_shape)
    next_state_shaped = np.reshape(next_state_with_history,input_shape)
    td_err = next_state.reward + DISCOUNT_FACTOR * target.predict(sess,next_state_shaped)[0][0][np.argmax(agent.predict(sess,next_state_shaped))] - agent.predict(sess,curr_state)[0][0][action]
    trans.add(state_with_history.reshape(-1), action_onehot, next_state_with_history.reshape(-1), next_state.reward, next_state.terminal, td_err)
    # mark next state as current state
    state_with_history = np.copy(next_state_with_history)
    state = next_state
    epi_step += 1
    
    #Training
    #Get batch
    if step % TRAINING_INTERVAL == 0:
        state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = trans.sample_minibatch()
        states = np.zeros((len(terminal_batch), opt.cub_siz*opt.pob_siz, opt.cub_siz*opt.pob_siz, opt.hist_len))
        targets = np.zeros(len(terminal_batch))
        action_batch = np.zeros((len(terminal_batch), opt.act_num))

        for i in range(len(terminal_batch)):
            # Double Q-learning to eliminate maximation bias
            # Q(s, a) <- r + \gamma * Q_t(s', max(Q(s', a)))
            next_state = np.reshape(next_state_batch[i], input_shape)
            targets[i] = reward_batch[i] + DISCOUNT_FACTOR * target.predict(sess, next_state)[0][0][np.argmax(agent.predict(sess, next_state))] * (1. - terminal_batch[i])
            states[i] = np.reshape(state_batch[i], input_shape)

        # Update Target Net and Agent
        target.update(sess)
        loss, tds = agent.update(sess, states.astype(np.float32), targets.astype(np.float32), action_batch.astype(np.float32))

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
    
    if step != 0 and step % CKPT_INTERVAL == 0:
        now = datetime.datetime.now()
        save_path = saver.save(sess, os.path.join(PATH,"my-model_"+now.strftime("%y_%m_%d_%H_%M_%S")+".ckpt"))
        np.savez(os.path.join(PATH,"my-data_"+now.strftime("%y_%m_%d_%H_%M_%S")+".npz"), loss=stats["loss"], loss_stamp =stats["loss_stamp"],\
                 avg_episode_len=stats["avg_episode_len"],avg_episode_len_stamp=stats["avg_episode_len_stamp"] )
    elif step == 0:
        now = datetime.datetime.now()
        save_path = saver.save(sess, os.path.join(PATH,"my-model_"+now.strftime("%y_%m_%d_%H_%M_%S")+"_start.ckpt"))
        np.savez(os.path.join(PATH,"my-data_"+now.strftime("%y_%m_%d_%H_%M_%S")+"_start.npz"), loss=stats["loss"], loss_stamp =stats["loss_stamp"],\
                 avg_episode_len=stats["avg_episode_len"],avg_episode_len_stamp=stats["avg_episode_len_stamp"] )    
""" End of training """
print("Training was successful. Now save all important data and variables...")
    
# 2. Save the model one last time
now = datetime.datetime.now()
save_path = saver.save(sess, os.path.join(PATH,"my-model_"+now.strftime("%y_%m_%d_%H_%M_%S")+"_final.ckpt"))
np.savez(os.path.join(PATH,"my-data_"+now.strftime("%y_%m_%d_%H_%M_%S")+"_final.npz"), loss=stats["loss"], loss_stamp =stats["loss_stamp"],\
         avg_episode_len=stats["avg_episode_len"],avg_episode_len_stamp=stats["avg_episode_len_stamp"] )
print("Data and variables were stored successfully.")
plot_stats(stats)
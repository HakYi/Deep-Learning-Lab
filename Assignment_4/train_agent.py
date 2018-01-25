import numpy as np
#import matplotlib
#matplotlib.use('TkAgg')
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randrange
import time
import os
plt.close('all')

# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator
from transitionTable import TransitionTable
from network import ConvNet

def append_to_hist(state, obs):
    """
    Add observation to the state.
    """
    for i in range(state.shape[0]-1):
        state[i, :] = state[i+1, :]
    state[-1, :] = obs

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
Q_Network = ConvNet(opt.cub_siz,opt.pob_siz,opt.hist_len,opt.act_num,opt.num_filt1,opt.kernel_size1,opt.num_filt2,opt.kernel_size2,
                 opt.pool_size,opt.dense_units,opt.dropout_rate,opt.learning_rate)
sess = tf.Session()
# Run the initializer
sess.run(Q_Network.init)
# setup a large transitiontable that is filled during training
maxlen = 100000
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                        opt.minibatch_size, maxlen)

if opt.disp_on:
    win_all = None
    win_pob = None
    
# lets assume we will train for a total of 1 million steps
# this is just an example and you might want to change it
steps = int(1e5)
epi_step = 0
nepisodes = 0

epsilon = 0.2

show_image = False
show_image_interval = [20,5] #every 20th episode, show 5 episodes
show_image_count = show_image_interval[0]
numbers_games_in_bins = 10
step_num_vec_temp = np.array([])
step_num_vec = []
time_stamp_vec = []
loss_vec = []
loss_stamp_vec = []

step_num_count = 0
start_time = time.time()

state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)
for step in range(steps):
    if state.terminal or epi_step >= opt.early_stop:
        step_num_vec_temp= np.append(step_num_vec_temp,epi_step)
        epi_step = 0
        nepisodes += 1
        print('Episode:',nepisodes)
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
            average_steps = np.mean(step_num_vec_temp)
            time_stamp = (time.time()-start_time)/60.
            step_num_vec.append(average_steps)
            time_stamp_vec.append(time_stamp)
            step_num_vec_temp = np.array([])
            print('Numbers of agent steps:',step_num_vec)
    epi_step += 1
    #epsilon = epsilon*0.9999    
    # agent takes its action
    actions = sess.run(Q_Network.Q, feed_dict={Q_Network.x:state_with_history.reshape((1,opt.hist_len*opt.state_siz)),Q_Network.keep_prob:1.})
    action = np.argmax(actions)
    #if action == 0:
     #   action = np.argmax(actions[:,1:])+1
    if np.random.rand() <= epsilon:
        action = randrange(opt.act_num) # this just gets a random action
    action_onehot = trans.one_hot_action(action)
    next_state = sim.step(action)
    # append to history
    append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
    # add to the transition table
    trans.add(state_with_history.reshape(-1), action_onehot, next_state_with_history.reshape(-1), next_state.reward, next_state.terminal)
    # mark next state as current state
    state_with_history = np.copy(next_state_with_history)
    state = next_state

    # train agent
    state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = trans.sample_minibatch()
    # 1) pre-define variables and networks as outlined above
    # 1) here: calculate best action for next_state_batch
    # 2) with that action make an update to the q values
    #    as an example this is how you could print the loss 
    #print(sess.run(loss, feed_dict = {x : state_batch, u : action_batch, ustar : action_batch_next, xn : next_state_batch, r : reward_batch, term : terminal_batch}))
    action_batch_next = sess.run(Q_Network.Qn, feed_dict={Q_Network.xn:next_state_batch,Q_Network.keep_prob:1.})
    action_batch_next = trans.one_hot_action(np.argmax(action_batch_next,1))
    sess.run(Q_Network.train_op, feed_dict={Q_Network.x:state_batch,Q_Network.u:action_batch,\
                                            Q_Network.xn:next_state_batch,Q_Network.ustar:action_batch_next,\
                                            Q_Network.r:reward_batch,Q_Network.term:terminal_batch,\
                                            Q_Network.keep_prob:Q_Network.dropout_rate})
    if step % 50 == 0:
        loss = sess.run(Q_Network.loss_op, feed_dict={Q_Network.x:state_batch,Q_Network.u:action_batch,\
                                            Q_Network.xn:next_state_batch,Q_Network.ustar:action_batch_next,\
                                            Q_Network.r:reward_batch,Q_Network.term:terminal_batch,\
                                            Q_Network.keep_prob:1.})
        print('Loss at step '+str(step)+':', loss, show_image)
        loss_vec.append(loss)
        loss_stamp_vec.append(step)
    
    # every once in a while test agent here so that one can track its performance
    if show_image:
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
save_path = 'checkpoints/'
saver = tf.train.Saver()
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_path = saver.save(sess, os.path.join(save_path,'DQN_Network'))

# Loss function
plt.figure()
plt.plot(loss_stamp_vec,loss_vec)
plt.xlabel('Simulation step')
plt.ylabel('Loss')

plt.figure()
plt.plot(time_stamp_vec,step_num_vec)
plt.xlabel('Time (min)')
plt.ylabel('Average number of agent steps for one episode')

import numpy as np
import os

class Options:
    #
    disp_on = True # you might want to set it to False for speed
    map_ind = 0
    change_tgt = False
    states_fil = "states.csv"
    labels_fil = "labels.csv"
    network_fil = "network.json"
    weights_fil = "network.h5"
    # simulator config
    disp_interval = .005
    if map_ind == 0:
        cub_siz = 5
        pob_siz = 5 # for partial observation
        # this defines the goal position
        tgt_y = 12
        tgt_x = 11
        early_stop = 50
    elif map_ind == 1:
        cub_siz = 10
        pob_siz = 3 # for partial observation
        # this defines the goal position
        tgt_y = 5
        tgt_x = 5
        early_stop = 75
    state_siz = (pob_siz * cub_siz) ** 2 # when use pob as input
    if change_tgt:
        tgt_y = 16
        tgt_x = 25
    act_num = 5

    # training hyper params
    hist_len = 4
    minibatch_size  = 32
    n_minibatches   = 500
    valid_size      = 500
    eval_nepisodes  = 50
    checkpoint_dir = '\\tmp\\tensorflow\\NeuralPlanner\\checkpoints'
    checkpoint_dir_save = '\\tmp\\tensorflow\\NeuralPlanner\\checkpoints\\save'
    num_epochs = 10
    # network params
    num_filt1=32
    kernel_size1=5
    num_filt2=64
    kernel_size2=5
    pool_size=2
    dense_units=1024
    dropout_rate=0.4

    data_steps  = n_minibatches * minibatch_size + valid_size
    eval_steps  = early_stop * eval_nepisodes
    eval_freq   = n_minibatches # evaluate after each epoch
    prog_freq   = 500

class State: # return tuples made easy
    def __init__(self, action, reward, screen, terminal, pob):
        self.action   = action
        self.reward   = reward
        self.screen   = screen
        self.terminal = terminal
        self.pob      = pob


# The following functions were taken from scikit-image
# https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py
        
def rgb2gray(rgb):
    if rgb.ndim == 2:
        return np.ascontiguousarray(rgb)

    gray = 0.2125 * rgb[..., 0]
    gray[:] += 0.7154 * rgb[..., 1]
    gray[:] += 0.0721 * rgb[..., 2]

    return gray

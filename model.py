
# import libs for numerical ops
import numpy as np
import random
import glob
import sys
import pdb

# neural net libs
from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, MaxPooling2D, GaussianDropout

# other libraries
import load

sys.setrecursionlimit(10000)
EPS = np.finfo(np.double).tiny
NUCLEOTIDES = ['A','C','G','T']

def get_pwms(path_to_pwms):

    pwm_files = glob.glob(path_to_pwms+'/*')
    pwm_arrs = []
    pwm_names = []
    max_pwm_len = 0
    for pwm_file in pwm_files:

        # load pwm as a dictionary
        pwm_dict = dict([(nuc,[]) for nuc in NUCLEOTIDES])
        handle = open(pwm_file, 'r')
        pwm_names.append(handle.next().strip())
        nucs = handle.next().strip().split()
        for line in handle:
            values = line.strip().split()
            ig = [pwm_dict[n].append(v) for n,v in zip(nucs,values)]
        handle.close()
        
        # process pwm into a normalized matrix
        pwm_len = len(pwm_dict['A'])
        pwm_arr = np.zeros((4,pwm_len), dtype='float32')
        for i,n in enumerate(NUCLEOTIDES):
            pwm_arr[i] = pwm_dict[n]

        # normalize so each position sums to 1
        pwm_arr = pwm_arr/np.sum(pwm_arr,0)

        pwm_arrs.append(pwm_arr)
        if pwm_len>max_pwm_len:
            max_pwm_len = pwm_len

    # collate all PWMs into a giant tensor
    pwms = []
    tfs = []
    for pwm_arr,pwm_name in zip(pwm_arrs,pwm_names):

        pwm_len = pwm_arr.shape[1]

        # positive strand PWM
        pwm = np.hstack((pwm_arr, 0.25*np.ones((4,max_pwm_len-pwm_len))))
        pwm[pwm==0] = EPS
        # non-informative positions have zeros for all nucleotides
        pwm = np.log2(pwm)+2
        pwms.append(pwm)
        tfs.append((pwm_name+':+',pwm_len))

        # negative strand PWM
        pwm = np.hstack((pwm_arr[:,::-1][::-1,:], 0.25*np.ones((4,max_pwm_len-pwm_len))))
        pwm[pwm==0] = EPS
        # non-informative positions have zeros for all nucleotides
        pwm = np.log2(pwm)+2
        pwms.append(pwm)
        tfs.append((pwm_name+':-',pwm_len))

    pwms = np.array(pwms)

    return pwms, tfs

def build_neural_network(num_outputs, output_activation, path_to_pwms, window_size):

    # get PWMs
    pwms, tfs = get_pwms(path_to_pwms)
    P, ig, L = pwms.shape

    # construct the first layer, with a convolution filter for each PWM
    network = Sequential([Convolution2D(P, 4, L, \
                          input_shape=(1, 4, window_size), \
                          weights=[pwms.reshape(P, 1, 4, L), np.zeros((P, ))], \
                          trainable=False, activation='relu')])

    # maxpool layer
    network.add(MaxPooling2D(pool_size=(1,4)))

    # 200 convolutional filters in second layer
    network.add(Convolution2D(200, 1, 6, \
                          activation='relu'))

    # maxpool layer
    network.add(MaxPooling2D(pool_size=(1,3)))

    # 500 neuron dense layer
    network.add(Flatten())
    network.add(Dense(500, activation='relu'))

    # 20% dropout
    network.add(GaussianDropout(0.2))

    # output layer
    network.add(Dense(num_outputs, activation=output_activation))

    return network, tfs

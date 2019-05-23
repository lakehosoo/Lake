'''
Created on Fri Mar  1 20:59:02 2019

@author: Rozenk
'''

"""
Implementation of DQN
Algorithm and hyperparameter details can be found here:
    http://arxiv.org/pdf/1509.02971v2.pdf
The algorithm is tested on the Pendulum-v0 OpenAI gym task
and developed with tflearn + Tensorflow
Author: Patrick Emami
"""

import time
start_time = time.time()

import tensorflow as tf
import numpy as np
import pickle
import random
#import matplotlib.pyplot as plt

from ou_noise import OUNoise
from Buffer import Buffer
from SMB_Env import SMB_Env
from SMB_Agent import agent
import os

# ==========================
#   Training Parameters
# ==========================
RANDOM_SEED = random.randint(0,100000)

# Pre training steps
MAX_TRA = 0
sample_num = 64

# Max training steps
INITIAL_POLICY_INDEX = 50
AC_PE_TRAINING_INDEX = 0
MAX_EPISODES = 50

gamma = 0.9

# Hyperparameter for the Critic Network
LEARNING_RATE = 0.0001
EPOCHS = 30
BATCH_NUM = 64
TAU = 0.03
validation_ratio = 0.2

# Size of replay buffer
BUFFER_SIZE = 8000
BUFFER_SIZE_TERMINAL = 12
AC_MINIBATCH_SIZE = 512
C_MINIBATCH_SIZE_TERMINAL = 4

noise_ini = 0 #1000 / 100000
noise_tra = 0 #1000 / 100000

scale2 = 4000


# ===========================
#   Agent Training
# ===========================

def train(sess, env, SMB_Agent):

    # Initialize target network weights
    # critic.update_target_network()

    state_dim = env.s_dim
    action_dim = env.a_dim
    reward_dim = 1
    N = env.N

    # Buffer + Data

    # Initialize exploration noise
    MAX_EP_STEPS = int(env.Total_simulation_time /env.st) # 1*8

    decay = 1

    # Training with simulation
    for i in range(MAX_EPISODES):
        print(i)
        s2 = time.time()
        product_ep = np.array([0.])
        reward_ep = np.array([0.])

        # Position value
        s, action_predict = env.reset()

        decay = decay - 1/(1.5*MAX_EPISODES)
        action_prev = np.array([[0.0191, 0.0191, 0.0135, 0.0135, 0.0165, 0.0165, 0.0105, 0.0105]])
        data = np.array(([np.zeros(2*env.s_dim + env.a_dim + 1)])) # dummy data need....

        for t in range(MAX_EP_STEPS):

            action_predict = SMB_Agent.choose_rand(s)
            action_predict = 1/scale2*np.reshape(action_predict, [1,-1])
            action = action_prev + action_predict #noise_ini * action_noise

            delaction = action - action_prev
            action_ub = np.array(([[0.0201, 0.0201, 0.0150, 0.0150, 0.0175, 0.0175, 0.0115, 0.0115]]))
            action_lb = np.array(([[0.0181, 0.0181, 0.0125, 0.0125, 0.0150, 0.0150, 0.0095, 0.0095]]))
            # action_lb2 = [[0.0181*50, action[0][0], 0.0120*50, action[0][2], action[0][3], action[0][4], 0.0095*50, action[0][6]]]
            action = np.clip(action, action_lb, action_ub)
            #action = np.clip(action, action_lb2, action_ub)

            for tt in range(env.Nt):
                ss, r = env.step(s[0], action[0])
                ss = np.array([ss])

                data_set = np.concatenate((s[0], scale2*delaction[0], np.array([r]), ss[0]))
                data = np.concatenate((data, np.array([data_set])))

                action_prev = action

                s = ss
                reward_ep = gamma*reward_ep + r

    with open('data1', 'wb') as data_file:
        pickle.dump(data, data_file)

def main():
    with tf.Session() as sess:
        env = SMB_Env()

        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        env.seed(RANDOM_SEED)

        state_dim = env.s_dim
        action_dim = env.a_dim

        SMB_Agent = agent(sess, state_dim + action_dim, LEARNING_RATE, EPOCHS, BATCH_NUM, TAU)

        train(sess, env, SMB_Agent)

        ("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()

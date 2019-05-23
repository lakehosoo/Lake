
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
RANDOM_SEED = 12345 * 1

# Pre training steps
MAX_TRA = 0
sample_num = 64

# Max training steps
INITIAL_POLICY_INDEX = 0
AC_PE_TRAINING_INDEX = 0
MAX_EPISODES = 11

gamma = 0.999

# Hyperparameter for the Critic Network
LEARNING_RATE = 0.1*0.0001
EPOCHS = 30
BATCH_NUM = 128
TAU = 0.05
validation_ratio = 0.2

# Size of replay buffer
BUFFER_SIZE = 50000
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
    Buffer_data = Buffer(BUFFER_SIZE, BUFFER_SIZE_TERMINAL, state_dim, action_dim, reward_dim, RANDOM_SEED)

    with open('data1', 'rb') as data_file:
        data1 = pickle.load(data_file)
    Buffer_data.data_add(data1)
    with open('data5', 'rb') as data_file:
        data5 = pickle.load(data_file)
    Buffer_data.data_add(data5)
    with open('data6', 'rb') as data_file:
        data6 = pickle.load(data_file)
    Buffer_data.data_add(data6)

    # Weight taking
    SMB_Agent.get_weight()
    print(33)
    print(len(Buffer_data.database))
    Buffer_data.Make_buffer(BUFFER_SIZE)

    # Initialize exploration noise
    exploration_noise = OUNoise(action_dim)

    MAX_EP_STEPS = int(env.Total_simulation_time /env.st) # 1*8

    decay = 1
    reward_exp = -1

    # Pretraining,
    for i in range(MAX_TRA):

        print("Pretraining %d" % i)
        s_batch, a_batch, r_batch, ss_batch = Buffer_data.sample_batch(sample_num)
        aa_batch = SMB_Agent.choose_action(ss_batch)
        x_batch = np.array([np.concatenate((s_batch[ii], a_batch[ii])) for ii in range(len(s_batch))])
        xx_batch = np.array([np.concatenate((ss_batch[ii], aa_batch[ii])) for ii in range(len(ss_batch))])
        target_q = SMB_Agent.predict_V(xx_batch)
        target_batch = r_batch + gamma * target_q
        SMB_Agent.train(x_batch, target_q, validation_ratio)

    # Training with simulation
    for i in range(MAX_EPISODES):

        s2 = time.time()
        product_ep = np.array([0.])
        reward_ep = np.array([0.])
        reward_exp = 0

        action_history = np.array([])
        state_history = np.array([])
        reward_history = np.array([])

        # Position value
        s, action_predict = env.reset()

        decay = decay - 1/(1.2*MAX_EPISODES)
        action_prev = np.array([[0.0191, 0.0191, 0.0135, 0.0135, 0.0165, 0.0165, 0.0105, 0.0105]])
        data = np.array(([np.zeros(2*env.s_dim + env.a_dim + 1)])) # dummy data need....

        for t in range(MAX_EP_STEPS):

            action_noise = np.array([np.random.randn(1)[0] for j in range(4)])
            action_noise = np.array([np.concatenate(([0, 0], action_noise, [0, 0]))])

            action_predict = SMB_Agent.choose_action(s)
            action_predict = 1/scale2*np.reshape(action_predict, [1,-1])

            if i < INITIAL_POLICY_INDEX:
                action_predict = SMB_Agent.choose_rand(s)
                action_predict = 1/scale2*np.reshape(action_predict, [1,-1])
                action = action_prev + action_predict #noise_ini * action_noise

            elif INITIAL_POLICY_INDEX <= i < AC_PE_TRAINING_INDEX:  # 이후에는 actor를 이용하여 simulation
                epsilon = 1 - (i - INITIAL_POLICY_INDEX) / (AC_PE_TRAINING_INDEX - INITIAL_POLICY_INDEX) * 0.99  # epsilon schedule
                action = action_prev + action_predict + noise_tra * action_noise * epsilon

            elif i == MAX_EPISODES - 1:
                action = np.array([[0.0191, 0.0191, 0.0135, 0.0135, 0.0165, 0.0165, 0.0105, 0.0105]])


            else:
                action = action_prev + action_predict

            delaction = action - action_prev
            action_ub = np.array(([[0.0201, 0.0201, 0.0150, 0.0150, 0.0175, 0.0175, 0.0115, 0.0115]]))
            action_lb = np.array(([[0.0181, 0.0181, 0.0125, 0.0125, 0.0150, 0.0150, 0.0095, 0.0095]]))
            # action_lb2 = [[0.0181*50, action[0][0], 0.0120*50, action[0][2], action[0][3], action[0][4], 0.0095*50, action[0][6]]]
            action = np.clip(action, action_lb, action_ub)
            #action = np.clip(action, action_lb2, action_ub)

            for tt in range(env.Nt):
                '''
                # noise1 = np.reshape(exploration_noise.noise(), [1, 1])

                obs_noise1 = 0 * np.random.randn(1) * np.ones((1, 2 * N))[0]
                obs_noise2 = 0 * np.random.randn(1) * np.ones((1, 2 * N))[0]
                obs_noise3 = 0 * np.random.randn(1) * np.ones((1, 2 * N))[0]
                obs_noise4 = 0 * np.random.randn(1) * np.ones((1, 2 * N))[0]
                obs_noise5 = 0 * np.random.randn(1) * np.ones((1, 2 * N))[0]
                obs_noise6 = 0 * np.random.randn(1) * np.ones((1, 2 * N))[0]
                obs_noise7 = 0 * np.random.randn(1) * np.ones((1, 2 * N))[0]
                obs_noise8 = 0 * np.random.randn(1) * np.ones((1, 2 * N))[0]
                obs_noise = np.array([np.concatenate((obs_noise1, obs_noise2, obs_noise3, obs_noise4, obs_noise5, obs_noise6, obs_noise7, obs_noise8, [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]))])
                '''
                ss, r = env.step(s[0], action[0])
                ss = np.array([ss])

                data_set = np.concatenate((s[0], scale2*delaction[0], np.array([r]), ss[0]))
                data = np.concatenate((data, np.array([data_set])))
                Buffer_data.add(np.array([data_set]))

                action_prev = action

                product_ep += (action[0][4] - action[0][3])

                s = ss
                reward_exp += 1
                reward_ep += (gamma**reward_exp)*r

                action_history = np.concatenate((action_history, action[0]))
                state_history = np.concatenate((state_history, ss[0]))
                reward_history = np.concatenate((reward_history, np.array([r])))

            s_batch, a_batch, r_batch, ss_batch = Buffer_data.sample_batch(AC_MINIBATCH_SIZE)
            aa_batch = SMB_Agent.choose_action(ss_batch)
            x_batch = np.array([np.concatenate((s_batch[ii], a_batch[ii])) for ii in range(len(s_batch))])
            xx_batch = np.array([np.concatenate((ss_batch[ii], aa_batch[ii])) for ii in range(len(ss_batch))])
            target_q = SMB_Agent.predict_Vt(xx_batch)
            target_q = np.clip(target_q,np.zeros_like(target_q),np.ones_like(target_q))
            target_batch = r_batch + gamma * target_q
            SMB_Agent.train(x_batch, target_batch, validation_ratio, decay)

            if np.mod(i, 5) == 1:
                SMB_Agent.update_target(TAU)
            # Need terminal..?

            if t == int(MAX_EP_STEPS - 1):
                E_tank = ss[0][16 * N + 10: 16 * N + 12]
                R_tank = ss[0][16 * N + 12: 16 * N + 14]
                print('Episode: %d' %i)
                print('Purity : ')
                print(E_tank[0] / (sum(E_tank)), R_tank[1] / (sum(R_tank)))
                print("Product: %f" %product_ep)
                print("Reward: %f" %reward_ep)
                print('---------')
                print("--- %s seconds ---" % (time.time() - s2))

        count_state = 'state_of_iteration_%d.csv' % i
        count_action = 'action_of_iteration_%d.csv' % i
        count_reward = 'reward_of_iteration_%d.csv' % i
        # count_E_tank = 'E_tank_of_iteration_%d.csv' % i
        # count_R_tank = 'R_tank_of_iteration_%d.csv' % i

        if not os.path.isdir('Data_save'):
            os.mkdir('Data_save')
        now_path = os.getcwd()
        data_path = now_path + '/Data_save'
        os.chdir(data_path)
        np.savetxt(count_state, state_history, delimiter=',', fmt='%s')
        np.savetxt(count_action, action_history, delimiter=',', fmt='%s')
        np.savetxt(count_reward, reward_history, delimiter=',', fmt='%s')
        np.savetxt('product.csv', product_ep, delimiter=',', fmt='%s')
        os.chdir(now_path)
        exploration_noise.reset()

    SMB_Agent.save_weight()

    Buffer_data.data_add(data)
    print(len(Buffer_data.database))
    with open('data1', 'wb') as data_file:
        pickle.dump(Buffer_data.database, data_file)

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

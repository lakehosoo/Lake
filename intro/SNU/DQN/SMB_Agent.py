import tensorflow as tf
import numpy as np
from tensorflow import keras
from random import randint
import random
import pickle

#from selu import selu

# 0.0191, 0.0135, 0.0165, 0.0105
a1_candi = 0
a2_candi = 4000*0.00001*(np.arange(11.0) - 5.)
a3_candi = a2_candi
a4_candi = 0

a_mesh = []
for i in range(len(a2_candi)):
    for ii in range(len(a3_candi)):
        a_mesh += [[0, 0, a2_candi[i], a2_candi[i], a3_candi[ii], a3_candi[ii], 0, 0]]

class agent(object):

    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, input_dim, learning_rate, epochs, batch_num, tau):

        self.sess = sess
        self.input_dim = input_dim
        #self.global_steps = tf.Variable(0, name='global_step', trainable=False)
        #self.learning_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=self.global_steps, decay_steps=12, decay_rate=0.95, staircase=True)
        # decay_steps 12 correct? what is the time step in tensorflow
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_num = batch_num
        self.tau = tau

        # Training options
        #self.callbacks_list = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1,parience=5)]
                               #keras.callbacks.TensorBoard(log_dir='my_log_dir', histogram_freq=1)]
                               #keras.callbacks.EarlyStopping(monitor ='acc', patience = 2),
                               #keras.callbacks.ModelCheckpoint(filepath='my_model.R', monitor='val_loss', save_best_only=True)]

        # Create the V network
        self.V_model = keras.models.Sequential()
        self.V_model.add(keras.layers.Dense(30, activation='tanh', input_shape=(self.input_dim,)))
        self.V_model.add(keras.layers.Dense(15, activation='tanh', input_shape=(self.input_dim,)))
        self.V_model.add(keras.layers.Dense(5, activation='tanh', input_shape=(self.input_dim,)))
        #self.V_model.add(keras.layers.Dense(8, activation='tanh'))
        #self.V_model.add(keras.layers.BatchNormalization())
        self.V_model.add(keras.layers.Dense(1, activation='linear'))
        optimizer = keras.optimizers.Adam(lr=self.learning_rate) #keras.optimizers.Adam(lr=self.learning_rate)
        self.V_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        print(self.V_model.summary())

        # Create V_target Network
        self.Vt_model = keras.models.Sequential()
        self.Vt_model.add(keras.layers.Dense(30, activation='tanh', input_shape=(self.input_dim,)))
        self.Vt_model.add(keras.layers.Dense(15, activation='tanh', input_shape=(self.input_dim,)))
        self.Vt_model.add(keras.layers.Dense(5, activation='tanh', input_shape=(self.input_dim,)))
        #self.Vt_model.add(keras.layers.BatchNormalization())
        #self.Vt_model.add(keras.layers.Dense(3, activation='relu'))
        #self.Vt_model.add(keras.layers.BatchNormalization())
        #self.Vt_model.add(keras.layers.Dense(1, activation='relu'))
        #self.Vt_model.add(keras.layers.BatchNormalization())
        self.Vt_model.add(keras.layers.Dense(1, activation='linear'))
        self.Vt_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        print(self.Vt_model.summary())

        # Op for periodically updating target network with online network
        #self.update_Vt_network_params = [self.Vt_network_params[i].assign(tf.multiply(self.V_network_params[i], self.tau) + tf.multiply(self.Vt_network_params[i], 1. - self.tau))  for i in range(len(self.Vt_network_params))]

        self.saver = tf.train.Saver() # ?

    # Create model of fully connected NN with slim api

    def train(self, x, y, ratio, decay):

        #for layer in self.V_model.layers:
        #    self.ww = layer.get_weights()

        keras.backend.set_value(self.V_model.optimizer.lr, self.learning_rate*decay)
        self.V_model.fit(x, y, epochs=self.epochs, verbose = 0, validation_split=ratio, shuffle = True, batch_size=self.batch_num) #callbacks = self.callbacks_list)

        #print(self.predict_V([[np.ones(102)]]))
        #for layer in self.V_model.layers:
        #    self.w = layer.get_weights()

        #print(np.array(self.ww) - np.array(self.w))

    def save_weight(self):
        a = self.V_model.get_weights()
        b = self.Vt_model.get_weights()
        with open('V_weight', 'wb') as data_file:
            pickle.dump(a, data_file)

        with open('Vt_weight', 'wb') as data_file:
            pickle.dump(b, data_file)

    def get_weight(self):
        with open('V_weight', 'rb') as data_file:
            a = pickle.load(data_file)

        with open('Vt_weight', 'rb') as data_file:
            b = pickle.load(data_file)

        self.V_model.set_weights(a)
        self.Vt_model.set_weights(b)

    def update_target(self, tau):
        self.Vt_model.set_weights(tau*np.array(self.V_model.get_weights()) + (1-tau)*np.array(self.Vt_model.get_weights()))

    def predict_V(self, x):
        y = self.V_model.predict(x)
        return y

    def predict_Vt(self, x):
        y = self.Vt_model.predict(x)
        return y

    def choose_action(self, x):
        action = []
        for i in range(len(x)):
            state = x[i]
            s_mesh = np.array([np.concatenate((state, a_mesh[-1]))])
            #print(self.predict_V(s_mesh))
            for j in range(len(a_mesh)-1):
                s_mesh = np.concatenate((s_mesh, np.array([np.concatenate((state, a_mesh[j]))])))

            V_candi = self.predict_V(s_mesh)
            #print(V_candi)
            action_num = np.argmax(V_candi, 0)
            # print(action_num)
            action += [s_mesh[action_num[0]][-8:]]
            # Should check index, value, and so on!!!
        return action

    '''
    def choose_action(self, x):
        act = np.zeros((len(x),8))
        for i in range(len(x)):
            V_candi = np.zeros(len(a_mesh))
            for ii in range(len(a_mesh)):
                mesh = np.array([np.concatenate((x[i],a_mesh[ii]))])
                V_candi[ii] = self.predict_V(mesh)
            act_num = np.argmax(V_candi,0)
            act[i] = a_mesh[act_num]
        return act
    '''

    def choose_rand(self, states):
        action = np.array([a_mesh[randint(0, len(a_mesh)-1)]])
        return action

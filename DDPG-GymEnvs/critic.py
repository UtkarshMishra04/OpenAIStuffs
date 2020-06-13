import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.initializers import RandomUniform
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, concatenate, LSTM, Reshape, BatchNormalization, Lambda, Flatten

class Critic:

    def __init__(self, state_dim, action_dim, lr, tau):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau, self.lr = tau, lr

        self.model = self.network()
        self.target_model = self.network()
        self.model.compile(Adam(self.lr), 'mse')
        self.target_model.compile(Adam(self.lr), 'mse')
        
        self.action_grads = K.function([self.model.input[0], self.model.input[1]], K.gradients(self.model.output, [self.model.input[1]]))

    def network(self):

        state = Input((self.state_dim,))
        action = Input((self.action_dim,))
        x = Dense(256, activation='relu')(state)
        #x = concatenate([Flatten()(x), action])
        x = concatenate([x, action])
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        out = Dense(1, activation='linear', kernel_initializer=RandomUniform())(x)
        return Model([state, action], out)

    def gradients(self, states, actions):

        return self.action_grads([states, actions])

    def target_predict(self, inp):

        return self.target_model.predict(inp)

    def train_on_batch(self, states, actions, critic_target):

        return self.model.train_on_batch([states, actions], critic_target)

    def transfer_weights(self):
  
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
        self.target_model.set_weights(target_W)

    def save(self, path):
        self.model.save_weights(path + '_critic.h5')
        self.target_model.save_weights(path + '_target_critic.h5')

    def load_weights(self, path):
        self.model.load_weights(path + '_critic.h5')
        self.target_model.load_weights(path + '_target_critic.h5')

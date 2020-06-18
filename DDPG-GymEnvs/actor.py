import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import Input, Dense, Reshape, LSTM, Lambda, BatchNormalization, GaussianNoise, Flatten

tf.compat.v1.disable_eager_execution()

class Actor:

    def __init__(self, env, state_dim, action_dim, action_range, lr, tau):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_range = action_range
        self.tau = tau
        self.lr = lr
        self.model = self.network()
        self.target_model = self.network()
        self.adam_optimizer = self.optimizer()

    def network(self):

        inp = Input(shape = (self.state_dim,))
        x = Dense(256, activation='relu')(inp)
        x = GaussianNoise(1.0)(x)
        #x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = GaussianNoise(1.0)(x)
        x = Dense(64, activation='relu')(x)
        x = GaussianNoise(1.0)(x)
        out = Dense(self.action_dim, activation='tanh', kernel_initializer=RandomUniform())(x)
        out = Lambda(lambda i: i * self.action_range)(out)

        return Model(inp, out)

    def predict(self, state):

        return self.model.predict(np.expand_dims(state, axis=0))
        
    def target_predict(self, inp):

        return self.target_model.predict(inp)

    def transfer_weights(self):

        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
        self.target_model.set_weights(target_W)

    def train(self, states, actions, grads):
        self.adam_optimizer([states, grads])

    def optimizer(self):
        action_gdts = K.placeholder(shape=(None, self.action_dim))
        params_grad = tf.gradients(self.model.output, self.model.trainable_weights, -action_gdts)
        grads = zip(params_grad, self.model.trainable_weights)
        return K.function(inputs=[self.model.input, action_gdts], outputs=[ K.constant(1)],updates=[tf.optimizers.Adam(self.lr).apply_gradients(grads)])

    def save(self, path):
        self.model.save_weights(path + str(self.env.unwrapped.spec.id)+'_actor.h5')
        self.target_model.save_weights(path + '_target_actor.h5')

    def load(self, path):
        self.model.load_weights(path + str(self.env.unwrapped.spec.id)+'_actor.h5')
        self.target_model.load_weights(path + '_target_actor.h5')

import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn.init as init

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Network:

    def __init__(self, state_dim, action_dim, action_range, init_w):

        self.action_range = action_range
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(246, 128)
        self.fc3 = nn.Linear(128,action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self,x):

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        out = out*self.action_range

        return out


class Actor:

    def __init__(self, env, state_dim, action_dim, action_range, lr, tau):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_range = action_range
        self.tau = tau
        self.lr = lr
        self.init_w = 3e-3
        self.model = Network(state_dim, action_dim, action_range, self.init_w)
        self.target_model = Network(state_dim, action_dim, action_range, self.init_w)
        self.adam_optimizer = self.optimizer()

    def predict(self, state):

        return self.model.forward(np.expand_dims(state, axis=0))
        
    def target_predict(self, inp):

        return self.target_model.forward(inp)

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

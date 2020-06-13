import numpy as np
import highway_env
import gym
import pandas as pd
import math

import tensorflow as tf
from keras.models import Sequential, Model, clone_model
from keras.layers import Input, Dense, Activation, Flatten, Concatenate, Add
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import layers
from keras import backend as K

from replay_buffer import ReplayBuffer

class Actor:

    def __init__(self,state_dim,action_dim):
        actor_1 = Input(shape=state_dim)
        flatten = Flatten()(actor_1)
        actor_2 = Dense(128, kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu')(flatten)
        actor_3 = Dense(64, kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu')(actor_2)
        actor_4 = Dense(32, kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu')(actor_3)
        actor_5 = Dense(action_dim, kernel_initializer='random_uniform', bias_initializer='zeros', activation='linear')(actor_4)

        self.actor = Model(inputs=actor_1, outputs=actor_5)

    def get_model(self):
        return self.actor

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.25, theta=.05, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


def preprocess_state(env,state,state_dim):

    return state.reshape((1,)+state_dim)

def select_action(a_val,epsilon):

    prob = np.random.rand()

    if prob <= epsilon:
        action = np.random.randint(a_val.shape[0])
    else:
        action = np.argmax(a_val)

    return action

def huber_loss(y_true, y_pred, clip_delta=1.0):

    error = y_true - y_pred
    cond  = K.abs(error) <= clip_delta

    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

    return K.mean(tf.where(cond, squared_loss, quadratic_loss))

def train(env, actor, actor_noise, state_dim, action_dim):

    num_ep = 500
    max_limit_size = 10000
    min_buffer_size = 150
    batch_size = 128

    Reward = np.zeros(num_ep)

    replay_buffer = ReplayBuffer(buffer_size=max_limit_size,random_seed=123)

    optimize_actor = Adam(learning_rate=0.0001)

    model_actor = actor.get_model()
    target_actor = clone_model(model_actor)

    target_actor.set_weights(model_actor.get_weights())

    model_actor.compile(loss=huber_loss, optimizer=optimize_actor)
    target_actor.compile(loss=huber_loss,optimizer=optimize_actor)

    total_reward = 0

    total_steps = 0

    epsilon = 0.5
    epsilon_decay = 0.99

    for i in range(num_ep):

        reward_ep = 0
        curr_state = env.reset()

        ep_ave_max_q  = 0

        terminal = False

        ep_length = 0

        while terminal==False:

            #print("s raw", curr_state)

            s = preprocess_state(env,curr_state,state_dim)

            #print("state before",s)

            a_val = model_actor.predict(s)[0] + actor_noise()

            #print("action value",a_val)
    
            action = select_action(a_val,epsilon)

            #print(a_val,action)

            #print("selected action",action)

            reshaped_s = np.reshape(s, (1,)+state_dim)
            next_s, r, terminal,_ = env.step(action)

            curr_state = next_s
            #print("next_s raw", curr_state)
            next_s = preprocess_state(env,next_s,state_dim)

            #assert np.linalg.norm(next_s-s)!=0

            #print("next state", next_s)
            #env.render()
            reshaped_a = np.reshape(action, (1,))
            reshaped_ns = np.reshape(np.array(next_s), (1,)+state_dim)
            
            replay_buffer.add(reshaped_s, reshaped_a, r, terminal, reshaped_ns)

            
                        
            if replay_buffer.size() > min_buffer_size:
                s_batch, a_batch, r_batch, t_batch, next_s_batch = replay_buffer.sample_batch(batch_size)

                for k in range(batch_size):

                    target = model_actor.predict(s_batch[k])

                    if t_batch[k]:
                        target[0][a_batch[k]] = r_batch[k]
                    else:
                        target[0][a_batch[k]] = r_batch[k]+0.998*np.amax(target_actor.predict(next_s_batch[k])[0])

                    model_actor.fit(s_batch[k],target,epochs=1,verbose=0)

                    ep_ave_max_q += np.amax(model_actor.predict(s_batch[k]))

            
               
            #curr_state = next_s
            reward_ep += r
            ep_length += 1
            total_steps+=1

                        

            if terminal:


                target_actor.set_weights(model_actor.get_weights())
                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f} | Steps: {:d} | Epsilon: {:4f}'.format(int(reward_ep),i, (ep_ave_max_q / float(ep_length)), ep_length, epsilon))#,(total_reward/float(i+1))))
                #print("Reward:",reward_ep,"Episode:",i,"Max Q:",ep_ave_max_q)
                pd.DataFrame(Reward).to_csv("./results/"+str(env.unwrapped.spec.id)+"reward.csv")

                model_actor.save_weights('./results/'+str(env.unwrapped.spec.id)+'actor_weights.h5')
                model_actor.save('./results/'+str(env.unwrapped.spec.id)+'actor_model.h5')

                epsilon = epsilon*epsilon_decay

        Reward[i] = reward_ep
        total_reward += reward_ep

    pd.DataFrame(Reward).to_csv("./results/"+str(env.unwrapped.spec.id)+"reward.csv")

    model_actor.save_weights('./results/'+str(env.unwrapped.spec.id)+'actor_weights.h5')
    model_actor.save('./results/'+str(env.unwrapped.spec.id)+'actor_model.h5')

def test(env, actor, state_dim, action_dim):

    model_actor = actor.get_model()

    model_actor.load_weights('./results/'+str(env.unwrapped.spec.id)+'actor_weights.h5')

    num_ep = 10

    for i in range(num_ep):

        reward_ep = 0
        curr_state = env.reset()
        terminal = False
        ep_length = 0
        ep_ave_max_q = 0
        epsilon = 0.05

        while terminal==False:

            s = preprocess_state(env,curr_state,state_dim)

            a_val = model_actor.predict(s)[0]
    
            action = select_action(a_val,epsilon)

            next_s, r, terminal,_ = env.step(action)

            env.render()

            curr_state = next_s

            reward_ep += r
            ep_length += 1
                        

            if terminal:

                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f} | Steps: {:d} | Epsilon: {:4f}'.format(int(reward_ep),i, (ep_ave_max_q / float(ep_length)), ep_length, epsilon))








if __name__ == '__main__':

    env = gym.make('highway-v0')

    state_dim = env.observation_space.shape
    action_dim = env.action_space.n

    actor = Actor(state_dim,action_dim)

    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

    #train(env, actor, actor_noise, state_dim, action_dim)

    test(env,actor,state_dim,action_dim)
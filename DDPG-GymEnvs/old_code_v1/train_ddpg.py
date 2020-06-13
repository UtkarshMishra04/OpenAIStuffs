import numpy as np
import roboschool
import gym
import pandas as pd
import math

from keras.models import Sequential, Model, clone_model
from keras.layers import Input, Dense, Activation, Flatten, Concatenate, Add
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import layers
from keras import backend as K

from replay_buffer import ReplayBuffer

class Actor:

    def __init__(self,state_dim,action_dim):
        actor_1 = Input(shape=(state_dim,))
        actor_2 = Dense(64, kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu')(actor_1)
        actor_3 = Dense(64, kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu')(actor_2)
        actor_4 = Dense(action_dim, kernel_initializer='random_uniform', bias_initializer='zeros',activation='tanh')(actor_3)

        self.actor = Model(inputs=actor_1, outputs=actor_4)

    def get_model(self):
        return self.actor

class Critic:

    def __init__(self,state_dim,action_dim):

        #critic_s1 = Input(shape=(1,)+state_dim)
        critic_s1 = Input(shape=(state_dim,))
        #critic_s2 = Flatten()(critic_s1)
        critic_s3 = Dense(128, kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu')(critic_s1)
        #critic_s4 = Dense(60, kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu')(critic_s3)

        critic_a1 = Input(shape=(action_dim,))
        #critic_a2 = Flatten()(critic_a1)
        critic_a3 = Dense(128, kernel_initializer='random_uniform', bias_initializer='zeros', activation='linear')(critic_a1)
        
        critic1 = Add()([critic_s3, critic_a3])
        critic2 = Dense(64, kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu')(critic1)

        out = Dense(1, kernel_initializer='random_uniform', bias_initializer='zeros', activation='linear')(critic2)
        self.critic = Model(inputs=[critic_s1, critic_a1], outputs=out)

        action_gradients = K.gradients(out, critic_a1)

        self.get_action_gradients = K.function(
            inputs=[*self.critic.input, K.learning_phase()],
            outputs=action_gradients)

    def get_model(self):
        return self.critic

    def action_gradients(self,inputs):

        return self.get_action_gradients(inputs)

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

def train(env, actor, critic, actor_noise, state_dim, action_dim):

    num_ep = 1000
    min_limit_size = 10000
    min_buffer_size = 150
    batch_size = 128

    Reward = np.zeros(num_ep)

    replay_buffer = ReplayBuffer(buffer_size=min_limit_size,random_seed=1234)

    optimize_actor = Adam(learning_rate=0.001)
    optimize_critic = Adam(learning_rate=0.01)

    model_actor = actor.get_model()
    model_critic = critic.get_model()
    target_actor = clone_model(model_actor)
    target_critic = clone_model(model_critic)

    target_actor.set_weights(model_actor.get_weights())
    target_critic.set_weights(model_critic.get_weights())

    model_actor.compile(loss='mse', optimizer=optimize_actor)
    model_critic.compile(loss='mse', optimizer=optimize_actor)
    target_critic.compile(loss='mse',optimizer='sgd')
    target_actor.compile(loss='mse',optimizer='sgd')

    total_reward = 0

    for i in range(num_ep):

        reward_ep = 0
        curr_state = env.reset()

        ep_ave_max_q  = 0

        terminal = False

        ep_length = 0

        #env.render()

        while terminal==False:
            s=curr_state
            inp = np.array(s).reshape((1,-1))
            a_val = model_actor.predict(inp)[0] + actor_noise()
            #action = [math.atan(val) for val in a_val]
            eps = np.random.rand()
            if eps < 0.5 and i < 10:
                action = np.random.rand(2)*2-1
            else:
                action = np.clip(a_val,env.action_space.low,env.action_space.high)

            reshaped_s = np.reshape(s, (state_dim,))
            next_s, r, terminal,_ = env.step(np.array(action))
            env.render()
            #next_s[2] = next_s[2]/8.0
            reshaped_a = np.reshape(action, (action_dim,))
            reshaped_ns = np.reshape(np.array(next_s).reshape((1,-1)), (state_dim,))
            
            replay_buffer.add(reshaped_s, reshaped_a, r, terminal, reshaped_ns)

            #print(np.linalg.norm(reshaped_s-reshaped_ns))
            
            if replay_buffer.size() > min_buffer_size:
                s_batch, a_batch, r_batch, t_batch, next_s_batch = replay_buffer.sample_batch(batch_size)

                next_a_batch = target_actor.predict(next_s_batch)
                target_q = target_critic.predict([next_s_batch,next_a_batch])

                
                y_i = []

                for k in range(batch_size):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k]+0.9*target_q[k])


                model_critic.fit([s_batch,a_batch],np.reshape(y_i,(batch_size,1)),verbose=0)
                
                ep_ave_max_q += np.amax(model_critic.predict([s_batch,a_batch]))

                
                a_outs = model_actor.predict(s_batch)
                
                evaluated_gradients = critic.action_gradients([s_batch,a_outs,0])
            
                grads = evaluated_gradients #gradients of output wrt critic action inputs (critic inputs are s_batch and a_outs)

                model_actor.fit(s_batch, grads[0], verbose=0)

                wa_t = np.array(target_actor.get_weights())
                wa = np.array(model_actor.get_weights())
                wc = np.array(model_critic.get_weights())
                wc_t = np.array(target_critic.get_weights())

                target_actor.set_weights(0.001*wa + (1-0.001)*wa_t)
                target_critic.set_weights(0.005*wc + (1-0.005)*wc_t)
            
            curr_state = next_s
            reward_ep += r
            ep_length += 1

            

            if terminal:
                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f} | Steps: {:d}'.format(int(reward_ep),i, (ep_ave_max_q / float(ep_length)), ep_length))#,(total_reward/float(i+1))))
                #print("Reward:",reward_ep,"Episode:",i,"Max Q:",ep_ave_max_q)
                pd.DataFrame(Reward).to_csv("./results/reward.csv")

                model_actor.save_weights('./results/actor_weights.h5')
                model_critic.save_weights('./results/critic_weights.h5')
                model_actor.save('./results/actor_model.h5')
                model_critic.save('./results/critic_model.h5')

        Reward[i] = reward_ep
        total_reward += reward_ep

    pd.DataFrame(Reward).to_csv("./results/reward.csv")

    model_actor.save_weights('./results/actor_weights.h5')
    model_critic.save_weights('./results/critic_weights.h5')
    model_actor.save('./results/actor_model.h5')
    model_critic.save('./results/critic_model.h5')

if __name__ == '__main__':

    env = gym.make('LunarLanderContinuous-v2')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = Actor(state_dim,action_dim)
    critic = Critic(state_dim,action_dim)

    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

    #print(env.action_space.high,env.action_space.low)

    #print(env.reset())
    #print(env.action_space.sample())

    train(env, actor, critic, actor_noise, state_dim, action_dim)
import sys
import argparse
import numpy as np
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
from actor import Actor
from critic import Critic


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

def bellman(rewards, q_values, dones,gamma):

    critic_target = np.asarray(q_values)
    for i in range(q_values.shape[0]):
        if dones[i]:
            critic_target[i] = rewards[i]
        else:
            critic_target[i] = rewards[i] + gamma * q_values[i]
    return critic_target

def train(env, actor, critic, actor_noise, state_dim, action_dim):

    num_ep = 10000
    batch_size = 500
    min_buffer_size = 1000
    buffer_size = 20000
    gamma = 0.95

    Reward = []

    replay_buffer = ReplayBuffer(buffer_size=buffer_size,random_seed=1234)

    total_reward = 0

    total_length = 0

    epsilon = 1
    epsilon_decay = 0.995

    for i in range(num_ep):

        reward_ep = 0
        curr_state = env.reset()

        ep_ave_max_q  = 0

        terminal = False

        ep_length = 0

        #env.render()

        while terminal==False:

            s = curr_state

            if np.random.rand() < epsilon:
                action = np.random.rand(action_dim)*2-1
            else:
                action = actor.predict(s)[0]
                action = np.clip(action+actor_noise(), env.action_space.low, env.action_space.high)

            next_s, r, terminal,_ = env.step(action)

            #env.render()

            replay_buffer.add(s, action, r, terminal, next_s)

            if replay_buffer.size() > min_buffer_size:

                print("Training")

                s_batch, a_batch, r_batch, t_batch, next_s_batch = replay_buffer.sample_batch(batch_size)

                q_values = critic.target_predict([next_s_batch, actor.target_predict(next_s_batch)])

                critic_target = bellman(r_batch, q_values, t_batch,gamma)

                
                critic.train_on_batch(s_batch, a_batch, critic_target)

                ep_ave_max_q += np.amax(critic.target_predict([s_batch,a_batch]))
                
                actions = actor.model.predict(s_batch)
                grads = critic.gradients(s_batch, actions)
                
                actor.train(s_batch, actions, np.array(grads).reshape((-1, action_dim)))
                
                if (total_length+1)%200==0:
                    #print("weights transferred")
                    total_length = 0
                    actor.transfer_weights()
                    critic.transfer_weights()


            curr_state = next_s
            reward_ep += r
            ep_length += 1  
            total_length += 1          

            if terminal:
                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f} | Steps: {:d} | Epsilon: {:.4f}'.format(int(reward_ep),i, (ep_ave_max_q / float(ep_length)), ep_length,epsilon))#,(total_reward/float(i+1))))
                #print("Reward:",reward_ep,"Episode:",i,"Max Q:",ep_ave_max_q)
                pd.DataFrame(np.array(Reward)).to_csv("./results/reward.csv")

                actor.save("./results/")
                critic.save("./results/")

                if epsilon > 0.05:
                    epsilon *= epsilon_decay

        Reward.append(reward_ep)
        total_reward += reward_ep

        if (i+1) % 100==0:
            avg_reward = 0
            for i in range(100):
                avg_reward += Reward[len(Reward)-1-i]

            print("Average of last 100 episodes",avg_reward/100.0)

            if avg_reward/100.0 > 50:
                break

    pd.DataFrame(Reward).to_csv("./results/reward.csv")

    actor.save("./results/")
    critic.save("./results/")

def main(args=None):

    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('--run', type=str, default='train',help="Algorithm to train or test")

    parser.parse_args(args)

    env = gym.make('MountainCarContinuous-v0')

    lr = 0.002
    tau = 0.001

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = Actor(env,state_dim,action_dim,env.action_space.high,0.2 * lr, tau)
    critic = Critic(state_dim,action_dim,lr, tau)


    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

    #print(env.action_space.high,env.action_space.low)

    #print(env.reset())
    #print(env.action_space.sample())
    algo = args[1]

    

    if algo=="train":
        train(env, actor, critic, actor_noise, state_dim, action_dim)

    elif algo == "test":
        actor.load("./results/")

        total_reward = 0

        for i in range(10):

            reward_ep = 0
            curr_state = env.reset()

            ep_ave_max_q  = 0

            terminal = False

            ep_length = 0

            while terminal==False:

                s = curr_state

                
                action = actor.predict(s)[0]
                #action = np.clip(action+actor_noise(), env.action_space.low, env.action_space.high)

                next_s, r, terminal,_ = env.step(action)
                #env.render()

                curr_state = next_s
                reward_ep += r
                ep_length += 1            

                if terminal:
                    print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f} | Steps: {:d}'.format(int(reward_ep),i, (ep_ave_max_q / float(ep_length)), ep_length))#,(total_reward/float(i+1))))
                    total_reward += int(reward_ep)



        print("Average Reward of 10 test episodes",int(total_reward)/10)

if __name__ == "__main__":
    main()

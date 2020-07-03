import sys
import argparse
import numpy as np
import gym
import pandas as pd
import math
import torch

from actor import Actor
from critic import Critic
from replay import ReplayBuffer


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
        print("test")
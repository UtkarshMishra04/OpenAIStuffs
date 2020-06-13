import numpy as np
import gym
from gym import wrappers
import highway_env
import pandas as pd
import math

from keras.models import load_model


def test(env, actor, state_dim):

	num_ep = 5

	Reward = np.zeros(num_ep)

	for i in range(num_ep):

		reward_ep = 0
		#env = wrappers.Monitor(env, "./gym-results",force=True)
		s = env.reset()

		terminal = False

		while terminal==False:
			inp = np.array(s).reshape((1,-1))
			a_val = actor.predict(inp)[0]
			#action = np.array([math.tanh(a_val)])
			action = np.clip(a_val,env.action_space.low,env.action_space.high)
			next_s, r, terminal, info = env.step(action)
			env.render()

			#print(s,next_s)

			s=next_s
			reward_ep += r

			if terminal:
				print('| Reward: {:d} | Episode: {:d}'.format(int(reward_ep),i))


if __name__ == '__main__':

	env = gym.make('LunarLanderContinuous-v2')

	state_dim = env.observation_space.shape[0]

	actor = load_model('./results/actor_model.h5')

	print(actor.summary())
	test(env, actor, state_dim)

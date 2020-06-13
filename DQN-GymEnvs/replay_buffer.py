from collections import deque
import random
import numpy as np

class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=100):
        
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, state, action, reward, terminal, next_state):
        experience = (state, action, reward, terminal, next_state)
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        state_b = np.array([_[0] for _ in batch])
        action_b = np.array([_[1] for _ in batch])
        reward_b = np.array([_[2] for _ in batch])
        terminal_b = np.array([_[3] for _ in batch])
        next_state_b = np.array([_[4] for _ in batch])

        return state_b, action_b, reward_b, terminal_b, next_state_b

    def clear(self):
        self.deque.clear()
        self.count = 0
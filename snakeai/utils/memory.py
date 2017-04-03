import collections
import random

import numpy as np


class ExperienceReplay(object):

    def __init__(self, input_shape, num_actions, memory_size=100):
        self.memory = collections.deque()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.memory_size = memory_size

    def reset(self):
        self.memory = collections.deque()

    def remember(self, state, action, reward, state_next, is_episode_end):
        memory_item = np.concatenate([
            state.flatten(),
            np.array(action).flatten(),
            np.array(reward).flatten(),
            state_next.flatten(),
            1 * np.array(is_episode_end).flatten()
        ])
        self.memory.append(memory_item)
        if 0 < self.memory_size < len(self.memory):
            self.memory.popleft()

    def get_batch(self, model, batch_size, discount_factor=0.9):
        batch_size = min(len(self.memory), batch_size)
        experience = np.array(random.sample(self.memory, batch_size))
        input_dim = np.prod(self.input_shape)

        # Extract [S, a, r, S', end] from experience.
        states = experience[:, 0:input_dim]
        actions = experience[:, input_dim]
        rewards = experience[:, input_dim + 1]
        states_next = experience[:, input_dim + 2:2 * input_dim + 2]
        episode_ends = experience[:, 2 * input_dim + 2]

        # Reshape to match the batch structure.
        states = states.reshape((batch_size, ) + self.input_shape)
        actions = np.cast['int'](actions)
        rewards = rewards.repeat(self.num_actions).reshape((batch_size, self.num_actions))
        states_next = states_next.reshape((batch_size, ) + self.input_shape)
        episode_ends = episode_ends.repeat(self.num_actions).reshape((batch_size, self.num_actions))

        # Predict future state-action values.
        X = np.concatenate([states, states_next], axis=0)
        y = model.predict(X)
        Q_next = np.max(y[batch_size:], axis=1).repeat(self.num_actions).reshape((batch_size, self.num_actions))

        delta = np.zeros((batch_size, self.num_actions))
        delta[np.arange(batch_size), actions] = 1

        targets = (1 - delta) * y[:batch_size] + delta * (rewards + discount_factor * (1 - episode_ends) * Q_next)
        return states, targets

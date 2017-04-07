""" Provides adapters for other AI/RL frameworks, such as OpenAI Gym. """

import json
import numpy as np

from .entities import ALL_SNAKE_ACTIONS
from .environment import Environment


class OpenAIGymEnvAdapter(object):
    """ Converts the Snake environment to OpenAI Gym environment format. """

    def __init__(self, env, action_space, observation_space):
        self.env = env
        self.action_space = OpenAIGymActionSpaceAdapter(action_space)
        self.observation_space = np.array(observation_space)

    def seed(self, value):
        self.env.seed(value)

    def reset(self):
        tsr = self.env.new_episode()
        return tsr.observation

    def step(self, action):
        self.env.choose_action(action)
        timestep_result = self.env.timestep()
        tsr = timestep_result
        return tsr.observation, tsr.reward, tsr.is_episode_end, {}


class OpenAIGymActionSpaceAdapter(object):
    """ Converts the Snake action space to OpenAI Gym action space format. """

    def __init__(self, actions):
        self.actions = np.array(actions)
        self.shape = self.actions.shape
        self.n = len(self.actions)

    def sample(self):
        return np.random.choice(self.actions)


def make_openai_gym_environment(config_filename):
    """
    Create an OpenAI Gym environment for the Snake game.
    
    Args:
        config_filename: JSON config for the Snake game level.

    Returns:
        An instance of OpenAI Gym environment.
    """

    with open(config_filename) as cfg:
        env_config = json.load(cfg)

    env_raw = Environment(config=env_config, verbose=1)
    env = OpenAIGymEnvAdapter(env_raw, ALL_SNAKE_ACTIONS, np.zeros((10, 10)))
    return env

import json
import numpy as np

from .entities import ALL_SNAKE_ACTIONS
from .environment import Environment


class OpenAIGymActionSpaceAdapter(object):
    def __init__(self, actions):
        self.actions = np.array(actions)
        self.shape = self.actions.shape
        self.n = len(self.actions)

    def sample(self):
        return np.random.choice(self.actions)


class OpenAIGymEnvAdapter(object):
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


def make_openai_gym_environment(config_filename):
    with open(config_filename) as cfg:
        env_config = json.load(cfg)

    env_raw = Environment(config=env_config, debug=True)
    env = OpenAIGymEnvAdapter(env_raw, ALL_SNAKE_ACTIONS, np.zeros((10, 10)))
    return env


def make_ql4k_game(config_filename):
    with open(config_filename) as cfg:
        env_config = json.load(cfg)

    env_raw = Environment(config=env_config, debug=True)
    env = env_raw

    return env

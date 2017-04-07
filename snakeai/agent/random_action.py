import random

from snakeai.agent import AgentBase
from snakeai.gameplay.entities import ALL_SNAKE_ACTIONS


class RandomActionAgent(AgentBase):
    """ Represents a Snake agent that takes a random action at every step. """

    def __init__(self):
        pass

    def begin_episode(self):
        pass

    def act(self, observation, reward):
        return random.choice(ALL_SNAKE_ACTIONS)

    def end_episode(self):
        pass

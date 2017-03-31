from snakeai.agent import AgentBase
from snakeai.gameplay.entities import SnakeAction


class HumanAgent(AgentBase):
    def __init__(self):
        pass

    def begin_episode(self):
        pass

    def act(self, observation, reward):
        return SnakeAction.MAINTAIN_DIRECTION

    def end_episode(self):
        pass

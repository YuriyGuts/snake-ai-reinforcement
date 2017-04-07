from snakeai.agent import AgentBase
from snakeai.gameplay.entities import SnakeAction


class HumanAgent(AgentBase):
    """ Represents an agent that is operated by a human (can be used by the GUI). """

    def __init__(self):
        pass

    def begin_episode(self):
        pass

    def act(self, observation, reward):
        # Do nothing by default. If the human player presses a keystroke,
        # this will be overridden on the GUI.
        return SnakeAction.MAINTAIN_DIRECTION

    def end_episode(self):
        pass

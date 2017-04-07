class AgentBase(object):
    def begin_episode(self):
        pass

    def act(self, observation, reward):
        return None

    def end_episode(self):
        pass


from .dqn import DeepQNetworkAgent
from .human import HumanAgent
from .random_action import RandomActionAgent

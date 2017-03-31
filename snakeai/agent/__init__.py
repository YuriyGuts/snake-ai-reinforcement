class AgentBase(object):
    def begin_episode(self):
        pass

    def act(self, observation, action):
        return None

    def end_episode(self):
        pass


from .human import HumanAgent
from .random_action import RandomActionAgent

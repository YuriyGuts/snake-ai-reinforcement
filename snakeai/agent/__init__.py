class AgentBase(object):
    """ Represents an intelligent agent for the Snake environment. """

    def begin_episode(self):
        """ Reset the agent for a new episode. """
        pass

    def act(self, observation, reward):
        """
        Choose the next action to take.

        Args:
            observation: observable state for the current timestep. 
            reward: reward received at the beginning of the current timestep.

        Returns:
            The index of the action to take next.
        """
        return None

    def end_episode(self):
        """ Notify the agent that the episode has ended. """
        pass


from .dqn import DeepQNetworkAgent
from .human import HumanAgent
from .random_action import RandomActionAgent

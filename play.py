import json
import os
import sys

from keras.models import load_model

from snakeai.agent import HumanAgent
from snakeai.agent.dqn import DeepQNetworkAgent
from snakeai.gameplay.environment import Environment
from snakeai.gui import PyGameGUI


def main():
    # In scope of this process, add self to PYTHONPATH to simplify imports.
    current_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(current_dir)

    with open('snakeai/levels/10x10-blank.json') as cfg:
        env_config = json.load(cfg)

    env = Environment(config=env_config)

    num_last_frames = 4
    model = load_model('dqn-final.model')
    agent = DeepQNetworkAgent(model=model, memory_size=-1, num_last_frames=num_last_frames)

    gui = PyGameGUI()
    gui.load_environment(env)
    gui.load_agent(agent)
    gui.run()


if __name__ == '__main__':
    main()

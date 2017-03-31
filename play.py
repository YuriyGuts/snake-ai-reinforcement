import json
import os
import sys

from snakeai.agent import HumanAgent
from snakeai.gameplay.environment import Environment
from snakeai.gui import PyGameGUI


def main():
    # In scope of this process, add self to PYTHONPATH to simplify imports.
    current_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(current_dir)

    with open('snakeai/levels/small_blank.json') as cfg:
        env_config = json.load(cfg)

    env = Environment(config=env_config)
    agent = HumanAgent()

    gui = PyGameGUI()
    gui.load_environment(env)
    gui.load_agent(agent)
    gui.run()


if __name__ == '__main__':
    main()

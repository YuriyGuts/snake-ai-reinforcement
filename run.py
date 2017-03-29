import json
import os
import sys

from snakeai.gameplay.environment import Environment
from snakeai.gui.window import GUI


def main():
    # In scope of this process, add self to PYTHONPATH to simplify imports.
    current_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(current_dir)

    with open('snakeai/levels/small_blank.json') as cfg:
        config = json.load(cfg)
    env = Environment(config=config)
    env.new_episode()

    gui = GUI()
    gui.load_environment(env)
    gui.run()


if __name__ == '__main__':
    main()

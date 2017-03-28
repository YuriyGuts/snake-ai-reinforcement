import json
import os
import sys

from snakeai.gameplay.environment import Environment


def main():
    # In scope of this process, add self to PYTHONPATH to simplify imports.
    current_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(current_dir)

    with open('snakeai/levels/small_blank.json') as cfg:
        config = json.load(cfg)
    env = Environment(config=config)

    print(env.new_episode())
    print(env.timestep())
    print(env.timestep())

    env.take_action(2)
    print(env.timestep())
    print(env.timestep())

    env.take_action(1)
    print(env.timestep())
    print(env.timestep())
    print(env.timestep())

    env.take_action(3)
    print(env.timestep())
    print(env.timestep())
    print(env.timestep())


if __name__ == '__main__':
    main()

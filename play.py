#!/usr/bin/env python3.6

""" Front-end script for replaying the Snake agent's behavior on a batch of episodes. """

import json
import sys
import numpy as np

from snakeai.gameplay.environment import Environment
from snakeai.gui import PyGameGUI
from snakeai.utils.cli import HelpOnFailArgumentParser


def parse_command_line_args(args):
    """ Parse command-line arguments and organize them into a single structured object. """

    parser = HelpOnFailArgumentParser(
        description='Snake AI replay client.',
        epilog='Example: play.py --agent dqn --model dqn-final.model --level 10x10.json'
    )

    parser.add_argument(
        '--interface',
        type=str,
        choices=['cli', 'gui'],
        default='gui',
        help='Interface mode (command-line or GUI).',
    )
    parser.add_argument(
        '--agent',
        required=True,
        type=str,
        choices=['human', 'dqn', 'random'],
        help='Player agent to use.',
    )
    parser.add_argument(
        '--model',
        type=str,
        help='File containing a pre-trained agent model.',
    )
    parser.add_argument(
        '--level',
        required=True,
        type=str,
        help='JSON file containing a level definition.',
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=10,
        help='The number of episodes to run consecutively.',
    )

    return parser.parse_args(args)


def create_snake_environment(level_filename):
    """ Create a new Snake environment from the config file. """

    with open(level_filename) as cfg:
        env_config = json.load(cfg)

    return Environment(config=env_config, verbose=1)


def load_model(filename):
    """ Load a pre-trained agent model. """

    from keras.models import load_model
    return load_model(filename)


def create_agent(name, model):
    """
    Create a specific type of Snake AI agent.
    
    Args:
        name (str): key identifying the agent type.
        model: (optional) a pre-trained model required by certain agents.

    Returns:
        An instance of Snake agent.
    """

    from snakeai.agent import DeepQNetworkAgent, HumanAgent, RandomActionAgent

    if name == 'human':
        return HumanAgent()
    elif name == 'dqn':
        if model is None:
            raise ValueError('A model file is required for a DQN agent.')
        return DeepQNetworkAgent(model=model, memory_size=-1, num_last_frames=4)
    elif name == 'random':
        return RandomActionAgent()

    raise KeyError(f'Unknown agent type: "{name}"')


def play_cli(env, agent, num_episodes=10):
    """
    Play a set of episodes using the specified Snake agent.
    Use the non-interactive command-line interface and print the summary statistics afterwards.
    
    Args:
        env: an instance of Snake environment.
        agent: an instance of Snake agent.
        num_episodes (int): the number of episodes to run.
    """

    fruit_stats = []

    print()
    print('Playing:')

    for episode in range(num_episodes):
        timestep = env.new_episode()
        agent.begin_episode()
        game_over = False

        while not game_over:
            action = agent.act(timestep.observation, timestep.reward)
            env.choose_action(action)
            timestep = env.timestep()
            game_over = timestep.is_episode_end

        fruit_stats.append(env.stats.fruits_eaten)

        summary = 'Episode {:3d} / {:3d} | Timesteps {:4d} | Fruits {:2d}'
        print(summary.format(episode + 1, num_episodes, env.stats.timesteps_survived, env.stats.fruits_eaten))

    print()
    print('Fruits eaten {:.1f} +/- stddev {:.1f}'.format(np.mean(fruit_stats), np.std(fruit_stats)))


def play_gui(env, agent, num_episodes):
    """
    Play a set of episodes using the specified Snake agent.
    Use the interactive graphical interface.
    
    Args:
        env: an instance of Snake environment.
        agent: an instance of Snake agent.
        num_episodes (int): the number of episodes to run.
    """

    gui = PyGameGUI()
    gui.load_environment(env)
    gui.load_agent(agent)
    gui.run(num_episodes=num_episodes)


def main():
    parsed_args = parse_command_line_args(sys.argv[1:])

    env = create_snake_environment(parsed_args.level)
    model = load_model(parsed_args.model) if parsed_args.model is not None else None
    agent = create_agent(parsed_args.agent, model)

    run_player = play_cli if parsed_args.interface == 'cli' else play_gui
    run_player(env, agent, num_episodes=parsed_args.num_episodes)


if __name__ == '__main__':
    main()

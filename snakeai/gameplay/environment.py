import pprint
import random
import time

import numpy as np
from .entities import Snake, Field, CellType, SnakeAction, ALL_SNAKE_ACTIONS


class Environment(object):

    def __init__(self, config, debug=False):
        self.field = Field(level_map=config['field'])
        self.snake = None
        self.fruit = None
        self.rewards = config['rewards']
        self.max_step_limit = config.get('max_step_limit', 1000)
        self.is_game_over = False

        self.timestep_index = 0
        self.current_action = None
        self.prev_distance_to_fruit = 0
        self.stats = EpisodeStatistics()
        self.debug = debug
        self.debug_file = None

    def seed(self, value):
        random.seed(value)
        np.random.seed(value)

    def new_episode(self):
        self.field.create_level()
        self.stats.reset()
        self.timestep_index = 0
        initial_snake_position = self.field.find_snake_head()

        self.snake = Snake(initial_snake_position)
        self.field.place_snake(self.snake)
        self.generate_fruit()
        self.prev_distance_to_fruit = self.get_distance_to_fruit()
        self.current_action = None
        self.is_game_over = False

        result = TimestepResult(
            observation=self.get_observation(),
            reward=0,
            is_episode_end=self.is_game_over
        )

        # Log detailed diagnostic info.
        if self.debug and self.debug_file is None:
            self.debug_file = open(f'snake-env-{time.time()}.log', 'w')
        if self.debug:
            print(result, file=self.debug_file)

        self.stats.record_timestep(self.current_action, result)
        return result

    def get_distance_to_fruit(self):
        diff = self.fruit - self.snake.head
        return abs(diff.x) + abs(diff.y)

    def get_observation(self):
        return np.copy(self.field._cells)

    def choose_action(self, action):
        self.current_action = action
        if action == SnakeAction.TURN_LEFT:
            self.snake.turn_left()
        elif action == SnakeAction.TURN_RIGHT:
            self.snake.turn_right()
        elif action == SnakeAction.REVERSE_DIRECTION:
            self.snake.reverse_direction()

    def timestep(self):
        self.timestep_index += 1
        reward = 0

        old_head = self.snake.head
        old_tail = self.snake.tail

        # Are we about to eat the fruit?
        if self.snake.peek_next_move() == self.fruit:
            self.snake.grow()
            self.generate_fruit()
            old_tail = None
            reward += self.rewards['ate_fruit']
            self.stats.fruits_eaten += 1

        # If not, just move forward.
        else:
            self.snake.move()
            distance_diff = self.prev_distance_to_fruit - self.get_distance_to_fruit()
            self.prev_distance_to_fruit = self.get_distance_to_fruit()

            # Reward the agent for getting closer to the fruit (or penalize otherwise)
            if distance_diff > 0:
                reward += self.rewards['moved_closer_to_fruit'] * abs(distance_diff)
            else:
                reward += self.rewards['moved_further_from_fruit'] * abs(distance_diff)

        self.field.update_snake_footprint(old_head, old_tail, self.snake.head)

        # Hit a wall or own body?
        if not self.is_alive():
            self.field[self.snake.head] = CellType.SNAKE_HEAD
            self.is_game_over = True
            reward = self.rewards['died']

        # Exceeded the limit of moves?
        if self.timestep_index >= self.max_step_limit:
            self.is_game_over = True

        result = TimestepResult(
            observation=self.get_observation(),
            reward=reward,
            is_episode_end=self.is_game_over
        )

        self.stats.record_timestep(self.current_action, result)
        self.stats.timesteps_survived = self.timestep_index

        if self.debug:
            print(result, file=self.debug_file)
            if result.is_episode_end:
                print(self.stats, file=self.debug_file)

        return result

    def generate_fruit(self, position=None):
        if position is None:
            position = self.field.get_random_empty_cell()
        self.field[position] = CellType.FRUIT
        self.fruit = position
        self.prev_distance_to_fruit = self.get_distance_to_fruit()

    def is_alive(self):
        return self.field[self.snake.head] not in (CellType.WALL, CellType.SNAKE_BODY)


class TimestepResult(object):
    def __init__(self, observation, reward, is_episode_end):
        self.observation = observation
        self.reward = reward
        self.is_episode_end = is_episode_end

    def __str__(self):
        field_map = '\n'.join([
            ''.join(str(cell) for cell in row)
            for row in self.observation
        ])
        return '{}\nR = {}   end={}\n'.format(field_map, self.reward, self.is_episode_end)


class EpisodeStatistics():
    def __init__(self):
        self.sum_episode_rewards = 0
        self.fruits_eaten = 0
        self.timesteps_survived = 0
        self.action_counter = None

    def reset(self):
        self.sum_episode_rewards = 0
        self.fruits_eaten = 0
        self.timesteps_survived = 0
        self.action_counter = {
            action: 0
            for action in ALL_SNAKE_ACTIONS
        }

    def record_timestep(self, action, result):
        self.sum_episode_rewards += result.reward
        if action is not None:
            self.action_counter[action] += 1

    def __str__(self):
        return pprint.pformat({
            'sum_episode_rewards': self.sum_episode_rewards,
            'fruits_eaten': self.fruits_eaten,
            'timesteps_survived': self.timesteps_survived,
            'action_counter': self.action_counter,
        })

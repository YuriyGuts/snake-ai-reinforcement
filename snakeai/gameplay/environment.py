import numpy as np
from .entities import Snake, Field, CellType, SnakeAction


class Environment(object):

    def __init__(self, config):
        self.field = Field(size=config['field_size'], level_map=config['field'])
        self.snake = None
        self.fruit = None
        self.rewards = config['rewards']
        self.is_game_over = False

    def new_episode(self):
        self.field.create_level()
        initial_snake_position = self.field.find_snake_head()

        self.snake = Snake(initial_snake_position)
        self.field.place_snake(self.snake)
        self.generate_fruit()
        self.is_game_over = False

        return TimestepResult(
            observation=self.get_observation(),
            reward=0,
            is_episode_end=self.is_game_over
        )

    def get_observation(self):
        return np.copy(self.field.cells)

    def take_action(self, action):
        if action == SnakeAction.TURN_LEFT:
            self.snake.turn_left()
        elif action == SnakeAction.TURN_RIGHT:
            self.snake.turn_right()
        elif action == SnakeAction.REVERSE_DIRECTION:
            self.snake.reverse_direction()

    def timestep(self):
        reward = 0

        old_head = self.snake.head
        old_tail = self.snake.tail

        # Are we standing next to a fruit?
        if self.snake.peek_next_move() == self.fruit:
            self.snake.grow()
            reward += self.rewards['ate_fruit']
        # If not, just move forward.
        else:
            self.snake.move()
            reward += self.rewards['timestep']

        if not self.is_alive():
            self.is_game_over = True
            reward += self.rewards['died']

        self.field.update_snake_footprint(old_head, old_tail, self.snake.head)
        return TimestepResult(
            observation=self.get_observation(),
            reward=reward,
            is_episode_end=self.is_game_over
        )

    def generate_fruit(self):
        fruit_cell = self.field.get_random_blank_cell()
        self.field[fruit_cell] = CellType.FRUIT
        self.fruit = fruit_cell

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

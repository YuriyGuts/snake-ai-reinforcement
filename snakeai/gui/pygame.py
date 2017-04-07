import numpy as np
import pygame

from snakeai.agent import HumanAgent
from snakeai.gameplay.entities import (CellType, SnakeAction, ALL_SNAKE_DIRECTIONS)


class PyGameGUI:
    """ Provides a Snake GUI powered by Pygame. """

    FPS_LIMIT = 60
    AI_TIMESTEP_DELAY = 100
    HUMAN_TIMESTEP_DELAY = 500
    CELL_SIZE = 20

    SNAKE_CONTROL_KEYS = [
        pygame.K_UP,
        pygame.K_LEFT,
        pygame.K_DOWN,
        pygame.K_RIGHT
    ]

    def __init__(self):
        pygame.init()
        self.agent = HumanAgent()
        self.env = None
        self.screen = None
        self.fps_clock = None
        self.timestep_watch = Stopwatch()

    def load_environment(self, environment):
        """ Load the RL environment into the GUI. """
        self.env = environment
        screen_size = (self.env.field.size * self.CELL_SIZE, self.env.field.size * self.CELL_SIZE)
        self.screen = pygame.display.set_mode(screen_size)
        self.screen.fill(Colors.SCREEN_BACKGROUND)
        pygame.display.set_caption('Snake')

    def load_agent(self, agent):
        """ Load the RL agent into the GUI. """
        self.agent = agent

    def render_cell(self, x, y):
        """ Draw the cell specified by the field coordinates. """
        cell_coords = pygame.Rect(
            x * self.CELL_SIZE,
            y * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE,
        )
        if self.env.field[x, y] == CellType.EMPTY:
            pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, cell_coords)
        else:
            color = Colors.CELL_TYPE[self.env.field[x, y]]
            pygame.draw.rect(self.screen, color, cell_coords, 1)

            internal_padding = self.CELL_SIZE // 6 * 2
            internal_square_coords = cell_coords.inflate((-internal_padding, -internal_padding))
            pygame.draw.rect(self.screen, color, internal_square_coords)

    def render(self):
        """ Draw the entire game frame. """
        for x in range(self.env.field.size):
            for y in range(self.env.field.size):
                self.render_cell(x, y)

    def map_key_to_snake_action(self, key):
        """ Convert a keystroke to an environment action. """
        actions = [
            SnakeAction.MAINTAIN_DIRECTION,
            SnakeAction.TURN_LEFT,
            SnakeAction.MAINTAIN_DIRECTION,
            SnakeAction.TURN_RIGHT,
        ]

        key_idx = self.SNAKE_CONTROL_KEYS.index(key)
        direction_idx = ALL_SNAKE_DIRECTIONS.index(self.env.snake.direction)
        return np.roll(actions, -key_idx)[direction_idx]

    def run(self, num_episodes=1):
        """ Run the GUI player for the specified number of episodes. """
        pygame.display.update()
        self.fps_clock = pygame.time.Clock()

        try:
            for episode in range(num_episodes):
                self.run_episode()
                pygame.time.wait(1500)
        except QuitRequestedError:
            pass

    def run_episode(self):
        """ Run the GUI player for a single episode. """

        # Initialize the environment.
        self.timestep_watch.reset()
        timestep_result = self.env.new_episode()
        self.agent.begin_episode()

        is_human_agent = isinstance(self.agent, HumanAgent)
        timestep_delay = self.HUMAN_TIMESTEP_DELAY if is_human_agent else self.AI_TIMESTEP_DELAY

        # Main game loop.
        running = True
        while running:
            action = SnakeAction.MAINTAIN_DIRECTION

            # Handle events.
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if is_human_agent and event.key in self.SNAKE_CONTROL_KEYS:
                        action = self.map_key_to_snake_action(event.key)
                    if event.key == pygame.K_ESCAPE:
                        raise QuitRequestedError

                if event.type == pygame.QUIT:
                    raise QuitRequestedError

            # Update game state.
            timestep_timed_out = self.timestep_watch.time() >= timestep_delay
            human_made_move = is_human_agent and action != SnakeAction.MAINTAIN_DIRECTION

            if timestep_timed_out or human_made_move:
                self.timestep_watch.reset()

                if not is_human_agent:
                    action = self.agent.act(timestep_result.observation, timestep_result.reward)

                self.env.choose_action(action)
                timestep_result = self.env.timestep()

                if timestep_result.is_episode_end:
                    self.agent.end_episode()
                    running = False

            # Render.
            self.render()
            score = self.env.snake.length - self.env.initial_snake_length
            pygame.display.set_caption(f'Snake  [Score: {score:02d}]')
            pygame.display.update()
            self.fps_clock.tick(self.FPS_LIMIT)


class Stopwatch(object):
    """ Measures the time elapsed since the last checkpoint. """

    def __init__(self):
        self.start_time = pygame.time.get_ticks()

    def reset(self):
        """ Set a new checkpoint. """
        self.start_time = pygame.time.get_ticks()

    def time(self):
        """ Get time (in milliseconds) since the last checkpoint. """
        return pygame.time.get_ticks() - self.start_time


class Colors:

    SCREEN_BACKGROUND = (170, 204, 153)
    CELL_TYPE = {
        CellType.WALL: (56, 56, 56),
        CellType.SNAKE_BODY: (105, 132, 164),
        CellType.SNAKE_HEAD: (122, 154, 191),
        CellType.FRUIT: (173, 52, 80),
    }


class QuitRequestedError(RuntimeError):
    """ Gets raised whenever the user wants to quit the game. """
    pass

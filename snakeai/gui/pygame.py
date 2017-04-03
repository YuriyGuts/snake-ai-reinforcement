import numpy as np
import pygame

from snakeai.agent import HumanAgent
from snakeai.gameplay.entities import (CellType, SnakeAction, ALL_SNAKE_DIRECTIONS)


class PyGameGUI:

    FPS_LIMIT = 60
    TIMESTEP_DELAY = 500
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
        self.env = environment
        screen_size = (self.env.field.size * self.CELL_SIZE, self.env.field.size * self.CELL_SIZE)
        self.screen = pygame.display.set_mode(screen_size)
        self.screen.fill(Colors.SCREEN_BACKGROUND)
        pygame.display.set_caption('Snake')

    def load_agent(self, agent):
        self.agent = agent

    def render_cell(self, x, y):
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
        for x in range(self.env.field.size):
            for y in range(self.env.field.size):
                self.render_cell(x, y)

    def map_key_to_snake_action(self, key):
        actions = [
            SnakeAction.MAINTAIN_DIRECTION,
            SnakeAction.TURN_LEFT,
            SnakeAction.MAINTAIN_DIRECTION,
            SnakeAction.TURN_RIGHT,
        ]

        key_idx = self.SNAKE_CONTROL_KEYS.index(key)
        direction_idx = ALL_SNAKE_DIRECTIONS.index(self.env.snake.direction)
        return np.roll(actions, -key_idx)[direction_idx]

    def run(self):
        pygame.display.update()
        self.fps_clock = pygame.time.Clock()
        self.timestep_watch.reset()

        # Initialize the environment.
        timestep_result = self.env.new_episode()
        self.agent.begin_episode()
        is_human_agent = isinstance(self.agent, HumanAgent)

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
                        running = False

                if event.type == pygame.QUIT:
                    running = False

            # Update game state.
            timestep_timed_out = self.timestep_watch.time() >= self.TIMESTEP_DELAY
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
            pygame.display.update()
            self.fps_clock.tick(self.FPS_LIMIT)


class Stopwatch(object):

    def __init__(self):
        self.start_time = pygame.time.get_ticks()

    def reset(self):
        self.start_time = pygame.time.get_ticks()

    def time(self):
        return pygame.time.get_ticks() - self.start_time


class Colors:

    SCREEN_BACKGROUND = (170, 204, 153)
    CELL_TYPE = {
        CellType.WALL: (56, 56, 56),
        CellType.SNAKE_BODY: (105, 132, 164),
        CellType.SNAKE_HEAD: (122, 154, 191),
        CellType.FRUIT: (173, 52, 80),
    }

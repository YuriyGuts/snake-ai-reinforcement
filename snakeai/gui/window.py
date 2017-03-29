import numpy as np
import pygame

from snakeai.gameplay.entities import CellType, SnakeAction, SnakeDirection


class GUI:

    FPS_LIMIT = 60
    TIMESTEP_DELAY = 300
    CELL_SIZE = 20

    SNAKE_DIRECTIONS = [
        SnakeDirection.NORTH,
        SnakeDirection.EAST,
        SnakeDirection.SOUTH,
        SnakeDirection.WEST
    ]
    SNAKE_CONTROL_KEYS = [
        pygame.K_UP,
        pygame.K_LEFT,
        pygame.K_DOWN,
        pygame.K_RIGHT
    ]
    SNAKE_ACTIONS = np.array([
        SnakeAction.MAINTAIN_DIRECTION,
        SnakeAction.TURN_LEFT,
        SnakeAction.REVERSE_DIRECTION,
        SnakeAction.TURN_RIGHT
    ])

    def __init__(self):
        pygame.init()
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
        key_idx = self.SNAKE_CONTROL_KEYS.index(key)
        direction_idx = self.SNAKE_DIRECTIONS.index(self.env.snake.direction)
        return np.roll(self.SNAKE_ACTIONS, -key_idx)[direction_idx]

    def run(self):
        pygame.display.update()
        self.fps_clock = pygame.time.Clock()
        self.timestep_watch.reset()

        # Main game loop.
        running = True
        while running:
            # Retrieve game state.
            action = SnakeAction.MAINTAIN_DIRECTION

            # Handle events.
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key in self.SNAKE_CONTROL_KEYS:
                        action = self.map_key_to_snake_action(event.key)
                    if event.key == pygame.K_ESCAPE:
                        running = False

                if event.type == pygame.QUIT:
                    running = False

            # Update game state.
            if self.timestep_watch.time() >= self.TIMESTEP_DELAY or action != SnakeAction.MAINTAIN_DIRECTION:
                self.timestep_watch.reset()

                self.env.take_action(action)
                timestep_result = self.env.timestep()

                if timestep_result.is_episode_end:
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

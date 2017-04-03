import itertools
import random

import numpy as np
from collections import deque, namedtuple


class Point(namedtuple('PointTuple', ['x', 'y'])):
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)


class CellType(object):
    EMPTY = 0.0
    FRUIT = 1.0
    SNAKE_HEAD = 3.0
    SNAKE_BODY = 3.5
    WALL = 4.0


class SnakeDirection(object):
    NORTH = Point(0, -1)
    EAST = Point(1, 0)
    SOUTH = Point(0, 1)
    WEST = Point(-1, 0)


ALL_SNAKE_DIRECTIONS = [
    SnakeDirection.NORTH,
    SnakeDirection.EAST,
    SnakeDirection.SOUTH,
    SnakeDirection.WEST,
]


class SnakeAction(object):
    MAINTAIN_DIRECTION = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2


ALL_SNAKE_ACTIONS = [
    SnakeAction.MAINTAIN_DIRECTION,
    SnakeAction.TURN_LEFT,
    SnakeAction.TURN_RIGHT,
]


class Snake(object):

    def __init__(self, start_coord, length=3):
        # Place the snake vertically, heading north.
        self.body = deque([
            Point(start_coord.x, start_coord.y + i)
            for i in range(length)
        ])
        self.direction = SnakeDirection.NORTH
        self.tail_direction = SnakeDirection.SOUTH
        self.directions = ALL_SNAKE_DIRECTIONS

    @property
    def head(self):
        return self.body[0]

    @property
    def tail(self):
        return self.body[-1]

    @property
    def length(self):
        return len(self.body)

    def peek_next_move(self):
        return self.head + self.direction

    def turn_left(self):
        direction_idx = self.directions.index(self.direction)
        self.direction = self.directions[direction_idx - 1]

    def turn_right(self):
        direction_idx = self.directions.index(self.direction)
        self.direction = self.directions[(direction_idx + 1) % len(self.directions)]

    def grow(self):
        self.body.appendleft(self.peek_next_move())

    def move(self):
        self.body.appendleft(self.peek_next_move())
        self.tail_direction = self.body[-1] - self.body[-2]
        self.body.pop()


class Field(object):

    def __init__(self, level_map=None):
        self.level_map = level_map
        self._cells = None
        self._empty_cells = set()
        self._level_map_to_cell_type = {
            'S': CellType.SNAKE_HEAD,
            's': CellType.SNAKE_BODY,
            '#': CellType.WALL,
            'O': CellType.FRUIT,
            '.': CellType.EMPTY,
        }
        self._cell_type_to_level_map = {
            cell_type: symbol
            for symbol, cell_type in self._level_map_to_cell_type.items()
        }

    def __getitem__(self, point):
        x, y = point
        return self._cells[y, x]

    def __setitem__(self, point, cell_type):
        x, y = point
        self._cells[y, x] = cell_type

        # Do some internal bookkeeping to not rely on random picking.
        if cell_type == CellType.EMPTY:
            self._empty_cells.add(point)
        else:
            if point in self._empty_cells:
                self._empty_cells.remove(point)

    def __str__(self):
        return '\n'.join(
            ''.join(self._cell_type_to_level_map[cell] for cell in row)
            for row in self._cells
        )

    @property
    def size(self):
        return len(self.level_map)

    def create_level(self):
        try:
            self._cells = np.array([
                [self._level_map_to_cell_type[symbol] for symbol in line]
                for line in self.level_map
            ])
            self._empty_cells = {
                Point(x, y)
                for y in range(self.size)
                for x in range(self.size)
                if self[(x, y)] == CellType.EMPTY
            }
        except KeyError as err:
            raise ValueError(f'Unknown level map symbol: "{err.args[0]}"')

    def find_snake_head(self):
        for y in range(self.size):
            for x in range(self.size):
                if self[(x, y)] == CellType.SNAKE_HEAD:
                    return Point(x, y)
        raise ValueError('Initial snake position not specified on the level map')

    def get_random_empty_cell(self):
        return random.choice(list(self._empty_cells))

    def place_snake(self, snake):
        self[snake.head] = CellType.SNAKE_HEAD
        for snake_cell in itertools.islice(snake.body, 1, len(snake.body)):
            self[snake_cell] = CellType.SNAKE_BODY

    def update_snake_footprint(self, old_head, old_tail, new_head):
        # Update field cells according to the new snake position.
        # Environment must be as fast as possible to speed up agent training.
        # Therefore, we'll sacrifice some duplication of information between
        # the snake body and the field just to execute timesteps faster.
        self[old_head] = CellType.SNAKE_BODY

        # If we've grown at this step, the tail cell shouldn't move.
        if old_tail:
            self[old_tail] = CellType.EMPTY

        # Support the case when we're chasing own tail.
        if self[new_head] not in (CellType.WALL, CellType.SNAKE_BODY) or new_head == old_tail:
            self[new_head] = CellType.SNAKE_HEAD

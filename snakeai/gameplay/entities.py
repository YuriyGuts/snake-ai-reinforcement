import itertools
import numpy as np

from collections import deque, namedtuple


class Point(namedtuple('PointTuple', ['x', 'y'])):
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)


class CellType(object):
    EMPTY = 0
    SNAKE_BODY = 1
    SNAKE_HEAD = 2
    FRUIT = 3
    WALL = 4


class SnakeDirection(object):
    NORTH = Point(0, -1)
    EAST = Point(1, 0)
    SOUTH = Point(0, 1)
    WEST = Point(-1, 0)


class SnakeAction(object):
    MAINTAIN_DIRECTION = 0
    REVERSE_DIRECTION = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3


class Snake(object):

    def __init__(self, start_coord, length=3):
        # Place the snake vertically, heading north.
        self.body = deque([
            Point(start_coord.x, start_coord.y + i)
            for i in range(length)
        ])
        self.direction = SnakeDirection.NORTH
        self.tail_direction = SnakeDirection.SOUTH
        self.directions = [
            SnakeDirection.NORTH,
            SnakeDirection.EAST,
            SnakeDirection.SOUTH,
            SnakeDirection.WEST,
        ]

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

    def reverse_direction(self):
        self.body = deque(reversed(self.body))
        self.direction = self.tail_direction

    def grow(self):
        self.body.appendleft(self.peek_next_move())

    def move(self):
        self.body.appendleft(self.peek_next_move())
        self.tail_direction = self.body[-1] - self.body[-2]
        self.body.pop()


class Field(object):

    def __init__(self, size=10, level_map=None):
        self.size = size
        self.cells = None
        self.level_map = level_map

    def __getitem__(self, point):
        x, y = point
        return self.cells[y, x]

    def __setitem__(self, point, cell_type):
        x, y = point
        self.cells[y, x] = cell_type

    def _map_symbol_to_cell_type(self, symbol):
        if symbol == 'S':
            return CellType.SNAKE_HEAD
        elif symbol == '#':
            return CellType.WALL
        elif symbol == '.':
            return CellType.EMPTY
        else:
            raise ValueError('Invalid level cell type: "{}"'.format(symbol))

    def create_level(self):
        self.cells = np.array([
            [self._map_symbol_to_cell_type(symbol) for symbol in line]
            for line in self.level_map
        ])

    def find_snake_head(self):
        for y in range(self.size):
            for x in range(self.size):
                if self[(x, y)] == CellType.SNAKE_HEAD:
                    return Point(x, y)
        raise ValueError('Initial snake position not specified on the level map')

    def get_random_blank_cell(self):
        while True:
            fruit_coords = np.random.randint(low=0, high=self.size, size=2)
            if self[fruit_coords] == CellType.EMPTY:
                return tuple(fruit_coords)

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

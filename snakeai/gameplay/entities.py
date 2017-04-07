import itertools
import random

import numpy as np
from collections import deque, namedtuple


class Point(namedtuple('PointTuple', ['x', 'y'])):
    """ Represents a 2D point with named axes. """

    def __add__(self, other):
        """ Add two points coordinate-wise. """
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        """ Subtract two points coordinate-wise. """
        return Point(self.x - other.x, self.y - other.y)


class CellType(object):
    """ Defines all types of cells that can be found in the game. """

    EMPTY = 0
    FRUIT = 1
    SNAKE_HEAD = 2
    SNAKE_BODY = 3
    WALL = 4


class SnakeDirection(object):
    """ Defines all possible directions the snake can take, as well as the corresponding offsets. """

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
    """ Defines all possible actions the agent can take in the environment. """

    MAINTAIN_DIRECTION = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2


ALL_SNAKE_ACTIONS = [
    SnakeAction.MAINTAIN_DIRECTION,
    SnakeAction.TURN_LEFT,
    SnakeAction.TURN_RIGHT,
]


class Snake(object):
    """ Represents the snake that has a position, can move, and change directions. """

    def __init__(self, start_coord, length=3):
        """
        Create a new snake.
        
        Args:
            start_coord: A point representing the initial position of the snake. 
            length: An integer specifying the initial length of the snake.
        """
        # Place the snake vertically, heading north.
        self.body = deque([
            Point(start_coord.x, start_coord.y + i)
            for i in range(length)
        ])
        self.direction = SnakeDirection.NORTH
        self.directions = ALL_SNAKE_DIRECTIONS

    @property
    def head(self):
        """ Get the position of the snake's head. """
        return self.body[0]

    @property
    def tail(self):
        """ Get the position of the snake's tail. """
        return self.body[-1]

    @property
    def length(self):
        """ Get the current length of the snake. """
        return len(self.body)

    def peek_next_move(self):
        """ Get the point the snake will move to at its next step. """
        return self.head + self.direction

    def turn_left(self):
        """ At the next step, take a left turn relative to the current direction. """
        direction_idx = self.directions.index(self.direction)
        self.direction = self.directions[direction_idx - 1]

    def turn_right(self):
        """ At the next step, take a right turn relative to the current direction. """
        direction_idx = self.directions.index(self.direction)
        self.direction = self.directions[(direction_idx + 1) % len(self.directions)]

    def grow(self):
        """ Grow the snake by 1 block from the head. """
        self.body.appendleft(self.peek_next_move())

    def move(self):
        """ Move the snake 1 step forward, taking the current direction into account. """
        self.body.appendleft(self.peek_next_move())
        self.body.pop()


class Field(object):
    """ Represents the playing field for the Snake game. """

    def __init__(self, level_map=None):
        """
        Create a new Snake field.
        
        Args:
            level_map: a list of strings representing the field objects (1 string per row).
        """
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
        """ Get the type of cell at the given point. """
        x, y = point
        return self._cells[y, x]

    def __setitem__(self, point, cell_type):
        """ Update the type of cell at the given point. """
        x, y = point
        self._cells[y, x] = cell_type

        # Do some internal bookkeeping to not rely on random selection of blank cells.
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
        """ Get the size of the field (size == width == height). """
        return len(self.level_map)

    def create_level(self):
        """ Create a new field based on the level map. """
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
        """ Find the snake's head on the field. """
        for y in range(self.size):
            for x in range(self.size):
                if self[(x, y)] == CellType.SNAKE_HEAD:
                    return Point(x, y)
        raise ValueError('Initial snake position not specified on the level map')

    def get_random_empty_cell(self):
        """ Get the coordinates of a random empty cell. """
        return random.choice(list(self._empty_cells))

    def place_snake(self, snake):
        """ Put the snake on the field and fill the cells with its body. """
        self[snake.head] = CellType.SNAKE_HEAD
        for snake_cell in itertools.islice(snake.body, 1, len(snake.body)):
            self[snake_cell] = CellType.SNAKE_BODY

    def update_snake_footprint(self, old_head, old_tail, new_head):
        """
        Update field cells according to the new snake position.
        
        Environment must be as fast as possible to speed up agent training.
        Therefore, we'll sacrifice some duplication of information between
        the snake body and the field just to execute timesteps faster.
        
        Args:
            old_head: position of the head before the move. 
            old_tail: position of the tail before the move.
            new_head: position of the head after the move.
        """
        self[old_head] = CellType.SNAKE_BODY

        # If we've grown at this step, the tail cell shouldn't move.
        if old_tail:
            self[old_tail] = CellType.EMPTY

        # Support the case when we're chasing own tail.
        if self[new_head] not in (CellType.WALL, CellType.SNAKE_BODY) or new_head == old_tail:
            self[new_head] = CellType.SNAKE_HEAD

from snakeai.gameplay.entities import Point, Snake, SnakeDirection


def test_place_snake_with_default_length_applies_default_layout():
    snake = Snake(Point(4, 2))
    assert snake.head == Point(4, 2)
    assert snake.tail == Point(4, 4)
    assert snake.length == 3
    assert snake.direction == SnakeDirection.NORTH
    assert list(snake.body) == [(4, 2), (4, 3), (4, 4)]


def test_place_snake_with_custom_length_respects_length():
    snake = Snake(Point(5, 1), length=5)
    assert snake.head == Point(5, 1)
    assert snake.tail == Point(5, 5)
    assert snake.length == 5
    assert list(snake.body) == [(5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]


def test_peek_next_move_respects_current_direction():
    snake = Snake(Point(2, 5))

    snake.direction = SnakeDirection.NORTH
    assert snake.peek_next_move() == (2, 4)

    snake.direction = SnakeDirection.EAST
    assert snake.peek_next_move() == (3, 5)

    snake.direction = SnakeDirection.SOUTH
    assert snake.peek_next_move() == (2, 6)

    snake.direction = SnakeDirection.WEST
    assert snake.peek_next_move() == (1, 5)


def test_grow_extends_head_in_current_direction():
    snake = Snake(Point(3, 5))

    snake.direction = SnakeDirection.NORTH
    snake.grow()
    assert snake.length == 4
    assert snake.head == (3, 4)
    assert snake.tail == (3, 7)

    snake.direction = SnakeDirection.WEST
    snake.grow()
    assert snake.length == 5
    assert snake.head == (2, 4)
    assert snake.tail == (3, 7)

    snake.direction = SnakeDirection.SOUTH
    snake.grow()
    assert snake.length == 6
    assert snake.head == (2, 5)
    assert snake.tail == (3, 7)

    snake.grow()
    snake.grow()
    snake.grow()

    snake.direction = SnakeDirection.EAST
    snake.grow()
    assert snake.length == 10
    assert snake.head == (3, 8)
    assert snake.tail == (3, 7)
    assert list(snake.body) == [
        (3, 8), (2, 8), (2, 7), (2, 6), (2, 5),
        (2, 4), (3, 4), (3, 5), (3, 6), (3, 7),
    ]


def test_single_move_moves_entire_body():
    snake = Snake(Point(4, 5))
    snake.move()
    assert snake.head == (4, 4)
    assert snake.length == 3
    assert snake.tail == (4, 6)
    assert list(snake.body) == [(4, 4), (4, 5), (4, 6)]


def test_multiple_moves_respect_direction():
    snake = Snake(Point(4, 5), length=5)
    snake.move()

    snake.direction = SnakeDirection.WEST
    snake.move()
    snake.move()

    snake.direction = SnakeDirection.SOUTH
    snake.move()

    assert list(snake.body) == [(2, 5), (2, 4), (3, 4), (4, 4), (4, 5)]


def test_turn_right_turns_relatively_to_current_direction():
    snake = Snake(Point(3, 5))

    snake.direction = SnakeDirection.NORTH
    snake.turn_right()
    assert snake.direction == SnakeDirection.EAST
    snake.turn_right()
    assert snake.direction == SnakeDirection.SOUTH
    snake.turn_right()
    assert snake.direction == SnakeDirection.WEST
    snake.turn_right()
    assert snake.direction == SnakeDirection.NORTH


def test_turn_left_turns_relatively_to_current_direction():
    snake = Snake(Point(3, 5))

    snake.direction = SnakeDirection.NORTH
    snake.turn_left()
    assert snake.direction == SnakeDirection.WEST
    snake.turn_left()
    assert snake.direction == SnakeDirection.SOUTH
    snake.turn_left()
    assert snake.direction == SnakeDirection.EAST
    snake.turn_left()
    assert snake.direction == SnakeDirection.NORTH

import pytest
from snakeai.gameplay.entities import Field, Snake


small_level_map = [
    '#######',
    '#.....#',
    '#.....#',
    '#..S..#',
    '#.....#',
    '#.....#',
    '#######',
]


def test_field_created_from_map_renders_same_map():
    field = Field(small_level_map)
    field.create_level()
    assert str(field).split('\n') == small_level_map


def test_find_snake_head_snake_present_finds_it():
    field = Field(small_level_map)
    field.create_level()
    assert field.find_snake_head() == (3, 3)


def test_find_snake_head_no_snake_on_map_throws():
    field = Field([
        '#####',
        '#...#',
        '#...#',
        '#...#',
        '#####',
    ])
    field.create_level()
    with pytest.raises(ValueError):
        field.find_snake_head()


def test_place_snake_given_position_on_map_places_correctly():
    field = Field(small_level_map)
    field.create_level()

    snake = Snake(field.find_snake_head(), length=3)
    field.place_snake(snake)
    assert str(field).split('\n') == [
        '#######',
        '#.....#',
        '#.....#',
        '#..S..#',
        '#..s..#',
        '#..s..#',
        '#######',
    ]


def test_get_random_empty_cell_many_available_returns_any():
    field = Field([
        '#####',
        '#.Ss#',
        '#..s#',
        '#.ss#',
        '#####',
    ])
    field.create_level()
    assert field.get_random_empty_cell() in ((1, 1), (1, 2), (2, 2), (1, 3))


def test_get_random_empty_cell_only_one_cell_left_returns_it():
    field = Field([
        '#####',
        '#Sss#',
        '#O.s#',
        '#sss#',
        '#####',
    ])
    field.create_level()
    assert field.get_random_empty_cell() == (2, 2)


def test_get_random_empty_cell_no_cells_left_throws():
    field = Field([
        '####',
        '#Ss#',
        '#ss#',
        '####',
    ])
    field.create_level()
    with pytest.raises(IndexError):
        field.get_random_empty_cell()

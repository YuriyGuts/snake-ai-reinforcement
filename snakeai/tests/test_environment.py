import json
import os

from snakeai.gameplay.entities import SnakeAction
from snakeai.gameplay.environment import Environment


def get_env_config_file(name):
    level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'levels'))
    return os.path.join(level_dir, name) + ".json"


def load_env(name):
    with open(get_env_config_file(name)) as cfg:
        env_config = json.load(cfg)
        return Environment(config=env_config, verbose=0)


def test_env_on_first_episode_has_consistent_initial_state():
    env = load_env('10x10-blank')
    env.new_episode()

    assert env.field.size == 10
    assert env.snake.head == (5, 4)
    assert env.timestep_index == 0
    assert env.fruit is not None


def test_env_idle_run_reports_correct_timesteps():
    env = load_env('10x10-blank')

    # This makes the fruit appear exactly 2 steps away from the snake,
    # and the next one appear far to the southwest.
    # TODO: This seems dirty and might not be portable across machines.
    env.seed(228)
    tsr = env.new_episode()
    print(tsr)

    assert tsr.reward == 0
    assert not tsr.is_episode_end

    env.choose_action(SnakeAction.MAINTAIN_DIRECTION)
    tsr = env.timestep()
    assert tsr.reward == 0
    assert not tsr.is_episode_end

    tsr = env.timestep()
    assert tsr.reward == 4
    assert not tsr.is_episode_end

    tsr = env.timestep()
    assert tsr.reward == 0
    assert not tsr.is_episode_end

    tsr = env.timestep()
    assert tsr.reward == -1
    assert tsr.is_episode_end

    assert env.stats.sum_episode_rewards == 3
    assert env.stats.timesteps_survived == 4
    assert env.stats.termination_reason == 'hit_wall'
    assert env.stats.action_counter == {
        SnakeAction.MAINTAIN_DIRECTION: 4,
        SnakeAction.TURN_LEFT: 0,
        SnakeAction.TURN_RIGHT: 0,
    }


def test_env_bite_own_tail_reports_game_over():
    env = load_env('10x10-blank')

    # Make 2 consecutive fruits appear directly on our path.
    env.seed(143)
    tsr = env.new_episode()
    print(tsr)

    assert tsr.reward == 0
    assert not tsr.is_episode_end

    # Collect 2 fruits.
    env.choose_action(SnakeAction.MAINTAIN_DIRECTION)
    tsr = env.timestep()
    assert tsr.reward == 0
    assert not tsr.is_episode_end

    tsr = env.timestep()
    assert tsr.reward == 4
    assert not tsr.is_episode_end

    tsr = env.timestep()
    assert tsr.reward == 5
    assert not tsr.is_episode_end

    # Turn right three times to bite own tail.
    env.choose_action(SnakeAction.TURN_RIGHT)
    tsr = env.timestep()
    assert tsr.reward == 0
    assert not tsr.is_episode_end

    env.choose_action(SnakeAction.TURN_RIGHT)
    tsr = env.timestep()
    assert tsr.reward == 0
    assert not tsr.is_episode_end

    env.choose_action(SnakeAction.TURN_RIGHT)
    tsr = env.timestep()
    assert tsr.reward == -1
    assert tsr.is_episode_end

    assert env.stats.sum_episode_rewards == 8
    assert env.stats.timesteps_survived == 6
    assert env.stats.termination_reason == 'hit_own_body'
    assert env.stats.action_counter == {
        SnakeAction.MAINTAIN_DIRECTION: 3,
        SnakeAction.TURN_LEFT: 0,
        SnakeAction.TURN_RIGHT: 3,
    }


def test_env_timestep_limit_exceeded_fails_gracefully():
    env = load_env('10x10-blank')
    env.new_episode()

    for i in range(env.max_step_limit - 1):
        env.choose_action(SnakeAction.TURN_RIGHT)
        tsr = env.timestep()
        assert not tsr.is_episode_end

    env.choose_action(SnakeAction.TURN_RIGHT)
    tsr = env.timestep()
    assert tsr.is_episode_end


def test_env_when_new_episode_starts_resets_previous_state():
    env = load_env('10x10-blank')
    env.new_episode()

    for i in range(env.max_step_limit - 1):
        env.choose_action(SnakeAction.TURN_RIGHT)
        env.timestep()

    tsr = env.timestep()
    assert tsr.is_episode_end
    assert env.stats.timesteps_survived == env.max_step_limit
    assert env.stats.termination_reason == 'timestep_limit_exceeded'

    env.new_episode()

    assert env.field.size == 10
    assert env.snake.head == (5, 4)
    assert env.timestep_index == 0
    assert env.fruit is not None

    assert env.stats.sum_episode_rewards == 0
    assert env.stats.fruits_eaten == 0
    assert env.stats.timesteps_survived == 0
    assert env.stats.termination_reason is None
    assert set(env.stats.action_counter.values()) == {0}

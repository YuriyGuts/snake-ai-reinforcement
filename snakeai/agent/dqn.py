import collections
import numpy as np

from snakeai.agent import AgentBase
from snakeai.utils.memory import ExperienceReplay


class DeepQNetworkAgent(AgentBase):
    def __init__(self, model, num_last_frames=4, memory_size=1000):
        assert model.input_shape[1] == num_last_frames, 'Model input shape should be (num_frames, grid_size, grid_size)'
        assert len(model.output_shape) == 2, 'Model output shape should be (num_samples, num_actions)'

        self.model = model
        self.num_last_frames = num_last_frames
        self.memory = ExperienceReplay((num_last_frames,) + model.input_shape[-2:], model.output_shape[-1], memory_size)
        self.frames = None

    def begin_episode(self):
        self.frames = None

    def get_last_frames(self, observation):
        frame = observation
        if self.frames is None:
            self.frames = collections.deque([frame] * self.num_last_frames)
        else:
            self.frames.append(frame)
            self.frames.popleft()
        return np.expand_dims(self.frames, 0)

    def train(self, env, num_episodes=1000, batch_size=50, discount_factor=0.9, checkpoint_freq=None,
              exploration_range=(1.0, 0.1), exploration_phase_size=0.5):

        max_exploration_rate, min_exploration_rate = exploration_range
        exploration_decay = ((max_exploration_rate - min_exploration_rate) / (num_episodes * exploration_phase_size))
        exploration_rate = max_exploration_rate

        for episode in range(num_episodes):
            timestep = env.new_episode()
            self.begin_episode()
            game_over = False
            loss = 0.0

            state = self.get_last_frames(timestep.observation)
            while not game_over:
                # Explore: take a random action.
                if np.random.random() < exploration_rate:
                    action = np.random.randint(env.num_actions)
                # Exploit: take the best known action for this state.
                else:
                    q = self.model.predict(state)
                    action = np.argmax(q[0])

                # Act on the environment.
                env.choose_action(action)
                timestep = env.timestep()

                # Remember a new piece of experience.
                reward = timestep.reward
                state_next = self.get_last_frames(timestep.observation)
                game_over = timestep.is_episode_end
                experience_item = [state, action, reward, state_next, game_over]
                self.memory.remember(*experience_item)
                state = state_next

                batch = self.memory.get_batch(
                    model=self.model,
                    batch_size=batch_size,
                    discount_factor=discount_factor
                )
                if batch:
                    inputs, targets = batch
                    loss += float(self.model.train_on_batch(inputs, targets))

            if checkpoint_freq and (episode % checkpoint_freq) == 0:
                self.model.save(f'dqn-{episode:08d}.model')

            if exploration_rate > min_exploration_rate:
                exploration_rate -= exploration_decay

            summary = 'Episode {:5d}/{:5d} | Loss {:8.4f} | Exploration {:.2f} | ' + \
                      'Fruits {:2d} | Timesteps {:4d} | Total Reward {:4d}'
            print(summary.format(
                episode + 1, num_episodes, loss, exploration_rate,
                env.stats.fruits_eaten, env.stats.timesteps_survived, env.stats.sum_episode_rewards,
            ))

        self.model.save('dqn-final.model')

    def act(self, observation, reward):
        state = self.get_last_frames(observation)
        q = self.model.predict(state)[0]
        return np.argmax(q)

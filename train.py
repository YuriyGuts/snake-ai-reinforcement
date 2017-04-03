from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

from snakeai.agent.dqn import DeepQNetworkAgent
from snakeai.gameplay.entities import ALL_SNAKE_ACTIONS
from snakeai.gameplay.wrappers import make_ql4k_game


snake = make_ql4k_game('snakeai/levels/10x10-blank.json')

grid_size = 10
num_last_frames = 4
num_actions = len(ALL_SNAKE_ACTIONS)

model = Sequential()

model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), data_format='channels_first', input_shape=(num_last_frames, grid_size, grid_size)))
model.add(Activation('relu'))

model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), data_format='channels_first'))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(num_actions))

model.summary()
model.compile(RMSprop(), 'MSE')

agent = DeepQNetworkAgent(model=model, memory_size=-1, num_last_frames=num_last_frames)
agent.train(snake, batch_size=64, num_episodes=30000, checkpoint_freq=1000, discount_factor=0.8)
agent.play(snake)

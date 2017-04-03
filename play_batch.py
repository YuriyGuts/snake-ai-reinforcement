import json

from keras.models import load_model

from snakeai.agent.dqn import DeepQNetworkAgent
from snakeai.gameplay.environment import Environment


with open('snakeai/levels/10x10-blank.json') as cfg:
    env_config = json.load(cfg)

env = Environment(config=env_config, debug=True)

num_last_frames = 4
model = load_model('dqn-final.model')

agent = DeepQNetworkAgent(model=model, memory_size=-1, num_last_frames=num_last_frames)
agent.play(env)

# snake-ai-reinforcement
AI for Snake game trained from pixels using Deep Reinforcement Learning (DQN).

Contains the tools for training and observing the behavior of the agents, either in CLI or GUI mode.

<img src="https://cloud.githubusercontent.com/assets/2750531/24808769/cc825424-1bc5-11e7-816f-7320f7bda2cf.gif" width="300px"><img src="https://cloud.githubusercontent.com/assets/2750531/24810302/9e4d6e86-1bca-11e7-869b-fc282cd600bb.gif" width="300px">


## Requirements
All components have been written in Python 3.6. Training on GPU is supported but disabled by default. If you have CUDA and would like to use a GPU, use the GPU version of TensorFlow by changing `tensorflow` to `tensorflow-gpu` in the requirements file.

To install all Python dependencies, run:
```
$ make deps
```

## Pre-Trained Models
You can find a few pre-trained DQN agents on the [Releases](https://github.com/YuriyGuts/snake-ai-reinforcement/releases) page. Pass the model file to the `play.py` front-end script (see `play.py -h` for help).

* `dqn-10x10-blank.model`
  
  An agent pre-trained on a blank 10x10 level (`snakeai/levels/10x10-blank.json`).
  
* `dqn-10x10-obstacles.model`

  An agent pre-trained on a 10x10 level with obstacles (`snakeai/levels/10x10-obstacles.json`).


## Training a DQN Agent
To train an agent using the default configuration, run:
```
$ make train
```

The trained model will be checkpointed during the training and saved as `dqn-final.model` afterwards.

Run `train.py` with custom arguments to change the level or the duration of the training (see `train.py -h` for help).

## Playback

The behavior of the agent can be tested either in batch CLI mode where the agent plays a set of episodes and outputs summary statistics, or in GUI mode where you can see each individual step and action.

To test the agent in batch CLI mode, run the following command and check the generated **.csv** file:
```
$ make play
```

To use the GUI mode, run:
```
$ make play-gui
```

To play on your own using the arrow keys (I know you want to), run:
```
$ make play-human
```

## Running Unit Tests
```
$ make test
```

# Implement Q-learning and use this to solve the cartpole-environment
import gym

# Source: https://github.com/JoeSnow7/Reinforcement-Learning/blob/master/Cartpole%20Q-learning.ipynb


# We define a class to contain the learning algorithm
from QLearning import QLearning

env = gym.make("CartPole-v0")
agent = QLearning(env)
agent.train()
agent.run()

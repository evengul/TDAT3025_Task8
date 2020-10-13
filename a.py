# Implement Q-learning and use this to solve the cartpole-environment
import gym
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
env = gym.make('CartPole-v0')
observation = env.reset()

# Source: https://github.com/JoeSnow7/Reinforcement-Learning/blob/master/Cartpole%20Q-learning.ipynb


# We define a class to contain the learning algorithm
class CartPoleQLearning:
    def __init__(self, buckets=(3, 3, 6, 6), num_episodes=500, min_lr=0.1, min_epsilon=0.1, discount=1.0, decay=25):
        # The amount of buckets for position, velocity, angle of the pole and velocity at the tip of the pole,
        # respectively. This makes it a lot easier to create discrete states we can work with
        self.buckets = buckets
        self.num_episodes = num_episodes    # How many episodes we wish to run in sequence.
        # The minimum learning rate of the algorithm. It gets smaller as the episode progresses
        self.min_lr = min_lr
        # The minimum epsilon for the algorithm. Also gets smaller.
        # If a state is smaller than epsilon, we choose a random action
        self.min_epsilon = min_epsilon
        # Used to compute the Q-table
        self.discount = discount
        # Used to ensure epsilon and lr gets smaller as the function runs
        self.decay = decay

        # The gym environment we're running
        self.env = gym.make('CartPole-v0')

        # The initial Q-table. It starts out as zeros, and enters ones at relevant positions
        self.Q_table = np.zeros(self.buckets + (self.env.action_space.n,))

        # The bucket bounds
        self.upper_bounds = [self.env.observation_space.high[0], 0.5,
                             self.env.observation_space.high[2], math.radians(50) / 1.]
        self.lower_bounds = [self.env.observation_space.low[0], -0.5,
                             self.env.observation_space.low[2], -math.radians(50) / 1.]

        # For visualising each step
        self.steps = np.zeros(self.num_episodes)

        # Initialising variables that will be initialised later (gets rid of pycharm warning...)
        self.learning_rate = 0
        self.epsilon = 0

    # Takes any observation/state and discretizes it into a relevant bucket
    def discretize_state(self, obs):
        discretized = list()
        for i in range(len(obs)):
            scaling = ((obs[i] + abs(self.lower_bounds[i])) / (self.upper_bounds[i] - self.lower_bounds[i]))
            new_obs = int(round((self.buckets[i] - 1) * scaling))
            new_obs = min(self.buckets[i] - 1, max(0, new_obs))
            discretized.append(new_obs)
        return tuple(discretized)

    # Selects an action based on a state (A(s))
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q_table[state])

    # Selects action based on (non-discretized) state
    def get_action(self, state, e):
        obs = self.discretize_state(state)
        action_vector = self.Q_table[obs]
        epsilon = self.get_epsilon(e)
        action_vector = self.normalize(action_vector, epsilon)
        return action_vector

    # Normalizes the action vector into probabilities that sum to 1
    def normalize(self, action_vector, epsilon):
        total = sum(action_vector)
        new_vector = (1-epsilon) * action_vector / total
        new_vector += epsilon / 2.0
        return new_vector

    # Update our q-table based on an action taken in the environment and the received reward and new state
    # Q(s, a) -> r, ns
    def update_q(self, state, action, reward, new_state):
        self.Q_table[state][action] += (self.learning_rate * (reward + self.discount * np.max(self.Q_table[new_state])
                                                              - self.Q_table[state][action]))

    # Get the new epsilon value as we decay during the episodes
    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1., 1. - math.log10((t + 1) / self.decay)))

    # Get the new learning rate as we decay during the episodes
    def get_learning_rate(self, t):
        return max(self.min_lr, min(1., 1. - math.log10((t + 1) / self.decay)))

    # Train the agent
    def train(self):
        # For every episode
        for e in range(self.num_episodes):
            # Get a start-position given by the environment
            current_state = self.discretize_state(self.env.reset())
            # Get the learning rate we're using for this episode (declining)
            self.learning_rate = self.get_learning_rate(e)
            # Get the epsilon we're using for this episode (declining)
            self.epsilon = self.get_epsilon(e)
            done = False
            # While the pole has not fallen or the amount of steps is > 198
            while not done:
                self.steps[e] += 1                                          # Add to our visualization array
                action = self.choose_action(current_state)                  # Choose an action from our current state
                obs, reward, done, _ = self.env.step(action)                # Perform the selected action
                new_state = self.discretize_state(obs)                      # Get the new discrete state value
                self.update_q(current_state, action, reward, new_state)     # Update the Q table with our findings
                current_state = new_state                                   # Update state for next step
        print("Done training")

    # Plot the learning of Q. How many steps did each episode need?
    def plot_learning(self):
        sns.lineplot(range(len(self.steps)), self.steps)
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.show()
        t = 0
        for i in range(self.num_episodes):
            if self.steps[i] == 200:
                t += 1
        print("%i episodes were completed" % t)

    # Render a visualisation of a run with the finished Q-table
    def run(self):
        t = 0
        done = False
        current_state = self.discretize_state(self.env.reset())
        while not done:
            self.env.render()
            t += 1
            action = self.choose_action(current_state)
            obs, reward, done, _ = self.env.step(action)
            new_state = self.discretize_state(obs)
            current_state = new_state
        self.env.close()
        return t


# Load an agent, train, plot and run it.
def load_q_learning():
    agent = CartPoleQLearning()
    agent.train()
    agent.plot_learning()
    agent.run()
    return agent


load_q_learning()

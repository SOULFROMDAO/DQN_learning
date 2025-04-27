import gym

from rl_train import train
from agent import Agent

env = gym.make('CartPole-v1')
n_episode = 5000
target_update_interval = 100
n_step = 1000
epsilon_decay_rate = 0.2
epsilon_start = 1.0
epsilon_end = 0.02
learn_rate = 1e-4

def agent_provider(state_size, action_size):
    agent = Agent(state_size, action_size, learn_rate)
    return agent


if __name__ == '__main__':
    train(env, n_episode, n_step, epsilon_decay_rate, epsilon_start, epsilon_end, target_update_interval, agent_provider)

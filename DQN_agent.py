import random
import gym
import os
import numpy as np
import torch
from torch import optim, nn
from utiles import get_device, get_logger

logger = get_logger(__name__)


class AgentMEMO:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory_size = 100000
        self.batch_size = 512

        self.all_state = np.empty((self.memory_size, self.state_size), dtype=np.float32)
        self.all_action = np.random.randint(0, self.action_size, self.memory_size, dtype=np.int64)
        self.all_reward = np.empty(self.memory_size, dtype=np.float32)
        self.all_done = np.random.randint(0, 2, self.memory_size, dtype=np.int32)
        self.all_state_new = np.empty((self.memory_size, self.state_size), dtype=np.float32)
        self.t_memo = 0
        self.t_max = 0

    def memo_add(self, state, action, reward, done, state_new):
        self.all_state[self.t_memo] = state
        self.all_action[self.t_memo] = action
        self.all_reward[self.t_memo] = reward
        self.all_done[self.t_memo] = done
        self.all_state_new[self.t_memo] = state_new
        self.t_memo = (self.t_memo + 1) % self.memory_size
        self.t_max = max(self.t_max, self.t_memo)


    def sample(self):
        if self.t_max >= self.batch_size:
            indexs = random.sample(range(self.t_max), self.batch_size)
        else:
            indexs = range(self.t_max)

        batch_state = []
        batch_action = []
        batch_reward = []
        batch_done = []
        batch_state_new = []

        for index in indexs:
            batch_state.append(self.all_state[index])
            batch_action.append(self.all_action[index])
            batch_reward.append(self.all_reward[index])
            batch_done.append(self.all_done[index])
            batch_state_new.append(self.all_state_new[index])

        batch_state_tensor = torch.from_numpy(np.asarray(batch_state)).to(get_device())
        batch_action_tensor = torch.from_numpy(np.asarray(batch_action)).unsqueeze(-1)
        batch_reward_tensor = torch.from_numpy(np.asarray(batch_reward)).unsqueeze(-1)
        batch_done_tensor = torch.from_numpy(np.asarray(batch_done)).unsqueeze(-1)
        batch_state_new_tensor = torch.from_numpy(np.asarray(batch_state_new)).to(get_device())

        return batch_state_tensor, batch_action_tensor, batch_reward_tensor, batch_done_tensor, batch_state_new_tensor


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hiden_size = 128

        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hiden_size, device=get_device(), dtype=torch.float32),
            nn.Tanh(),
            nn.Linear(self.hiden_size, self.output_size, device=get_device(), dtype=torch.float32),
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=get_device())
        q_value = self(obs_tensor.unsqueeze(0))
        max_q_idx = torch.argmax(input=q_value)
        action = max_q_idx.detach().item()
        return action


class DQNAgent:
    def __init__(self, state_size, action_size, learn_rate, gamma):
        self.state_size = state_size
        self.action_size = action_size
        self.GAMMA = gamma
        self.learn_rate = learn_rate

        self.memo = AgentMEMO(state_size, action_size)
        self.online_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.learn_rate)

    def load_checkpoint(self, train_iter):
        path = f"DQN_model_{train_iter}.pth"
        logger.info(f"Loading model from {path}")
        self.online_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(torch.load(path))

    def save_checkpoint(self, train_iter):
        path = f"DQN_model_{train_iter}.pth"
        logger.info(f"Saving model to {path}")
        torch.save(self.target_net.state_dict(), path)

def initialize_dqn(learn_rate, gamma, game_name='CartPole-v1'):
    env = gym.make(game_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, learn_rate, gamma)
    return env, agent

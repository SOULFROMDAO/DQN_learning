import numpy as np
import random

import torch

from utiles import get_device, get_logger

logger = get_logger(__name__)



def get_epsilon_func(n_episode, n_step, epsilon_decay_rate, epsilon_start, epsilon_end):
    epsilon_decay = epsilon_decay_rate * n_episode * n_step

    def epsilon_func(i_episode, i_step):
        return np.interp(i_episode * n_step + i_step, [0, epsilon_decay], [epsilon_start, epsilon_end])

    return epsilon_func


def train(env, n_episode, n_step, epsilon_decay_rate, epsilon_start, epsilon_end, target_update_interval, agent_provider):
    _agent = agent_provider(state_size=env.observation_space.shape[0], action_size=env.action_space.n)
    epsilon_func = get_epsilon_func(n_episode, n_step, epsilon_decay_rate, epsilon_start, epsilon_end)
    update_reward_storage = np.zeros(target_update_interval)
    state = env.reset()
    for i_episode in range(n_episode):
        episode_reward = 0
        for i_step in range(n_step):
            epsilon = epsilon_func(i_episode, i_step)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = _agent.online_net.act(state)
            state_new, reward, done, info = env.step(action)
            _agent.memo.memo_add(state, action, reward, done, state_new)
            state = state_new

            episode_reward += reward
            if done:
                state = env.reset()
                update_reward_storage[i_episode % target_update_interval] = episode_reward
                break

            batch_state, batch_action, batch_reward, batch_done, batch_state_new = _agent.memo.sample()

            # Compute target
            target_q_values = _agent.target_net(batch_state_new).to('cpu')
            max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
            targets = batch_reward + _agent.GAMMA * max_target_q_values * (1 - batch_done)

            #Comput q_values
            q_values = _agent.online_net(batch_state)
            action_q_values = torch.gather(q_values, 1, index=batch_action.to(get_device())).to('cpu')

            # Compute loss
            loss = torch.nn.functional.smooth_l1_loss(targets, action_q_values)
            loss_for_log = loss.detach().numpy()

            #Gradient desent
            _agent.optimizer.zero_grad()
            loss.backward()
            _agent.optimizer.step()

        if i_episode % target_update_interval == 0 and i_episode != 0:
            _agent.target_net.load_state_dict(_agent.online_net.state_dict())

            #log
            logger.info(f"episode: {i_episode:>5}/{n_episode:>5} | "
                  f"reward: {np.mean(update_reward_storage):3.2f} | "
                  f"max reward: {np.max(update_reward_storage):3.2f} | "
                  f"loss: {loss_for_log:2.8f}")

            valid_agent = False
            if valid_agent:
                valid(update_reward_storage, _agent, state, env, 200)

            update_reward_storage = np.zeros(target_update_interval)


def valid(update_reward_storage, _agent, state, env, threshold):
    if np.mean(update_reward_storage) > threshold:
        while True:
            action = _agent.online_net.act(state)
            state, reward, done, info = env.step(action)
            env.render()

            if done:
                env.reset()
                break
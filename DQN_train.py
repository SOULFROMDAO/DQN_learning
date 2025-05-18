import numpy as np
import random
import torch
from arguments import Epsilon
from utiles import get_device, get_logger

logger = get_logger(__name__)


def get_epsilon_func(n_episode, n_step):
    epsilon_decay = Epsilon.epsilon_decay_rate * n_episode * n_step

    def epsilon_func(i_episode, i_step):
        return np.interp(i_episode * n_step + i_step, [0, epsilon_decay], [Epsilon.epsilon_start, Epsilon.epsilon_end])

    return epsilon_func


def train(env, agent, iteration, steps_per_episode, batch_episode, save_interval, do_log=True):
    n_episode = iteration * batch_episode
    train_iter = 0
    loss_for_log = 0
    plot_data = {}
    while train_iter < iteration:
        update_reward_storage = np.zeros(batch_episode)
        for i_episode in range(batch_episode):

            episode_reward, state, loss_for_log = train_step(steps_per_episode, i_episode,
                                                             batch_episode, train_iter, env, agent, n_episode)

            update_reward_storage[i_episode % batch_episode] = episode_reward
        # update target net
        agent.target_net.load_state_dict(agent.online_net.state_dict())
        train_iter += 1

        if do_log:
            mean_reward = np.mean(update_reward_storage)
            plot_data[train_iter] = mean_reward
            max_reward = np.max(update_reward_storage)
            #log
            logger.info(f"iteration: {train_iter:>5.0f}/{iteration:>5.0f} | "
                  f"mean reward: {mean_reward:>3.2f} | "
                  f"max reward: {max_reward:>3.2f} | "
                  f"loss: {loss_for_log:>2.8f}")

        if train_iter % save_interval == 0:
            agent.save_checkpoint(train_iter)

    agent.save_checkpoint(train_iter)
    return plot_data


def train_step(steps_per_episode, i_episode, batch_episode, train_iter, env, agent, n_episode):
    epsilon_func = get_epsilon_func(n_episode, steps_per_episode)
    episode_reward = 0
    state = env.reset()
    loss_for_log = 0
    for i_step in range(steps_per_episode):
        epsilon = epsilon_func(i_episode + batch_episode * train_iter, i_step)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = agent.online_net.act(state)
        state_new, reward, done, info = env.step(action)
        reward = reward_func(reward, state_new, env)
        agent.memo.memo_add(state, action, reward, done, state_new)
        state = state_new
        episode_reward += 1
        if done:
            break

        batch_state, batch_action, batch_reward, batch_done, batch_state_new = agent.memo.sample()

        # Compute target
        target_q_values = agent.target_net(batch_state_new).to('cpu')
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
        targets = batch_reward + agent.GAMMA * max_target_q_values * (1 - batch_done)

        # Comput q_values
        q_values = agent.online_net(batch_state)
        action_q_values = torch.gather(q_values, 1, index=batch_action.to(get_device())).to('cpu')

        # Compute loss
        loss = torch.nn.functional.smooth_l1_loss(targets, action_q_values)
        loss_for_log = loss.detach().numpy()

        # Gradient desent
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

    return episode_reward, state, loss_for_log


def reward_func(reward, state, env):
    reward += 1 - abs(state[0]) / env.x_threshold
    reward += 1 - abs(state[2]) / env.theta_threshold_radians
    return reward

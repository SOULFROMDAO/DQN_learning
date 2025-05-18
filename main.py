import atexit

from DQN_train import train
from DQN_test import valid
from DQN_agent import initialize_dqn
from tools.plot_train_result import plot_train_process
from utiles import rename_train_log

iteration = 500
steps_per_episode = 500
batch_episode = 50
save_interval = 100

learn_rate = 1e-3
gamma = 0.99

do_train = True
do_test = True

if __name__ == '__main__':
    env, agent = initialize_dqn(learn_rate, gamma)
    if do_train:
        plot_data = train(env, agent, iteration, steps_per_episode, batch_episode, save_interval)
        plot_train_process(plot_data)
    if do_test:
        valid(env, agent, iteration)


atexit.register(rename_train_log)
from DQN_train import train
from DQN_test import valid
from DQN_agent import initialize_dqn


iteration = 500
steps_per_episode = 500
batch_episode = 50
save_interval = 100

learn_rate = 1e-3
gamma = 0.99

do_train = False
do_test = True
ckpt_path = f"DQN_model.pth"

if __name__ == '__main__':
    env, agent = initialize_dqn(learn_rate, gamma)
    if do_train:
        train(env, agent, iteration, steps_per_episode, batch_episode, save_interval, ckpt_path)
    if do_test:
        valid(env, agent, ckpt_path)

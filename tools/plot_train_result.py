import os
import re

import matplotlib.pyplot as plt
import datetime
from pathlib import Path

project_path = Path(__file__).parent.parent
log_path = project_path / "log"

def plot_train_process(plot_data):
    iterations = []
    rewards = []
    for iteration, reward in plot_data.items():
        iterations.append(iteration)
        rewards.append(reward)

    fig = plt.figure()
    plt.plot(iterations, rewards, 'b-', label='reward')
    plt.xlabel('iteration')
    plt.ylabel('reward')
    plt.title('training process')
    plt.legend()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"train_reward_{timestamp}.png")
    plt.close(fig)


def plot_compare(log_folder):
    log_list = os.listdir(log_folder)
    print(log_list)
    fig = plt.figure()
    for log in log_list:
        iterations = []
        rewards = []
        with open(log_folder / log, "r") as f:
            for line in f.readlines():
                print(line)
                match = re.search(r"iteration:\s*(\d+).*?mean reward:\s*([\d.]+)", line)
                if match:
                    iterations.append(int(match.group(1)))
                    rewards.append(float(match.group(2)))
        plt.plot(iterations, rewards, '-', label=log)
    plt.xlabel('iteration')
    plt.ylabel('reward')
    plt.title('training compare')
    plt.legend()
    plt.savefig(project_path / "train_compare.png")
    plt.close(fig)


if __name__ == '__main__':
    plot_compare(log_path)
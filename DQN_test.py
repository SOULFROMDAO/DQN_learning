import time


def valid(env, agent, version):
    agent.load_checkpoint(version)
    state = env.reset()
    while True:
        action = agent.online_net.act(state)
        state, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.01)
        if done:
            env.reset()
            break

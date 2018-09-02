## inspired by Siraj Raval：https://www.youtube.com/watch?v=05NqKJ0v7EE

from dqn_agent import Agent
from model import QNetwork
import matplotlib.pyplot as plt
import numpy as np
import torch

STATE_SIZE = 10
EPISODE_COUNT = 1000


def dqn(n_episodes=EPISODE_COUNT, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []
    for i_episode in range(1, n_episodes+1):
        print("Episode" + str(i_episode))
        state = getState(stockData, 0, STATE_SIZE + 1)
        total_profit = 0
        agent.inventory = []
        eps = eps_start

        for t in range(l):
            action = agent.act(state, eps)
            next_state = getState(stockData, t + 1, STATE_SIZE + 1)
            reward = 0

            if action == 1:# 买入
                agent.inventory.append(stockData[t])
                # print("buy" + str(stockData[t]))
            elif action == 2 and len(agent.inventory) > 0: # 卖出
                bought_price = agent.inventory.pop(0)
                total_profit += stockData[t] - bought_price
                # reward = max(stockData[t] - bought_price, 0)
                reward = stockData[t] - bought_price
                # print("Sell: " + str(stockData[t]) + " | Profit: " + str(stockData[t] - bought_price))
            done = 1 if t == l - 1 else 0
            agent.step(state, action, reward, next_state, done)
            eps = max(eps_end, eps * eps_decay)
            state = next_state

            if done:
                print("------------------------------")
                print("total_profit = " + str(total_profit))
                print("------------------------------")
        scores.append(total_profit)
    return scores


def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]
    res = []
    for i in range(n - 1):
        res.append(block[i + 1] - block[i])
    return np.array([res])


if __name__ == '__main__':
    stockData = []
    file = open("600967.txt").read().splitlines()
    # 打开数据文件
    for item in file[1:]:
        stockData.append(float(item.split("\t")[4]))

    agent = Agent(state_size=STATE_SIZE, action_size=3)
    l = len(stockData) - 1

    scores = dqn()
    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


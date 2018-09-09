
import pandas as pd
import  matplotlib
import matplotlib.pyplot as plt
from dqn_agent import Agent
import torch
import numpy as np
from main import getState, money_calculate, cost_calculate


STATE_SIZE = 10

# df = pd.read_csv('600967_eval.csv', encoding='gbk')
# #print(df['收盘价'])
# df['日期'] = pd.to_datetime(df['日期'])
# df.set_index("日期", inplace=True)
# df.sort_index(axis=0)
# df = df[(df['收盘价'] != 0)]
df = pd.read_csv('600967.txt', encoding='gbk', sep='\t')
print(df['收盘'])
df['date'] = pd.to_datetime(df['时间'])
df.set_index('date', inplace=True)

agent = Agent(state_size=STATE_SIZE, action_size=3)
agent.qnetwork_local.load_state_dict(torch.load('checkpoint1.pth', map_location=lambda storage, loc: storage))


stockData = list(df['收盘'])
l = len(stockData)-1
window_size = 10
state = getState(stockData, 0, window_size + 1)
# total_profit = 0
agent.inventory = []
action_list = []
pos_list = []
pos_old = 0
total_share = 0
cost = 0
money_initial = 10000
money = money_initial
for t in range(l):
    action = agent.act(state, eps=0, is_eval=True)
    next_state = getState(stockData, t + 1, STATE_SIZE + 1)
    if action == 1:# 买入
        pos_new = min(pos_old + 0.2, 1)
        total_share += money * (pos_new - pos_old) / stockData[t]
        #agent.inventory.append(stockData[t])
                # print("buy" + str(stockData[t]))
    elif action == 2 :
        pos_new = max(pos_old - 0.2, 0)
        total_share += money * (pos_new - pos_old) / stockData[t]
        #bought_price = agent.inventory.pop(0)
        #total_profit += stockData[t] - bought_price
    else:
        pos_new = pos_old
    money = money_calculate(money, total_share, stockData[t], pos_new)

    done = 1 if t == l - 1 else 0
    state = next_state
    pos_old = pos_new
    action_list.append(action)
    pos_list.append(pos_new)
    total_profit = (money - money_initial) / money_initial
    # value_list.append(stockData[t])
    if done:
        print("------------------------------")
        print("total_profit = " + str(total_profit))
        print("------------------------------")
        #plt.plot(np.arange(len(value_list)), value_list)
        action_list.append(0)
        pos_list.append(pos_new)
        df['action'] = pd.DataFrame(action_list).values
        df['pos'] = pd.DataFrame(pos_list).values

    #plt(x=df.index[i], y=df['action'][i], c=color)
        fig, ax1 = plt.subplots()

        df["收盘"].plot(figsize=(8, 5), grid=True, label='1st')
    #plt.plot(x=df.index.values, y=df["收盘"])
    #print(df)
        sell = (df['action'].values == 2)
        plt.scatter(df.index[sell], df["收盘"].values[sell], c='r')
        buy = (df['action'].values == 1)
        plt.scatter(df.index[buy], df["收盘"].values[buy], c='g')
        plt.legend(['value', 'sell', 'buy'])
        ax2 = ax1.twinx()
        plt.bar(df.index, df['pos'].values, label='2st')
        plt.ylim([0, 4])
        ax1.set_ylabel("Rate of Return")
        ax2.set_ylabel("Position")
        plt.show()



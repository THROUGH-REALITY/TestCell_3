import time

import matplotlib.pyplot as plt
import numpy as np

from grid_world import GridWorld
from qlearning_agent import Summon

# 定数
NB_EPISODE = 3   # エピソード数
X_MAX = 15
Y_MAX = 18
POPULATION = 10
start = time.time()

if __name__ == '__main__':
    grid_env = GridWorld(   # grid worldの環境の初期化
        x_max=X_MAX,
        y_max=Y_MAX)

    summon = Summon(    # エージェントの召喚
        zero_list=grid_env.zero_list,
        population=POPULATION)
    print(f"StaRt time = {time.time()-start}")
    #times = np.zeros((POPULATION,NB_EPISODE))
    episode_reward = np.zeros((POPULATION,NB_EPISODE))  # 1エピソードの累積報酬
    for episode in range(NB_EPISODE):   # 実験
        is_end_episode = np.zeros(POPULATION)
        for i in summon.agents:
            start_x = i.observation[0]
            start_y = i.observation[1]
            grid_env.map[start_x][start_y] = 3
        while(np.all(is_end_episode) == False):    # 全員がゴールするまで続ける
            for id,agent in enumerate(summon.agents):
                if is_end_episode[id] == False:
                    start_x = agent.observation[0]
                    start_y = agent.observation[1]
                    grid_env.map[start_x,start_y] = 0
                    action = agent.act()  # 行動選択
                    state, reward, is_end_episode[id] = grid_env.step(start_x, start_y, action)
                    grid_env.map[state[0]][state[1]] = 3
                    agent.observe(state, reward)   # 状態と報酬の観測 
                    episode_reward[id][episode] += 1
                else:grid_env.map[0,0] = 1
        for id in summon.agents:
            id.observation = grid_env.reset(id.init_pos)  # 初期化
            #print(summon.agents[id].observation)
        #agent.observe(state)    # エージェントを初期位置に
        print(f"EP.{episode +1} End time = {time.time()-start}") #(t = {len(episode_reward)})"# 所要時間の計算
    plt.imshow(episode_reward)
    plt.savefig(f"result.png")
    plt.show()
import time

import numpy as np
import matplotlib.pyplot as plt
from grid_world import GridWorld
from qlearning_agent import Summon

# 定数
NB_EPISODE = 1   # エピソード数
X_MAX = 15
Y_MAX = 18
POPULATION = 2
start = time.time()

if __name__ == '__main__':
    grid_env = GridWorld(   # grid worldの環境の初期化
        x_max=X_MAX,
        y_max=Y_MAX,)

    summon = Summon(    # エージェントの召喚
        zero_list=grid_env.zero_list,
        population=POPULATION)
    
    times = []
    is_end_episode = []
    for episode in range(NB_EPISODE):   # 実験
        #episode_reward = []  # 1エピソードの累積報酬
        for i in range(POPULATION):
            start_x = summon.agents[i].observation[0]
            start_y = summon.agents[i].observation[1]
            grid_env.map[start_x][start_y] = 3
            is_end_episode.append(False)
            print(start_x,start_y)
        plt.imshow(grid_env.map)
        plt.show()
        print(is_end_episode)
        while(False in is_end_episode):    # 全員がゴールするまで続ける
            for id in range(POPULATION):
                if is_end_episode[id] == False:
                    start_x = summon.agents[id].observation[0]
                    start_y = summon.agents[id].observation[1]
                    grid_env.map[start_x][start_y] = 0
                    print(start_x,start_y)
                    action = summon.agents[id].act()  # 行動選択
                    print(action)
                    state, reward, is_end_episode[id] = grid_env.step(start_x, start_y, action)
                else: grid_env.map[0,0] = 1
                if is_end_episode[id] == False:
                    grid_env.map[state[0]][state[1]] = 3
                    print(state,reward)
                    summon.agents[id].observe(state, reward)   # 状態と報酬の観測
                plt.imshow(grid_env.map)
                plt.show()
        print(is_end_episode)    
            #episode_reward.append(reward)
            #rewards.append(np.sum(episode_reward))  # このエピソードの平均報酬を与える
        #times.append(len(episode_reward)) #かかった時間をリストに追加
        state = grid_env.reset()  # 初期化
        #agent.observe(state)    # エージェントを初期位置に
        print(f"EP.{episode +1} End") #(t = {len(episode_reward)})"

    # 所要時間の計算
    print(f"time = {time.time()-start}")

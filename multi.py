import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from grid_world import GridWorld
from qlearning_agent import Summon
from main import Plotting

# 定数
NB_EPISODE = 31   # エピソード数
X_MAX = 12
Y_MAX = 12
POPULATION = 10
start = time.time()

if __name__ == '__main__':
    grid_env = GridWorld(   # grid worldの環境の初期化
        x_max=X_MAX,
        y_max=Y_MAX)

    decided_zero = [grid_env.zero_list[10],
                    grid_env.zero_list[16],
                    grid_env.zero_list[23],
                    grid_env.zero_list[39],
                    grid_env.zero_list[44],
                    grid_env.zero_list[51],
                    grid_env.zero_list[53],
                    grid_env.zero_list[68],
                    grid_env.zero_list[79],
                    grid_env.zero_list[91]]
    
    summon = Summon(             # エージェントの召喚
        zero_list=decided_zero,
        population=POPULATION)
    print(f"StaRt time = {time.time()-start}")    #times = []

    episode_reward = np.zeros((POPULATION,NB_EPISODE))  # 1エピソードの累積報酬
    for episode in range(NB_EPISODE):   # 実験
        is_end_episode = np.zeros(POPULATION)
        for i in summon.agents:
            start_x = i.observation[0]
            start_y = i.observation[1]
            grid_env.map[start_x][start_y] = 3
        init_map = np.array(grid_env.map)
        while(np.all(is_end_episode) == False):    # 全員がゴールするまで続ける
            for id,agent in enumerate(summon.agents):
                if is_end_episode[id] == False:
                    start_x = agent.observation[0]
                    start_y = agent.observation[1]
                    grid_env.map[start_x][start_y] = 0
                    action = agent.act()  # 行動選択
                    state, reward, is_end_episode[id] = grid_env.step(start_x, start_y, action)
                    grid_env.map[state[0]][state[1]] = 3
                    agent.observe(state, reward)   # 状態と報酬の観測 
                    episode_reward[id][episode] += 1
                else:
                    grid_env.map[0][0] = 1
                    #grid_env.map[X_MAX-1][0] = 1
        for id in summon.agents:
            id.observation = grid_env.reset(id.init_pos)  # 初期化
            #print(summon.agents[id].observation)
        #agent.observe(state)    # エージェントを初期位置に
        print(f"EP.{episode +1} End time = {time.time()-start}") #(t = {len(episode_reward)})"# 所要時間の計算
    fig = plt.figure(figsize=(18,10), tight_layout=True)  # 図を描く大きさと、図の変数名を宣言
    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(
        gs[0, 0],
        xlabel="X",
        ylabel="Y",
        title="init_state")
    ax1.text(0, 0, 'G', size=15, ha='center', va='center', color='white')
    ax1.imshow(init_map)
    ax2 = fig.add_subplot(
        gs[0, 1:],
        xlabel="episode",
        ylabel="times",
        title="Result",
        xlim=(0,NB_EPISODE-1))
    lines = [ax2.plot(np.arange(NB_EPISODE),episode_reward[id]) for id in range(POPULATION)]
    y_min, max_y = ax2.get_ylim()
    ax2.set_ylim(0, max_y)
    
    #ax2.legend(handles=lines[::-1], labels=[e for e in range(POPULATION)])
    fig.legend()
    ax1.xaxis.set_major_locator(ticker.MultipleLocator())
    ax1.yaxis.set_major_locator(ticker.MultipleLocator())
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(100)) 
    #Plotting.text(episode_reward,ax2,"deeppink")
    for index,value in enumerate(summon.agents):
        ax1.text(value.init_pos[1], value.init_pos[0], index, size=15, color='deeppink', ha='center', va='center')
    plt.savefig(f"result.png")
    plt.show()
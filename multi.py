import time
import numpy as numpy

from grid_world import GridWorld
from main import Summon

X_MAX = 15
Y_MAX = 18
START_X = X_MAX - 1     # 端からスタートさせる
START_Y = Y_MAX - 1
POPULATION = 2
start = time.time()

grid_env = GridWorld(   # grid worldの環境の初期化
    x_max=X_MAX,
    y_max=Y_MAX,
    start_x=START_X,
    start_y=START_Y)

summon = Summon(population= POPULATION)

print(summon.agents)

# 所要時間の計算
print(f"time = {time.time()-start}")
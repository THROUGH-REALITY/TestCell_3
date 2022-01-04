import copy
import numpy as np


class GridWorld:

    def __init__(self, x_max, y_max):

        self.x_max = x_max
        self.y_max = y_max
        self.filed_type = {
            "N": 0,  # 通常
            "G": 1,  # ゴール
            "W": 2,  # 壁
            "H": 3,  # 他の人間
        }
        self.actions = {
            "UP": 0,
            "DOWN": 1,
            "LEFT": 2,
            "RIGHT": 3
        }
        self.map_arr = np.zeros((y_max,x_max))
        self.map_arr[::2] = 2 
        self.map_arr[:, ::2] = 0
        self.map_arr[0,0] = 1
        #self.map_arr[x_max-1,0] = 1
        self.map = self.map_arr.tolist()

        self.zero_list = list(zip(*np.where( self.map_arr < 1)))
        #self.start_pos = start_x,start_y   # エージェントのスタート地点(x, y)
        #self.agent_pos = copy.deepcopy(self.start_pos)  # エージェントがいる地点

    def step(self, start_x, start_y):
        """
            行動の実行
            状態、ゴールしたかを返却
        """
        to_x, to_y = start_x, start_y
        # 移動可能かどうかの確認。(左→無理なら上)
        left_possibility = self._is_possible_action(to_x, to_y, action=self.actions["UP"])
        if left_possibility != 0:
            to_x += -1
        elif self._is_possible_action(to_x, to_y, action=self.actions["LEFT"]):
            to_y += -1
        #up_possibility = self._is_possible_action(to_x, to_y, action=self.actions["LEFT"])
        is_goal = self._is_end_episode(to_x, to_y)  # エピソードの終了の確認
        self.agent_pos = to_x, to_y
        return self.agent_pos, is_goal

    def _is_end_episode(self, x, y):
        """
            x, yがエピソードの終了かの確認。
        """
        if self.map[x][y] == self.filed_type["G"]: # ゴール
            #print("Agent GOAL")
            return True
        else:
            return False

    def _is_wall(self, x, y):
        """
            x, yが壁または人間かどうかの確認
        """
        if self.map[x][y] == self.filed_type["W"]:
            return True
        else:
            return False
    
    def _is_other_agent(self, x, y):
        if self.map[x][y] == self.filed_type["H"]:
            return True
        else:
            return False

    def _is_possible_action(self, x, y, action):
        """
            実行可能な行動かどうかの判定
        """
        to_x = x
        to_y = y

        if action == self.actions["UP"]:
            #print("下(左)に行った")
            to_y += -1
            #print(to_y,to_x)
        elif action == self.actions["DOWN"]:
            #print("上(右)に行った")
            to_y += 1
            #print(to_y,to_x)
        elif action == self.actions["LEFT"]:
            #print("左(上)に行った")
            to_x += -1
            #print(to_y,to_x)
        elif action == self.actions["RIGHT"]:
            #print("右(下)に行った")
            to_x += 1
            #print(to_y,to_x)

        if len(self.map[1]) <= to_y or 0 > to_y:
            #print("y行き過ぎ")
            return 1
        elif len(self.map[0]) <= to_x or 0 > to_x:
            #print("x行き過ぎ")
            return 1
        elif self._is_wall(to_x, to_y):
            #print("壁だった")
            #print(to_y,to_x)
            return 1
        elif self._is_other_agent(to_x, to_y):
            #print("人だった")
            #print(to_y,to_x)
            return 2

        return 0

    def _compute_reward(self, x, y):
        if self.map[x][y] == self.filed_type["N"]:
            return 0
        elif self.map[x][y] == self.filed_type["G"]:
            return 100
        #elif self.map[y,x] == self.filed_type["T"]:
            return -100

    def reset(self,init_pos):
        self.map = self.map_arr.tolist()
        self.agent_pos = init_pos
        return self.agent_pos

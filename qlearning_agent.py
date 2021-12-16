import copy
import random

import numpy as np

EPSILON = .1    # 探索率
ALPHA = .1      # 学習率
GAMMA = .90     # 割引率
ACTIONS = np.arange(4)  # 行動の集合


class Summon:

    def __init__(self, zero_list, population=2):
        self.agents = self.__generate_agents(zero_list, population)

    def __generate_agents(self, zero_list, population):
        agents = []
        for id in range(population):
            ini_state = random.choice(zero_list) # 初期状態（エージェントのスタート地点の位置）
            agents.append(
                QLearningAgent(
                    alpha=ALPHA,
                    gamma=GAMMA,
                    epsilon=EPSILON,
                    actions=ACTIONS,
                    observation=ini_state))
        return agents



class QLearningAgent:
    """
        Q学習 エージェント
    """

    def __init__(
            self,
            alpha=.2,
            epsilon=.1,
            gamma=.99,
            actions=None,
            observation=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.reward_history = []
        self.actions = actions
        self.init_pos = observation
        self.observation = observation
        self.state = str(observation)
        self.ini_state = str(observation)
        self.previous_state = None
        self.previous_action = None
        self.q_values = self._init_q_values()

    def _init_q_values(self):
        """
           Q テーブルの初期化
        """
        q_values = {}
        q_values[self.state] = np.repeat(0.0, len(self.actions))
        return q_values

    def init_state(self):
        """
            状態の初期化
        """
        self.previous_state = copy.deepcopy(self.ini_state)
        self.state = copy.deepcopy(self.ini_state)
        return self.state

    def act(self):
        # ε-greedy選択
        if np.random.uniform() < self.epsilon:  # random 行動
            action = np.random.randint(0, len(self.q_values[self.state]))
        else:   # greedy 行動
            action = np.argmax(self.q_values[self.state])

        self.previous_action = action
        return action

    def observe(self, next_state, reward=None):
        """
            次の状態と報酬の観測
        """
        self.observation = next_state
        next_state = str(next_state)
        if next_state not in self.q_values:  # 始めて訪れる状態であれば
            self.q_values[next_state] = np.repeat(0.0, len(self.actions))

        self.previous_state = copy.deepcopy(self.state)
        self.state = next_state
        
        if reward is not None:
            self.reward_history.append(reward)
            self.learn(reward)

    def learn(self, reward):
        """
            Q値の更新
        """
        q = self.q_values[self.previous_state][self.previous_action]  # Q(s, a)
        max_q = max(self.q_values[self.state])  # max Q(s')
        # Q(s, a) = Q(s, a) + alpha*(r+gamma*maxQ(s')-Q(s, a))
        self.q_values[self.previous_state][self.previous_action] = q + \
            (self.alpha * (reward + (self.gamma * max_q) - q))

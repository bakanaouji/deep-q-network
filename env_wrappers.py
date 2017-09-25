import gym
import numpy as np
from collections import deque

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, no_op_max):
        super(NoopResetEnv, self).__init__(env)
        self.no_op_max = no_op_max

    def _reset(self):
        self.env.reset()
        # ランダムなフレーム数分「何もしない」
        T = np.random.randint(1, self.no_op_max + 1)
        observation = None
        for _ in range(T):
            # 「何もしない」で，次の画面を返す
            # @todo pongの場合，0：何もしない，1：何もしない，2：上，3：下なので修正が必要と思われる
            observation, _, _, _ = self.env.step(0)
        return observation

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, action_repeat):
        super(MaxAndSkipEnv, self).__init__(env)
        self.observation_buffer = deque(maxlen=2)
        self.action_repeat = action_repeat

    def _step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self.action_repeat):
            observation, reward, done, info = self.env.step(action)
            self.observation_buffer.append(observation)
            total_reward += reward
            if done:
                break

        # 前のフレームの観測との最大値を状態として返す
        max_frame = np.max(np.stack(self.observation_buffer), axis=0)
        return max_frame, total_reward, done, info

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super(FireResetEnv, self).__init__(env)

    def _reset(self):
        self.env.reset()
        observation, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        observation, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return observation

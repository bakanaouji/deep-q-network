import gym
import numpy as np

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
            self.env.render()
        return observation

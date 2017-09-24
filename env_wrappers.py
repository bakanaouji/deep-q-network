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
            # @todo 0番目が「何もしない行動かどうかをチェック」
            observation, _, _, _ = self.env.step(0)
        return observation

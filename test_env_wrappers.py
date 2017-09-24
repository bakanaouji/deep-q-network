import unittest
import gym
from env_wrappers import NoopResetEnv, MaxAndSkipEnv

class TestEnvWrappers(unittest.TestCase):

    def test_noop_reset_env(self):
        """
        no_op_maxフレーム分だけ何もしてなければおｋ
        """
        env = gym.make("PongNoFrameskip-v4")
        env = NoopResetEnv(env, no_op_max=30)
        env.reset()

    def test_max_and_skip_env(self):
        """
        action_repeatフレーム分だけ行動を繰り返している，
        かつ，
        状態が前フレームの観測値との最大値であればおｋ
        """
        env = gym.make("PongNoFrameskip-v4")
        env = MaxAndSkipEnv(env, action_repeat=4)
        env.reset()
        env.step(0)

if __name__ == '__main__':
    unittest.main()

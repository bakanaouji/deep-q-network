import unittest
import gym
from env_wrappers import NoopResetEnv

class TestEnvWrappers(unittest.TestCase):

    def test_noop_reset_env(self):
        """
        no_op_maxフレーム分だけ何もしてなければおｋ
        """
        env = gym.make("PongNoFrameskip-v4")
        env = NoopResetEnv(env, no_op_max=300)
        env.reset()

if __name__ == '__main__':
    unittest.main()

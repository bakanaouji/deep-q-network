import unittest
import gym
import numpy as np
from env_wrappers import NoopResetEnv, MaxAndSkipEnv, FireResetEnv, ProcessFrame84, FrameStack, ClippedRewardsWrapper, ScaledFloatFrame, EpisodicLifeEnv, wrap_dqn
import cv2

class TestEnvWrappers(unittest.TestCase):

    def test_episodic_life_env(self):
        """
        Breakoutなどのライフがあるゲームで，
        1ライフ失う毎にエピソードが終わったことになっていればok
        """
        env = gym.make("BreakoutNoFrameskip-v4")
        env = EpisodicLifeEnv(env)
        env.reset()
        while True:
            _, _, done, _ = env.step(1)
            if done:
                env.reset()
                break
            env.render()

    def test_noop_reset_env(self):
        """
        no_op_maxフレーム分だけ何もしてなければおｋ
        """
        env = gym.make("BreakoutNoFrameskip-v4")
        env = NoopResetEnv(env, no_op_max=30)
        env.reset()

    def test_max_and_skip_env(self):
        """
        action_repeatフレーム分だけ行動を繰り返している，
        かつ，
        状態が前フレームの観測値との最大値であればおｋ
        """
        env = gym.make("BreakoutNoFrameskip-v4")
        env = MaxAndSkipEnv(env, action_repeat=4)
        env.reset()
        env.step(0)

    def test_fire_reset_env(self):
        """
        ゲームが開始されていれば観測値が前フレームと変わっているはずなので，
        観測値が前フレームと比べて変化していればok
        """
        env = gym.make("Breakout-v0")
        env = FireResetEnv(env)
        pre_observation = env.reset()
        observation, _, _, _ = env.step(0)
        self.assertFalse((observation == pre_observation).all())

    def test_process_frame84(self):
        """
        cv2で画像を保存してみて，グレースケールならok
        """
        env = gym.make("BreakoutNoFrameskip-v4")
        env = ProcessFrame84(env)
        observation = env.reset()
        cv2.imwrite("test_process_frame84.png", observation[:,:,0] * 255)

    def test_frame_stack(self):
        """
        スタックするフレーム数分だけ次元数が増加していればok
        """
        k = 4
        env = gym.make("BreakoutNoFrameskip-v4")
        expected_observation_shape = (k * env.observation_space.shape[2], env.observation_space.shape[0], env.observation_space.shape[1])
        env = FrameStack(env, k)
        observation = env.reset()
        self.assertEqual(expected_observation_shape, np.array(observation).shape)
        self.assertEqual(env.observation_space.shape, np.array(observation).shape)
        observation, _, _, _ = env.step(0)
        self.assertEqual(env.observation_space.shape, np.array(observation).shape)

    def test_clipped_rewards_wrapper(self):
        """
        報酬が-1~1の間に入っていればok
        """
        env = gym.make("BreakoutNoFrameskip-v4")
        env = ClippedRewardsWrapper(env)
        env.reset()
        _, reward, _, _ = env.step(0)
        self.assertLessEqual(reward, 1.0)
        self.assertGreaterEqual(reward, -1.0)

    def test_scaled_float_frame(self):
        """
        状態の各値が0~1に入っていればok
        """
        env = gym.make("BreakoutNoFrameskip-v4")
        env = ScaledFloatFrame(env)
        observation = env.reset()
        self.assertTrue((observation <= 1.0).all())
        self.assertTrue((observation >= 0.0).all())

    def test_wrap_dqn(self):
        env = gym.make("BreakoutNoFrameskip-v4")
        env = wrap_dqn(env)
        # 1 : EpisodicLifeEnvテスト
        # Breakoutなどのライフがあるゲームで，
        # 1ライフ失う毎にエピソードが終わったことになっていればok
        env.reset()
        while True:
            _, _, done, _ = env.step(0)
            if done:
                break
            env.render()
        # 2 : NoopResetEnvテスト
        # no_op_maxフレーム分だけ何もしてなければおｋ
        env.reset()
        while True:
            _, _, done, _ = env.step(2)
            if done:
                break
            env.render()
        # 3 : MaxAndSkipEnvテスト
        # action_repeatフレーム分だけ行動を繰り返している，
        # かつ，
        # 状態が前フレームの観測値との最大値であればおｋ
        env.reset()
        while True:
            _, _, done, _ = env.step(env.action_space.sample())
            if done:
                break
            env.render()
        # 4 : FireResetEnvテスト
        # ゲームが開始されていれば観測値が前フレームと変わっているはずなので，
        # 観測値が前フレームと比べて変化していればok
        pre_observation = env.reset()
        observation, _, _, _ = env.step(0)
        self.assertFalse((observation == pre_observation).all())
        # 5 : ProcessFrame84テスト
        # cv2で画像を保存してみて，グレースケールならok
        observation = env.reset()
        cv2.imwrite("test_wrap_dqn.png", observation[0] * 255)
        # 6 : FrameStackテスト
        # スタックするフレーム数分だけ次元数が増加していればok
        observation = env.reset()
        self.assertEqual((4, 84, 84), np.array(observation).shape)
        observation, _, _, _ = env.step(0)
        self.assertEqual((4, 84, 84), np.array(observation).shape)
        # 7 : ClippedRewardsWrapperテスト
        # 報酬が-1~1の間に入っていればok
        env.reset()
        _, reward, _, _ = env.step(0)
        self.assertLessEqual(reward, 1.0)
        self.assertGreaterEqual(reward, -1.0)
        # 8 : ScaledFloatFrameテスト
        # 状態の各値が0~1に入っていればok
        observation = env.reset()
        self.assertTrue((observation <= 1.0).all())
        self.assertTrue((observation >= 0.0).all())

if __name__ == '__main__':
    unittest.main()

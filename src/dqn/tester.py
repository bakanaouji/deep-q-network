import tensorflow as tf

from models.models import CNN
from util.log.logger import restore_sess
from agents.agents import *


class Tester(object):
    def __init__(self, env, args):
        """
        DQNで学習したモデルをテストするクラス．

        Parameters
        ----------
        env: gym.wrappers.time_limit.TimeLimit
            open ai gymの環境
        args:
            テストに必要なパラメータのリスト
        """
        self.env = env
        self.width = args.width
        self.height = args.height

        self.tmax = args.tmax
        self.history_len = args.history_len

        self.save_network_path = "../data/" + args.save_network_path + "/" + args.env_name + \
                                 args.save_option_name + "/model.ckpt"

    def test(self):
        """
        mainメソッド．
        学習後のモデルをテストする．
        """
        # Q-Network
        q_func = CNN(self.env.action_space.n, self.history_len, self.width, self.height)
        # Sessionの構築
        sess = tf.InteractiveSession()
        # session読み込み
        restore_sess(sess, self.save_network_path)
        # エージェント初期化
        agent = DQNTestAgent(num_actions=self.env.action_space.n,
                             q_func=q_func)

        # メインループ
        t = 0
        episode = 0
        while t < self.tmax:
            # エピソード実行
            episode += 1
            duration = 0
            total_reward = 0.0
            done = False
            # 環境初期化
            obs = self.env.reset()
            # エピソード終了まで実行
            while not done:
                # 行動を選択
                action = agent.action(t, obs)
                # 行動を実行し，報酬と次の画面とdoneを観測
                obs, reward, done, info = self.env.step(action)
                self.env.render()
                total_reward += reward
                t += 1
                duration += 1
            print('EPISODE: {0:6d} / TIME_STEP: {1:8d} / DURATION: {2:5d} / TOTAL_REWARD: {3:3.0f}'.format(
                episode, t, duration, total_reward))

import numpy as np
import tensorflow as tf

from replay_memory import ReplayMemory
from model import init_q_network

class Trainer(object):
    def __init__(self, env, **params):
        self.env = env
        self.learning_rate = params['learning_rate']
        self.gradient_momentum = params['gradient_momentum']
        self.squared_gradient_momuntum = params['squared_gradient_momuntum']
        self.min_squared_gradient = params['min_squared_gradient']
        self.initial_exploration = params['initial_exploration']
        self.final_exploration = params['final_exploration']
        self.final_exploration_frame = params['final_exploration_frame']
        self.replay_memory_size = params['replay_memory_size']
        self.frame_width = params['frame_width']
        self.frame_height = params['frame_height']
        self.agent_history_length = params['agent_history_length']
        self.tmax = params['tmax']
        
    def build_training_op(self, num_actions, q_values, q_network_weights):
        a = tf.placeholder(tf.int64, [None])  # 行動
        y = tf.placeholder(tf.float32, [None])  # 教師信号

        a_one_hot = tf.one_hot(a, num_actions, 1.0,
                0.0)  # 行動をone hot vectorに変換する
        q_value = tf.reduce_sum(tf.multiply(q_values, a_one_hot),
                reduction_indices=1)  # 行動のQ値の計算
        # エラークリップ
        error = y - q_value
        quadratic_part = tf.clip_by_value(error, -1.0, 1.0)
        loss = tf.reduce_mean(tf.square(quadratic_part))  # 誤差関数

        optimizer = tf.train.RMSPropOptimizer(self.learning_rate,
                momentum=self.gradient_momentum,
                decay=self.squared_gradient_momuntum,
                epsilon=self.min_squared_gradient)  # 最適化手法を定義
        grad_update = optimizer.minimize(loss,
                var_list=q_network_weights)  # 誤差最小化

        return a, y, loss, grad_update

    def choose_action_by_epsilon_greedy(self, action_num, s, q_values, epsilon, observation):
        if epsilon >= np.random.rand():
            action = np.random.randint(action_num)
        else:
            action = np.argmax(q_values.eval(
                feed_dict={s: [observation]}))
        return action

    def train(self):
        # Replay Memory
        replay_memory = ReplayMemory(self.replay_memory_size)

        # Q-Network
        s, q_values, q_network = init_q_network(self.env.action_space.n, self.agent_history_length, self.frame_width, self.frame_height)
        q_network_weights = q_network.trainable_weights # 学習される重み

        # TargetNetwork
        target_s, target_q_values, target_network = init_q_network(self.env.action_space.n, self.agent_history_length, self.frame_width, self.frame_height)
        target_network_weights = target_network.trainable_weights  # 重みのリスト

        # 定期的にTargetNetworkをQ-Networkで同期する処理
        assign_target_network = [
                target_network_weights[i].assign(q_network_weights[i]) for
                i in range(len(target_network_weights))]

        # 誤差関数や最適化のための処理
        a, y, loss, grad_update = self.build_training_op(self.env.action_space.n, q_values, q_network_weights)

        # Sessionの構築
        sess = tf.InteractiveSession()

        # 変数の初期化（Q Networkの初期化）
        sess.run(tf.global_variables_initializer())

        # Target Networkの初期化
        sess.run(assign_target_network)

        t = 0
        episode = 0
        epsilon = self.initial_exploration

        # メインループ
        while t < self.tmax:
            # エピソード実行
            episode += 1
            duration = 0
            total_reward = 0.0
            total_q_max = 0.0
            total_loss = 0
            done = False
            # 環境初期化
            observation = self.env.reset()
            # エピソード終了まで実行
            while not done:
                # ε-greedyに従って行動を選択
                action = self.choose_action_by_epsilon_greedy(self.env.action_space.n, s, q_values, epsilon, observation)
                # εを線形減少
                epsilon = max(self.final_exploration,
                                    epsilon - (
                                    self.initial_exploration - self.final_exploration) / self.final_exploration_frame)

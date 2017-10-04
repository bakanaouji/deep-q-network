import numpy as np
import tensorflow as tf

from replay_memory import ReplayMemory
from model import CNN
from util.linear_schedule import LinearSchedule
from logger import save_sess, restore_sess, Logger

class Trainer(object):
    def __init__(self, env, **params):
        """
        学習を行うTrainerを構築．

        Parameters
        ----------
        env: gym.env
            open ai gymの環境
        **params: {}
            学習に必要なパラメータのリスト
        """
        self.env = env
        self.frame_width = params['frame_width']
        self.frame_height = params['frame_height']
        self.agent_history_length = params['agent_history_length']
        self.replay_start_size = params['replay_start_size']

        self.learning_rate = params['learning_rate']
        self.tmax = params['tmax']
        self.replay_memory_size = params['replay_memory_size']
        self.exploration_fraction = params['exploration_fraction']
        self.final_exploration = params['final_exploration']
        self.learn_frequency = params['learn_frequency']
        self.discount_factor = params['discount_factor']
        self.minibatch_size = params['minibatch_size']
        self.target_network_update_frequency = params['target_network_update_frequency']

        self.render = params['render']
        self.save_network_frequency = params['save_network_frequency']
        self.save_network_path = params['save_network_path']
        self.save_summary_path = params['save_summary_path']

    def build_training_op(self, num_actions, q_func):
        """
        学習に必要な処理の構築．

        Parameters
        ----------
        num_actions: int
            環境の行動数
        q_func: model.CNN
            Q関数

        Returns
        ----------
        a: tf.placeholder(tf.int64, [None])
            エージェントが選択する行動値
        y: tf.placeholder(tf.float32, [None])
            教師信号．Q^*関数のQ値．
        loss:
            誤差関数．yとの誤差．
        grad_update:
            誤差を最小化する処理
        """
        # 行動
        a = tf.placeholder(tf.int64, [None])
        # 教師信号
        y = tf.placeholder(tf.float32, [None])
        # 行動をone hot vectorに変換する
        a_one_hot = tf.one_hot(a, num_actions, 1.0, 0.0)
        # 行動のQ値の計算
        q_value = tf.reduce_sum(tf.multiply(q_func.q_values, a_one_hot), reduction_indices=1)
        # エラークリップ
        error = y - q_value
        loss = tf.where(tf.abs(error) < 1.0, tf.square(error) * 0.5, tf.abs(error) - 0.5)
        # 誤差関数
        loss = tf.reduce_mean(loss)
        # 最適化手法を定義
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # 誤差最小化の処理
        grad_update = optimizer.minimize(loss, var_list=q_func.model.trainable_weights)

        return a, y, loss, grad_update

    def choose_action_by_epsilon_greedy(self, num_actions, q_func, epsilon, observation):
        """
        ε-greedyに従って行動を選択．

        Parameters
        ----------
        num_actions: int
            環境の行動数
        q_func: model.CNN
            Q関数
        epsilon: float
            ε．ランダムに行動を決定する確率．

        Returns
        ----------
        action: int
            選択した行動
        """
        if epsilon >= np.random.rand():
            action = np.random.randint(num_actions)
        else:
            action = np.argmax(q_func.q_values.eval(
                feed_dict={q_func.s: [observation]}))
        return action

    def train(self, sess, q_func, a, y, loss, grad_update, replay_memory, target_func):
        """
        学習を実行．
        @todo: open ai実装と比較．実装がかなり怪しい．

        Parameters
        ----------
        sess:
            tensorflowのセッション
        q_func: model.CNN
            Q関数
        a: tf.placeholder(tf.int64, [None])
            エージェントが選択する行動値
        y: tf.placeholder(tf.float32, [None])
            教師信号．Q^*関数のQ値．
        loss:
            誤差関数．yとの誤差．
        grad_update:
            誤差を最小化する処理
        replay_memory: replay_memory.ReplayMemory
            Replay Memory．このメモリからミニバッチして学習．
            
        Returns
        ----------
        l: float
            教師信号との誤差
        """
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        # Replay Memoryからランダムにミニバッチをサンプル
        minibatch = replay_memory.sample(self.minibatch_size)

        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            done_batch.append(data[4])

        # 終了判定を，1（True），0（False）に変換
        done_batch = np.array(done_batch) + 0

        # Target Networkで次の状態でのQ値を計算
        target_q_values_batch = target_func.q_values.eval(
            feed_dict={target_func.s: np.float32(np.array(
                next_state_batch))})
        # 教師信号を計算
        y_batch = reward_batch + (
            1.0 - done_batch) * self.discount_factor * np.max(
            target_q_values_batch,
            axis=1)
        # 誤差最小化
        l, _ = sess.run([loss, grad_update], feed_dict={
            q_func.s: np.float32(np.array(state_batch)),
            a: action_batch,
            y: y_batch
        })
        return l

    def learn(self):
        """
        mainメソッド．
        DQNのアルゴリズムを回す．
        """
        # Replay Memory
        replay_memory = ReplayMemory(self.replay_memory_size)

        # Q-Network
        q_func = CNN(self.env.action_space.n, self.agent_history_length, self.frame_width, self.frame_height)
        q_network_weights = q_func.model.trainable_weights # 学習される重み
        # TargetNetwork
        target_func = CNN(self.env.action_space.n, self.agent_history_length, self.frame_width, self.frame_height)
        target_network_weights = target_func.model.trainable_weights  # 重みのリスト

        # 定期的にTargetNetworkをQ-Networkで同期する処理
        assign_target_network = [
                target_network_weights[i].assign(q_network_weights[i]) for i in range(len(target_network_weights))]

        # 誤差関数や最適化のための処理
        a, y, loss, grad_update = self.build_training_op(self.env.action_space.n, q_func)

        # Sessionの構築
        sess = tf.InteractiveSession()

        # 変数の初期化（Q Networkの初期化）
        sess.run(tf.global_variables_initializer())

        # Target Networkの初期化
        sess.run(assign_target_network)

        # ε．時間に対して線形で減少させる．
        epsilon = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * self.tmax),
                                initial_p=1.0,
                                final_p=self.final_exploration)

        # Logger
        logger = Logger(sess, self.save_summary_path)

        t = 0
        episode = 0
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
                # 前の状態を保存
                pre_observation = observation.copy()
                # ε-greedyに従って行動を選択
                action = self.choose_action_by_epsilon_greedy(self.env.action_space.n, q_func, epsilon.value(t), observation)
                # 行動を実行し，報酬と次の画面とdoneを観測
                observation, reward, done, info = self.env.step(action)
                # replay memoryに(s_t,a_t,r_t,s_{t+1},done)を追加
                # deepcopyになってるかな？？
                replay_memory.add(pre_observation, action, reward, observation, done)
                if self.render:
                    self.env.render()
                if t > self.replay_start_size and t % self.learn_frequency:
                    # Q-Networkの学習
                    total_loss += self.train(sess, q_func, a, y, loss, grad_update, replay_memory, target_func)
                if t > self.replay_start_size and t % self.target_network_update_frequency == 0:
                    # Target Networkの更新
                    sess.run(assign_target_network)
                if t > self.replay_start_size and t % self.save_network_frequency == 0:
                    save_sess(sess, self.save_network_path, t)
                total_reward += reward
                total_q_max += np.max(q_func.q_values.eval(
                    feed_dict={q_func.s: [observation]}))
                t += 1
                duration += 1
            if t >= self.replay_start_size:
                logger.write(sess, total_reward, total_q_max / float(duration),
                             duration, total_loss / float(duration), t, episode)
            if t < self.replay_start_size:
                mode = 'random'
            elif self.replay_start_size <= t < self.replay_start_size + self.exploration_fraction * self.tmax:
                mode = 'explore'
            else:
                mode = 'exploit'
            print(
                'EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.0f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7}'.format(
                    episode, t, duration, epsilon.value(t),
                    total_reward, total_q_max / float(duration),
                    total_loss / float(duration), mode))

    def test(self):
        # Q-Network
        q_func = CNN(self.env.action_space.n, self.agent_history_length, self.frame_width, self.frame_height)
        # Sessionの構築
        sess = tf.InteractiveSession()
        # session読み込み
        restore_sess(sess, self.save_network_path)

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
            observation = self.env.reset()
            # エピソード終了まで実行
            while not done:
                # 行動を選択
                action = self.choose_action_by_epsilon_greedy(self.env.action_space.n, q_func, 0.0, observation)
                # 行動を実行し，報酬と次の画面とdoneを観測
                observation, reward, done, info = self.env.step(action)
                self.env.render()
                total_reward += reward
                t += 1
                duration += 1
            print('EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / TOTAL_REWARD: {3:3.0f}'.format(
                    episode, t, duration, total_reward))


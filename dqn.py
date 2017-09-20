# coding:utf-8
import random
import os
import numpy as np
import gym
import tensorflow as tf
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.layers import Conv2D, Flatten, Dense
from keras.models import Sequential


class DQN():
    def __init__(self, **params):
        self.env_name = params['env_name']
        self.load_network = params['load_network']
        self.train = params['train']
        self.seed = params['seed']
        self.tmax = params['tmax']
        self.episode_num_at_test = params['episode_num_at_test']
        self.save_network_freq = params['save_network_freq']
        self.save_network_path = params['save_network_path']
        self.save_summary_path = params['save_summary_path']

        self.frame_width = params['frame_width']
        self.frame_height = params['frame_height']

        self.minibatch_size = params['minibatch_size']
        self.replay_memory_size = params['replay_memory_size']
        self.agent_history_length = params['agent_history_length']
        self.target_network_update_frequency = params[
            'target_network_update_frequency']
        self.discount_factor = params['discount_factor']
        self.action_repeat = params['action_repeat']
        self.learning_rate = params['learning_rate']
        self.gradient_momentum = params['gradient_momentum']
        self.squared_gradient_momuntum = params['squared_gradient_momuntum']
        self.min_squared_gradient = params['min_squared_gradient']
        self.initial_exploration = params['initial_exploration']
        self.final_exploration = params['final_exploration']
        self.final_exploration_frame = params['final_exploration_frame']
        self.replay_start_size = params['replay_start_size']
        self.no_op_max = params['no_op_max']

    def get_action_num(self, env):
        """
        学習対象のatariゲームの行動数を返す
        :return:
            num_actions: 行動数
        """
        num_actions = env.action_space.n
        if (self.env_name == "Pong-v0" or self.env_name == "Breakout-v0"):
            # Gymの環境では，pongとbreakoutの行動数は6になっているが，
            # 3つしか必要としない．
            num_actions = 3
            acts = [1, 2, 3]
        else:
            acts = range(num_actions)
        return num_actions, acts

    def init_replay_memory(self):
        """
        Replay Memoryの初期化
        :return:
        """
        self.replay_memory = deque()

    def init_q_network(self, num_actions):
        """
        Q-Networkをランダムな重みで初期化
        :param num_actions:　行動数
        :return:
            s: Q-Networkへの入力
            q_values: Q-Networkの出力
            model: Q-Networkのモデル
        """
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(self.agent_history_length, self.frame_width, self.frame_height), data_format='channels_first'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(num_actions))
        s = tf.placeholder(tf.float32,
                           [None, self.agent_history_length, self.frame_width,
                            self.frame_height])
        q_values = model(s)

        return s, q_values, model

    def update_target_network(self):
        """
        Target Networkを更新．
        Q-Networkと重みを同期する．
        :return:
        """
        self.sess.run(self.assign_target_network)

    def choose_action_by_epsilon_greedy(self, action_num, state):
        if self.epsilon >= np.random.rand():
            action = np.random.randint(action_num)
        else:
            action = np.argmax(self.q_values.eval(
                feed_dict={self.s: [np.float32(state / 255.0)]}))
        return action

    def first_preprosess_state(self, observation, last_observation):
        # 現在のゲーム画面と前のゲーム画面の，各ピクセルごとの最大値を取る
        processed_observation = np.maximum(observation, last_observation)
        # グレースケール変換，リサイズ
        processed_observation = np.uint8(
            resize(rgb2gray(processed_observation),
                   (self.frame_width, self.frame_height)) * 255)
        # 同じ画像を並べてスタック
        state = [processed_observation for _ in
                 range(self.agent_history_length)]
        return np.stack(state, axis=0)

    def preprocess_state(self, state, observation, last_observation):
        # 現在のゲーム画面と前のゲーム画面の，各ピクセルごとの最大値を取る
        processed_observation = np.maximum(observation, last_observation)
        # グレースケール変換，リサイズ
        processed_observation = np.uint8(
            resize(rgb2gray(processed_observation),
                   (self.frame_width, self.frame_height)) * 255)
        processed_observation = np.reshape(processed_observation, (
            1, self.frame_width, self.frame_height))
        # 次の状態を作成
        next_state = np.append(state[1:, :, :], processed_observation, axis=0)
        return next_state

    def add_memory(self, state, action, reward, next_state, done):
        # 報酬が正なら1に，負なら-1に変換
        reward = np.sign(reward)
        # Replay Memoryに遷移を保存
        self.replay_memory.append((state, action, reward, next_state, done))
        # Replay Memoryが一定数を超えたら，古い遷移から削除
        if len(self.replay_memory) > self.replay_memory_size:
            self.replay_memory.popleft()

    def learn(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        # Replay Memoryからランダムにミニバッチをサンプル
        minibatch = random.sample(self.replay_memory, self.minibatch_size)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            done_batch.append(data[4])

        # 終了判定を，1（True），0（False）に変換
        done_batch = np.array(done_batch) + 0

        target_q_values_batch = self.target_q_values.eval(
            feed_dict={self.target_s: np.float32(np.array(
                next_state_batch) / 255.0)})  # Target Networkで次の状態でのQ値を計算
        y_batch = reward_batch + (
                                     1.0 - done_batch) * self.discount_factor * np.max(
            target_q_values_batch,
            axis=1)  # 教師信号を計算
        # 勾配法による誤差最小化
        loss, _ = self.sess.run([self.loss, self.grad_update], feed_dict={
            self.s: np.float32(np.array(state_batch) / 255.0),
            self.a: action_batch,
            self.y: y_batch
        })
        return loss

    def build_training_op(self, num_actions):
        a = tf.placeholder(tf.int64, [None])  # 行動
        y = tf.placeholder(tf.float32, [None])  # 教師信号

        a_one_hot = tf.one_hot(a, num_actions, 1.0, 0.0)  # 行動をone hot vectorに変換する
        q_value = tf.reduce_sum(tf.multiply(self.q_values, a_one_hot), reduction_indices=1)  # 行動のQ値の計算
        # エラークリップ
        error = y - q_value
        quadratic_part = tf.clip_by_value(error, -1.0, 1.0)
        loss = tf.reduce_mean(tf.square(quadratic_part))  # 誤差関数

        optimizer = tf.train.RMSPropOptimizer(self.learning_rate,
                                              momentum=self.gradient_momentum,
                                              decay=self.squared_gradient_momuntum,
                                              epsilon=self.min_squared_gradient)  # 最適化手法を定義
        grad_update = optimizer.minimize(loss,
                                         var_list=self.q_network_weights)  # 誤差最小化

        return a, y, loss, grad_update

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        tf.summary.scalar(self.env_name + '/Total Reward/Episode', episode_total_reward)
        episode_avg_max_q = tf.Variable(0.)
        tf.summary.scalar(self.env_name + '/Average Max Q/Episode', episode_avg_max_q)
        episode_duration = tf.Variable(0.)
        tf.summary.scalar(self.env_name + '/Duration/Episode', episode_duration)
        episode_avg_loss = tf.Variable(0.)
        tf.summary.scalar(self.env_name + '/Average Loss/Episode', episode_avg_loss)
        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def choose_action_at_test(self, action_num, state):
        if np.random.rand() < 0.0:
            action = np.random.randint(action_num)
        else:
            avalue = self.q_values.eval(
                feed_dict={self.s: [np.float32(state / 255.0)]})
            action = np.argmax(avalue)
            # print(avalue)
        return action

    def main(self):
        # シードセット（統一できてない）
        np.random.seed(self.seed)
        # 環境
        env = gym.make(self.env_name)
        action_num, acts = self.get_action_num(env)
        # デフォルトではフレームがスキップされるので，1に戻す
        # env.frameskip = 1

        # Replay Memoryを初期化
        self.init_replay_memory()

        # Q-Networkをランダムな重みθで初期化
        self.s, self.q_values, self.q_network = self.init_q_network(action_num)
        self.q_network_weights = self.q_network.trainable_weights  # 学習される重みのリスト

        # Target_networkを重みθ^-=θで初期化
        self.target_s, self.target_q_values, self.target_network = self.init_q_network(
            action_num)
        self.target_network_weights = self.target_network.trainable_weights  # 重みのリスト

        # 定期的にTarget Networkを更新するための処理の構築
        self.assign_target_network = [
            self.target_network_weights[i].assign(self.q_network_weights[i]) for
            i in
            range(len(self.target_network_weights))]

        # 誤差関数や最適化のための処理の構築
        self.a, self.y, self.loss, self.grad_update = self.build_training_op(
            action_num)

        # Sessionの構築
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver(self.q_network_weights)
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(self.save_summary_path, self.sess.graph)

        if not os.path.exists(self.save_network_path):
            os.makedirs(self.save_network_path)

        # 変数の初期化（Q Networkの初期化）
        self.sess.run(tf.initialize_all_variables())

        # Target Networkの初期化
        self.sess.run(self.assign_target_network)

        if self.load_network:
            checkpoint = tf.train.get_checkpoint_state(self.save_network_path)
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
                print(
                    'Successfully loaded: ' + checkpoint.model_checkpoint_path)
            else:
                print('Training new network...')

        t = 0
        if self.train:
            self.epsilon = self.initial_exploration

            # ε-greedyの設定
            episode = 0
            while t < self.tmax:
                episode += 1
                done = False
                observation = env.reset()
                # ランダムなフレーム数分「何もしない」
                T = np.random.randint(1, self.no_op_max)
                for _ in range(T):
                    last_observation = observation
                    # 「何もしない」で，次の画面を返す
                    observation, _, _, _ = env.step(acts[0])
                last_action = 0
                # 初期状態の生成
                state = self.first_preprosess_state(observation,
                                                    last_observation)
                duration = 0
                total_reward = 0.0
                total_q_max = 0.0
                total_loss = 0
                while not done:
                    last_observation = observation
                    if t <= self.replay_start_size:
                        # ランダムに行動を決定
                        action = np.random.randint(action_num)
                    else:
                        if t % self.action_repeat == 0:
                            # ε-greedyに従って行動を選択
                            action = self.choose_action_by_epsilon_greedy(
                                action_num, state)
                        else:
                            # 前フレームに選択した行動を選択
                            action = last_action
                        # εを線形減少
                        self.epsilon = max(self.final_exploration,
                                           self.epsilon - (
                                               self.initial_exploration - self.final_exploration) / self.final_exploration_frame)
                        # ネットワークを保存
                        if t % self.save_network_freq == 0:
                            save_path = self.saver.save(self.sess,
                                                        self.save_network_path + '/' + self.env_name,
                                                        global_step=t)
                            print('Successfully saved: ' + save_path)
                    last_action = action
                    # 行動を実行し，報酬と次の画面とdoneを観測
                    observation, reward, done, info = env.step(acts[action])
                    env.render()
                    # 前処理して次の状態s_{t+1}を生成
                    next_state = self.preprocess_state(state, observation,
                                                       last_observation)
                    # replay memoryに(s_t,a_t,r_t,s_{t+1},done)を追加
                    # deepcopyになってるかな？？
                    self.add_memory(state, action, reward, next_state, done)
                    # 状態を更新
                    state = next_state.copy()
                    if t > self.replay_start_size:
                        if t % self.action_repeat == 0:
                            # Q-Networkの学習
                            total_loss += self.learn()
                        if t % (
                                    self.target_network_update_frequency * self.action_repeat) == 0:
                            # Target Networkの更新
                            self.update_target_network()
                    total_reward += reward
                    total_q_max += np.max(self.q_values.eval(
                        feed_dict={self.s: [np.float32(state / 255.0)]}))
                    t += 1
                    duration += 1
                    # Write summary
                if t >= self.replay_start_size:
                    stats = [total_reward, total_q_max / float(duration),
                             duration, total_loss / (
                                 float(duration) / float(self.action_repeat))]
                    for i in range(len(stats)):
                        self.sess.run(self.update_ops[i], feed_dict={
                            self.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = self.sess.run(self.summary_op)
                    self.summary_writer.add_summary(summary_str, episode + 1)
                if t < self.replay_start_size:
                    mode = 'random'
                elif self.replay_start_size <= t < self.replay_start_size + self.final_exploration_frame:
                    mode = 'explore'
                else:
                    mode = 'exploit'
                print(
                    'EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.0f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7}'.format(
                        episode + 1, t, duration, self.epsilon,
                        total_reward, total_q_max / float(duration),
                        total_loss / (
                            float(duration) / float(self.action_repeat)), mode))
        else:
            # env.monitor.start(ENV_NAME + '-test')
            for episode in range(self.episode_num_at_test):
                terminal = False
                observation = env.reset()
                # ランダムなフレーム数分「何もしない」
                T = np.random.randint(1, self.no_op_max)
                for _ in range(T):
                    last_observation = observation
                    observation, _, _, _ = env.step(acts[0])  # Do nothing
                    env.render()
                last_action = 0
                state = self.first_preprosess_state(observation,
                                                    last_observation)
                duration = 0
                total_reward = 0.0
                while not terminal:
                    env.render()
                    if np.sum(last_observation != observation) == 0:
                        action = 0
                    elif t % self.action_repeat == 0:
                        # ε-greedyに従って行動を選択
                        action = self.choose_action_at_test(action_num, state)
                    else:
                        # 前フレームに選択した行動を選択
                        action = last_action
                    last_observation = observation
                    observation, reward, terminal, _ = env.step(acts[action])
                    next_state = self.preprocess_state(state, observation,
                                                       last_observation)
                    # 状態を更新
                    state = next_state.copy()
                    total_reward += reward
                    duration += 1
                print(
                    'EPISODE: {0:6d} / DURATION: {1:5d} / TOTAL_REWARD: {2:3.0f}'.format(
                        episode + 1, duration,
                        total_reward))
                # env.monitor.close()


if __name__ == '__main__':
    params = {}
    params['env_name'] = 'Breakout-v0'  # 環境の名称
    params['load_network'] = True
    params['train'] = False
    params['seed'] = 0  # 乱数シード
    params['save_network_freq'] = 300000  # Q_networkを保存する頻度（フレーム数で計測）
    params['save_network_path'] = 'saved_networks/' + params['env_name']
    params['save_summary_path'] = 'summary/' + params['env_name']
    params['tmax'] = 50000000  # 学習をやめる行動数
    params['episode_num_at_test'] = 100
    params['frame_width'] = 84  # リサイズ後のフレー0ム幅
    params['frame_height'] = 84  # リサイズ後のフレーム高さ

    # DQNのユーザパラメータ
    params['minibatch_size'] = 32  # SGDによる更新に用いる訓練データの数
    # params['replay_memory_size'] = 1000000  # SGDによる更新に用いるデータは，このサイズの直近のフレームデータからサンプルする
    params[
        'replay_memory_size'] = 500000  # SGDによる更新に用いるデータは，このサイズの直近のフレームデータからサンプルする
    params['agent_history_length'] = 4  # Q_networkの入力として与える，直近のフレームの数
    params[
        'target_network_update_frequency'] = 10000  # target_networkが更新される頻度（パラメータの更新頻度数で計測）
    params['discount_factor'] = 0.99  # Q_learningの更新でも用いられる割引率γ
    params['action_repeat'] = 1  # エージェントは，このフレーム数毎に行動選択を行う．
    params['learning_rate'] = 0.00025  # RMSPropで使用される学習率
    params['gradient_momentum'] = 0.95  # RMSPropで使用されるGradient momuntum
    params[
        'squared_gradient_momuntum'] = 0.95  # RMSPropで使用されるSquared gradient (denominator) momentum
    params[
        'min_squared_gradient'] = 0.01  # RMSPropの更新の際，Squared gradient (denominator)に加算される定数
    params['initial_exploration'] = 1.0  # ε-greedyにおけるεの初期値
    params['final_exploration'] = 0.1  # ε-greedyにおけるεの最終値
    params[
        'final_exploration_frame'] = 1000000  # εが初期値から最終値に線形減少した時に，最終値に到達するフレーム数
    # params['replay_start_size'] = 50000  # 学習が始める前に，このフレーム数に対して一様ランダムに行動を選択する政策が実行され，その経験がReplay memoryに蓄えられる
    params[
        'replay_start_size'] = 25000  # 学習が始める前に，このフレーム数に対して一様ランダムに行動を選択する政策が実行され，その経験がReplay memoryに蓄えられる
    params['no_op_max'] = 5  # エピソードの開始時に，エージェントが「何もしない」行動を選択するフレーム数
    dqn = DQN(**params)
    dqn.main()

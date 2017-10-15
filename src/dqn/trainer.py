import tensorflow as tf

from dqn.replay_memory import ReplayMemory
from models.models import CNN
from util.log.logger import save_sess, restore_sess, Logger
from agents.agents import *


class Trainer(object):
    def __init__(self, env, args):
        """
        学習を行うTrainerクラス．

        Parameters
        ----------
        env: gym.wrappers.time_limit.TimeLimit
            open ai gymの環境
        args:
            学習に必要なパラメータのリスト
        """
        self.env = env
        self.width = args.width
        self.height = args.height

        self.tmax = args.tmax
        self.batch_size = args.batch_size
        self.mem_size = args.mem_size
        self.history_len = args.history_len
        self.update_freq = args.update_freq
        self.discount_fact = args.discount_fact
        self.learn_freq = args.learn_freq
        self.learn_rate = args.learn_rate
        self.fin_expl = args.fin_expl
        self.expl_frac = args.expl_frac
        self.replay_st_size = args.replay_st_size

        self.render = args.render
        self.save_network_freq = args.save_network_freq
        self.save_network_path = "../data/" + args.save_network_path + "/" + args.env_name + \
                                 args.save_option_name + "/model.ckpt"
        self.save_summary_path = "../data/" + args.save_summary_path + "/" + args.env_name + args.save_option_name

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
        a: tf.python.framework.ops.Tensor(tf.int64, [None])
            エージェントが選択する行動値
        y: tf.python.framework.ops.Tensor(tf.float32, [None])
            教師信号．Q^*関数のQ値．
        loss: tf.python.framework.ops.Tensor
            誤差関数．yとの誤差．
        grad_update: tf.python.framework.ops.Tensor
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
        errors = tf.where(tf.abs(error) < 1.0, tf.square(error) * 0.5, tf.abs(error) - 0.5)
        # 誤差関数
        loss = tf.reduce_mean(errors)
        # 最適化手法を定義
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate)
        # 誤差最小化の処理
        grad_update = optimizer.minimize(loss, var_list=q_func.model.trainable_weights)

        return a, y, loss, grad_update

    def train(self, sess, q_func, a, y, loss, grad_update, replay_mem, target_func):
        """
        学習を実行．

        Parameters
        ----------
        sess: tf.python.client.session.InteractiveSession
            tensorflowのセッション
        q_func: model.CNN
            Q関数
        a: tf.python.framework.ops.Tensor(tf.int64, [None])
            エージェントが選択する行動値
        y: tf.python.framework.ops.Tensor(tf.float32, [None])
            教師信号．Q^*関数のQ値．
        loss: tf.python.framework.ops.Tensor
            誤差関数．yとの誤差．
        grad_update: tf.python.framework.ops.Tensor
            誤差を最小化する処理
        replay_mem: replay_memory.ReplayMemory
            Replay Memory．このメモリからミニバッチして学習．
        target_func: model.CNN
            Target Network

        Returns
        ----------
        l: numpy.float32
            教師信号との誤差
        """
        obs_batch = []
        action_batch = []
        reward_batch = []
        next_obs_batch = []
        done_batch = []
        # Replay Memoryからランダムにミニバッチをサンプル
        batch = replay_mem.sample(self.batch_size)

        for data in batch:
            obs_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_obs_batch.append(data[3])
            done_batch.append(data[4])

        # 終了判定を，1（True），0（False）に変換
        done_batch = np.array(done_batch) + 0

        # Target Networkで次の状態でのQ値を計算
        target_q_values_batch = target_func.q_values.eval(
            feed_dict={target_func.s: np.float32(np.array(
                next_obs_batch))})
        # 教師信号を計算
        y_batch = reward_batch + (
            1.0 - done_batch) * self.discount_fact * np.max(
            target_q_values_batch,
            axis=1)
        # 誤差最小化
        l, _ = sess.run([loss, grad_update], feed_dict={
            q_func.s: np.float32(np.array(obs_batch)),
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
        replay_mem = ReplayMemory(self.mem_size)

        # Q-Network
        q_func = CNN(self.env.action_space.n, self.history_len, self.width, self.height)
        q_network_weights = q_func.model.trainable_weights  # 学習される重み
        # TargetNetwork
        target_func = CNN(self.env.action_space.n, self.history_len, self.width, self.height)
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

        # エージェント初期化
        agent = DQNAgent(num_actions=self.env.action_space.n,
                         q_func=q_func,
                         schedule_time_steps=int(self.expl_frac * self.tmax),
                         initial_time_step=self.replay_st_size,
                         final_p=self.fin_expl)

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
            obs = self.env.reset()
            # エピソード終了まで実行
            while not done:
                # 前の状態を保存
                pre_obs = obs.copy()
                # ε-greedyに従って行動を選択
                action = agent.action(t, obs)
                # 行動を実行し，報酬と次の画面とdoneを観測
                obs, reward, done, info = self.env.step(action)
                # replay memoryに(s_t,a_t,r_t,s_{t+1},done)を追加
                replay_mem.add(pre_obs, action, reward, obs, done)
                if self.render:
                    self.env.render()
                if t > self.replay_st_size and t % self.learn_freq:
                    # Q-Networkの学習
                    total_loss += self.train(sess, q_func, a, y, loss, grad_update, replay_mem, target_func)
                if t > self.replay_st_size and t % self.update_freq == 0:
                    # Target Networkの更新
                    sess.run(assign_target_network)
                if t > self.replay_st_size and t % self.save_network_freq == 0:
                    save_sess(sess, self.save_network_path, t)
                total_reward += reward
                total_q_max += np.max(q_func.q_values.eval(
                    feed_dict={q_func.s: [obs]}))
                t += 1
                duration += 1
            if t >= self.replay_st_size:
                logger.write(sess, total_reward, total_q_max / float(duration),
                             duration, total_loss / float(duration), t, episode)
            print(
                'EPISODE: {0:6d} / TIME_STEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.0f} '
                '/ AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f}'.format(
                    episode, t, duration, agent.epsilon.value(t),
                    total_reward, total_q_max / float(duration),
                    total_loss / float(duration)))

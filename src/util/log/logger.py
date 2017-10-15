import os
import tensorflow as tf


def save_sess(sess, file_name, t):
    """
    学習中のセッションを保存．

    Parameters
    ----------
    sess: tf.python.client.session.InteractiveSession
        tensorflowのセッション
    file_name: str
        書き込み先のパス
    t: int
        ステップ数
    """
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    saver = tf.train.Saver()
    save_path = saver.save(sess, file_name, global_step=t)
    print('Successfully saved: ' + save_path)


def restore_sess(sess, file_name):
    """
    セッションを復元．

    Parameters
    ----------
    sess: tf.python.client.session.InteractiveSession
        tensorflowのセッション
    file_name: str
        読み込み先のパス
    """

    ckpt = tf.train.get_checkpoint_state(os.path.dirname(file_name))
    if ckpt:
        model_path = ckpt.model_checkpoint_path
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        print('Successfully loaded: ' + ckpt.model_checkpoint_path)
    else:
        print('Training new network')


class Logger(object):
    def __init__(self, sess, file_name):
        """
        学習推移を保存するLogger．

        Parameters
        ----------
        sess: tf.python.client.session.InteractiveSession
            tensorflowのセッション
        file_name: str
            書き込み先のパス
        """
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        episode_total_reward = tf.Variable(0.)
        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        episode_avg_max_q = tf.Variable(0.)
        tf.summary.scalar('Average max Q/Episode', episode_avg_max_q)
        episode_duration = tf.Variable(0.)
        tf.summary.scalar('Duration/Episode', episode_duration)
        episode_avg_loss = tf.Variable(0.)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)
        total_step = tf.Variable(0.)
        tf.summary.scalar('Total Step', total_step)
        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss, total_step]
        self.summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        self.update_ops = [summary_vars[i].assign(self.summary_placeholders[i]) for i in range(len(summary_vars))]
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(file_name, sess.graph)

    def write(self, sess, total_reward, average_q_max, duration, average_loss, t, episode):
        """
        エピソード毎の学習結果を書き込み．

        Parameters
        ----------
        sess: tf.python.client.session.InteractiveSession
            tensorflowのセッション
        total_reward: numpy.float64
            エピソードの合計報酬
        average_q_max: numpy.float64
            Q値のエピソード内の平均値
        duration: int
            エピソードの長さ
        average_loss: numpy.float64
            損失誤差のエピソード内の平均値
        t: int
            総ステップ数
        episode: int
            総エピソード数
        """
        stats = [total_reward, average_q_max, duration, average_loss, t]
        for i in range(len(stats)):
            sess.run(self.update_ops[i], feed_dict={
                self.summary_placeholders[i]: float(stats[i])
            })
        summary_str = sess.run(self.summary_op)
        self.summary_writer.add_summary(summary_str, episode)

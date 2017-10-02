import os
import tensorflow as tf

def save_sess(sess, file_name, t):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    saver = tf.train.Saver()
    save_path = saver.save(sess, file_name, global_step=t)
    print('Successfully saved: ' + save_path)

def restore_sess(sess, file_name):
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
        episode_total_reward = tf.Variable(0.)
        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        episode_avg_max_q = tf.Variable(0.)
        tf.summary.scalar('Average max Q/Episode', episode_avg_max_q)
        episode_duration = tf.Variable(0.)
        tf.summary.scalar('Duration/Episode', episode_duration)
        episode_avg_loss = tf.Variable(0.)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)
        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss]
        self.summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        self.update_ops = [summary_vars[i].assign(self.summary_placeholders[i]) for i in range(len(summary_vars))]
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(file_name, sess.graph)

    def write(self, sess, total_reward, average_q_max, duration, average_loss, episode):
        stats = [total_reward, average_q_max, duration, average_loss]
        for i in range(len(stats)):
            sess.run(self.update_ops[i], feed_dict={
                self.summary_placeholders[i]: float(stats[i])
            })
        summary_str = sess.run(self.summary_op)
        self.summary_writer.add_summary(summary_str, episode)




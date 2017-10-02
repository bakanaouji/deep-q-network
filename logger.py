import os
import tensorflow as tf

def save_sess(sess, file_name, t):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    saver = tf.train.Saver()
    saver.save(sess, file_name, global_step=t)

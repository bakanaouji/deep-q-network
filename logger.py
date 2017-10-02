import os
import tensorflow as tf

def save_sess(sess, path, t):
    os.makedirs(path, exist_ok=True)
    saver = tf.train.Saver()
    saver.save(sess, path, global_step=t)

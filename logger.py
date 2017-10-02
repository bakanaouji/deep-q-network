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

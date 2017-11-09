import tensorflow as tf
from keras.layers import Conv2D, Flatten, Dense
from keras.models import Sequential


class CNN(object):
    def __init__(self, num_actions, agent_history_length, frame_width, frame_height):
        """
        Q-Networkをランダムな重みで初期化

        Parameters
        ----------
        num_actions: int
            行動数
        agent_history_length: int
            入力として受け取る観測の履歴の数
        frame_width: int
            画面の幅
        frame_height: int
            画面の高さ
        """
        self.model = Sequential()
        self.model.add(
            Conv2D(
                32, (8, 8), strides=(4, 4), activation='relu',
                input_shape=(
                    agent_history_length,
                    frame_width,
                    frame_height
                ),
                data_format='channels_first'
            )
        )
        self.model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu', data_format='channels_first'))
        self.model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', data_format='channels_first'))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(num_actions))
        self.s = tf.placeholder(
            tf.float32, [None, agent_history_length, frame_width, frame_height]
        )
        self.q_values = self.model(self.s)

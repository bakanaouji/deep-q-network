import tensorflow as tf
from keras.layers import Conv2D, Flatten, Dense
from keras.models import Sequential

class CNN(object):
    def __init__(self, num_actions, agent_history_length, frame_width, frame_height):
        """
        Q-Networkをランダムな重みで初期化
        :param num_actions:　行動数
        :return:
            s: Q-Networkへの入力
            q_values: Q-Networkの出力
            model: Q-Networkのモデル
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
        self.model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(num_actions))
        self.s = tf.placeholder(
                tf.float32, [None, agent_history_length, frame_width, frame_height]
                )
        self.q_values = self.model(self.s)

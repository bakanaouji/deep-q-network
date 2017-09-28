import tensorflow as tf
from keras.layers import Conv2D, Flatten, Dense
from keras.models import Sequential

def init_q_network(num_actions, agent_history_length, frame_width, frame_height):
    """
    Q-Networkをランダムな重みで初期化
    :param num_actions:　行動数
    :return:
        s: Q-Networkへの入力
        q_values: Q-Networkの出力
        model: Q-Networkのモデル
    """
    # @todo 初期化方法を指定
    model = Sequential()
    model.add(
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
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_actions))
    s = tf.placeholder(
        tf.float32,[None, agent_history_length, frame_width, frame_height]
        )
    q_values = model(s)

    return s, q_values, model

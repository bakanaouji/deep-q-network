import unittest

from keras.utils.vis_utils import plot_model

from models.models import CNN


class TestModel(unittest.TestCase):
    def test_cnn(self):
        num_actions = 4
        agent_history_length = 4
        frame_width = 84
        frame_height = 84
        cnn = CNN(num_actions, agent_history_length, frame_width, frame_height)
        s_shape = cnn.s.get_shape().as_list()
        self.assertEqual(s_shape, [None, agent_history_length, frame_width, frame_height])
        q_values_shape = cnn.q_values.get_shape().as_list()
        self.assertEqual(q_values_shape, [None, num_actions])
        plot_model(cnn.model, show_shapes=True, show_layer_names=True, to_file='model.png')


if __name__ == '__main__':
    unittest.main()

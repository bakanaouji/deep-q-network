import unittest
import model
import tensorflow as tf
import numpy as np
from keras.utils.vis_utils import plot_model

class TestModel(unittest.TestCase):
    
    def test_init_q_network(self):
        num_actions = 4
        agent_history_length = 4
        frame_width = 84
        frame_height = 84
        s, q_values, q_network = model.init_q_network(num_actions, agent_history_length, frame_width, frame_height)
        s_shape = s.get_shape().as_list()
        self.assertEqual(s_shape, [None, agent_history_length, frame_width, frame_height])
        q_values_shape = q_values.get_shape().as_list()
        self.assertEqual(q_values_shape, [None, num_actions])
        plot_model(q_network, to_file='model.png')

if __name__ == '__main__':
    unittest.main()

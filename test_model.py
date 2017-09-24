import unittest
import model

class TestModel(unittest.TestCase):
    
    def test_init_q_network(self):
        num_actions = 4
        agent_history_length = 4
        frame_width = 84
        frame_height = 84
        s, q_values, q_network = model.init_q_network(num_actions, agent_history_length, frame_width, frame_height)

if __name__ == '__main__':
    unittest.main()

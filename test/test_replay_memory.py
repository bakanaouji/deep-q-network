import unittest

from dqn.replay_memory import ReplayMemory


class TestReplayMemory(unittest.TestCase):
    def test_add(self):
        size = 5
        replay_memory = ReplayMemory(size)
        for i in range(size):
            replay_memory.add(i, i + size, i + size * 2, i + size * 3, i + size * 4)
        for i in range(size):
            data = replay_memory[i]
            for j in range(5):
                self.assertEqual(data[j], i + size * j)

    def test_len(self):
        max_size = 10
        size = 5
        replay_memory = ReplayMemory(max_size)
        for i in range(size):
            replay_memory.add(i, i + size, i + size * 2, i + size * 3, i + size * 4)
        self.assertEqual(min(size, max_size), len(replay_memory))

    def test_sample(self):
        size = 5
        sample_size = 3
        replay_memory = ReplayMemory(size)
        for i in range(size):
            replay_memory.add(i, i + size, i + size * 2, i + size * 3, i + size * 4)
        sample = replay_memory.sample(sample_size)
        print(sample)


if __name__ == '__main__':
    unittest.main()

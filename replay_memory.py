import numpy as np
import random
from collections import deque

class ReplayMemory(object):
    def __init__(self, size):
        self._deque = deque()
        self._max_size = size

    def __len__(self):
        return len(self._deque)

    def add(self, state, action, reward, next_state, done):
        # Replay Memoryに遷移を保存
        self._deque.append((state, action, reward, next_state, done))
        # Replay Memoryが一定数を超えたら，古い遷移から削除
        if len(self._deque) > self._max_size:
            self._deque.popleft()

    def sample(self, minibatch_size):
        # Replay Memoryからランダムにミニバッチをサンプル
        # @todo 非復元か復元か確認
        idxes = [np.random.randint(0, len(self._deque)) for _ in range(minibatch_size)]
        data = [self._deque[i] for i in idxes]
        return data

    def __getitem__(self, index):
        return self._deque[index]

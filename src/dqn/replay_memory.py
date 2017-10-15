import numpy as np
from collections import deque


class ReplayMemory(object):
    def __init__(self, size):
        """
        Experience Replayに用いるためのReplay Memory．
        指定フレーム数分の直近の履歴から，ミニバッチがサンプルされる．

        Parameters
        ----------
        size: int
            保存する履歴の数
        """
        self._deque = deque(maxlen=size)

    def __len__(self):
        return len(self._deque)

    def add(self, obs, action, reward, next_obs, done):
        """
        Replay Memoryに遷移を保存．

        Parameters
        ----------
        obs: np.ndarray
            観測
        action: int
            観測に対して選択した行動
        reward: numpy.float64
            行動の結果得られた報酬
        next_obs: numpy.ndarray
            行動を取った後の観測
        done: bool
            エピソードが終了したかどうか
        """
        self._deque.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        """
        Replay Memoryからランダムにミニバッチをサンプル．

        Parameters
        ----------
        batch_size: int
            バッチサイズ
        """
        indexes = [np.random.randint(0, len(self._deque)) for _ in range(batch_size)]
        data = [self._deque[i] for i in indexes]
        return data

    def __getitem__(self, index):
        return self._deque[index]

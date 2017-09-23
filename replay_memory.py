class ReplayMemory():
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

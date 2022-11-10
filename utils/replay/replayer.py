import numpy as np

from collections import deque
import random

from utils.replay.replayer_interface import ReplayInterface

class Replay(ReplayInterface):
    def __init__(self, capacity: int = 1e6):
        self.capacity = capacity
        self._buffer = deque()

    def __len__(self):
        return len(self._buffer)

    def __getitem__(self, idx: int):
        return self._buffer[idx]

    def push(self, transition):
        self._buffer.append(transition)
        if len(self._buffer) >= self.capacity:
            self._buffer.popleft()
    
    def batch_replay(self, batch_size: int):
        idx_batch = random.sample(range(len(self._buffer)), batch_size)
        return [self._buffer[idx_batch[i]] for i in range(batch_size)] 


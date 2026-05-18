import numpy as np
import random
from collections import deque
from typing import Tuple
from config import DuelingDQNConfig

class ReplayBuffer:
    
    def __init__(self):
        capacity = DuelingDQNConfig.MEMORY_SIZE
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batc_size: int) -> Tuple:
        # Buffer'dan rastgele batch_size kadar deneyim çeker
        batch = random.sample(self.buffer, batc_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )
    
    def __len__(self) -> int:
        # Buffer'da şu an kaç deneyim olduğunu döndürür
        return len(self.buffer)
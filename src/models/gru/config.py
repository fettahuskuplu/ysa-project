import torch

class GRUDQNConfig:
    # Veriseti: 2015-2021 BIST Verisi
    STATE_WINDOW_SIZE = 30
    ACTION_SIZE = 3

    # Neural Network (GRU)
    HIDDEN_SIZE = 64
    NUM_LAYERS = 1

    # Training
    LEARNING_RATE = 0.001
    GAMMA = 0.95
    BATCH_SIZE = 64
    EPISODES = 500  # 50 yerine 500 yapıyoruz.

    # Epsilon-Greedy Strategy
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995  # Buraya dokunmuyoruz, standart ve güvenli bir orandır.

    # Replay Buffer
    MEMORY_SIZE = 10000

    # Target Network
    TARGET_UPDATE_FREQUENCY = 5

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
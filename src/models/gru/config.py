import torch

class GRUDQNConfig:
    # Environment
    STATE_WINDOW_SIZE = 30
    ACTION_SIZE = 3

    # Neural Network (GRU)
    HIDDEN_SIZE = 64
    NUM_LAYERS = 1

    # Training
    LEARNING_RATE = 0.001        # GRU mimarisinde 0.001 genelde iyidir, gerekirse 0.0005'e çekilebilir.
    GAMMA = 0.99                 # Ekip standardı: Uzun vadeli ödülleri önemsemesi için 0.95'ten 0.99'a çıkarıldı.
    BATCH_SIZE = 64
    EPISODES = 300               # Hedeflenen optimum episode sayısı.

    # Epsilon-Greedy Strategy
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.985        # 300 episode için özel olarak hesaplandı. (Keşif süreci 250. episode civarı biter)

    # Replay Buffer
    MEMORY_SIZE = 10000

    # Target Network
    TARGET_UPDATE_FREQUENCY = 10 # Ekip standardı: Dueling DQN ile aynı koşullarda yarışması için eşitlendi.

    # Device Setup
    if torch.backends.mps.is_available():
        DEVICE = "mps"           # Apple Silicon donanım hızlandırması
    elif torch.cuda.is_available():
        DEVICE = "cuda"          # Nvidia GPU donanım hızlandırması
    else:
        DEVICE = "cpu"
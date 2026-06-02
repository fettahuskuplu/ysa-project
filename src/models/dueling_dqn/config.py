class DuelingDQNConfig:
    # Environment
    STATE_WINDOW_SIZE = 30
    ACTION_SIZE = 3 

    # Neural Network
    HIDDEN_SIZE = 128

    # Training
    LEARNING_RATE = 0.0005
    GAMMA = 0.99
    BATCH_SIZE = 64
    EPISODES = 200

    # Epsilon-Greedy Strategy
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.98

    # Replay Buffer
    MEMORY_SIZE = 50000

    # Target Network
    TARGET_UPDATE_FREQUENCY = 10

    # Device
    DEVICE = "cpu"
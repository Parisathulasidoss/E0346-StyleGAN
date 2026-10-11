"""Shared hyper-parameters for StyleGAN-CIFAR10."""
SEED = 42
Z_DIM = 128
W_DIM = 128
NUM_STYLES = 8  # 2 styled convs x 4 resolutions (4, 8, 16, 32)
CHANNELS = {4: 128, 8: 128, 16: 64, 32: 32}

TRAIN_LIMIT = 20000
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 2e-4
BETA_1 = 0.0
BETA_2 = 0.99

CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs"

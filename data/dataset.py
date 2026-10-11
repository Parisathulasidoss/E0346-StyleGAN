"""CIFAR-10 loading and tf.data pipeline (pixels scaled to [-1, 1])."""
import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import BATCH_SIZE, SEED, TRAIN_LIMIT


def load_cifar10(limit=TRAIN_LIMIT, synthetic=False):
    """Return float32 images in [-1, 1]. `synthetic=True` uses random data (smoke tests, no download)."""
    if synthetic:
        x = np.random.RandomState(SEED).randint(0, 256, size=(limit, 32, 32, 3)).astype("float32")
    else:
        (x, _), _ = keras.datasets.cifar10.load_data()
        x = x[:limit].astype("float32")
    return (x - 127.5) / 127.5


def make_dataset(images, batch_size=BATCH_SIZE):
    return (
        tf.data.Dataset.from_tensor_slices(images)
        .shuffle(len(images), seed=SEED, reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

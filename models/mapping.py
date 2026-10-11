"""Mapping network: latent z -> intermediate style space w."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import W_DIM


class MappingNetwork(keras.Model):
    """8 dense layers with pixel-norm on the input and LeakyReLU activations."""

    def __init__(self):
        super().__init__()
        self.blocks = [layers.Dense(W_DIM, kernel_initializer="he_normal") for _ in range(8)]

    def call(self, z):
        z = z * tf.math.rsqrt(tf.reduce_mean(tf.square(z), axis=1, keepdims=True) + 1e-8)
        for dense in self.blocks:
            z = tf.nn.leaky_relu(dense(z), alpha=0.2)
        return z

"""Adaptive Instance Normalization (AdaIN)."""
import tensorflow as tf
from tensorflow.keras import layers


class AdaIN(layers.Layer):
    """Normalizes each feature map, then applies a style-dependent scale and bias."""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = layers.LayerNormalization(axis=[1, 2], center=False, scale=False)
        self.affine = layers.Dense(2 * channels, kernel_initializer="zeros", bias_initializer="zeros")

    def call(self, x, w):
        scale, bias = tf.split(self.affine(w), 2, axis=-1)
        return self.norm(x) * (1.0 + scale[:, None, None, :]) + bias[:, None, None, :]

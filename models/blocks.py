"""Building block of the synthesis network: conv + noise + AdaIN."""
import tensorflow as tf
from tensorflow.keras import layers

from models.adain import AdaIN


class StyledConv(layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.conv = layers.Conv2D(channels, 3, padding="same")
        self.adain = AdaIN(channels)

    def build(self, shapes):
        self.noise_strength = self.add_weight(
            name="noise_strength", shape=(), initializer="zeros", trainable=True
        )

    def call(self, x, w, noise=None):
        x = self.conv(x)
        if noise is None:
            noise = tf.random.normal([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 1])
        x = x + self.noise_strength * noise
        return tf.nn.leaky_relu(self.adain(x, w), alpha=0.2)

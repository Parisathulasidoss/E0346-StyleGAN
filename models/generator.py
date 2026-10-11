"""Style-based generator: learned constant -> styled convs -> RGB."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import CHANNELS, NUM_STYLES
from models.blocks import StyledConv
from models.mapping import MappingNetwork


class StyleGenerator(keras.Model):
    def __init__(self):
        super().__init__()
        self.mapping = MappingNetwork()
        self.constant = self.add_weight(
            name="learned_constant", shape=(1, 4, 4, CHANNELS[4]),
            initializer="random_normal", trainable=True,
        )
        self.convs = [StyledConv(CHANNELS[r]) for r in (4, 4, 8, 8, 16, 16, 32, 32)]
        self.to_rgb = layers.Conv2D(3, 1, padding="same", activation="tanh")

    def call(self, inputs, training=False, return_styles=False):
        z, styles = inputs if isinstance(inputs, (tuple, list)) else (inputs, None)
        if styles is None:
            styles = tf.repeat(self.mapping(z)[:, None, :], NUM_STYLES, axis=1)
        x = tf.repeat(self.constant, tf.shape(z)[0], axis=0)
        for i, conv in enumerate(self.convs):
            if i in (2, 4, 6):
                x = tf.image.resize(x, [tf.shape(x)[1] * 2, tf.shape(x)[2] * 2], method="nearest")
            x = conv(x, styles[:, i, :])
        image = self.to_rgb(x)
        return (image, styles) if return_styles else image

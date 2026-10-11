"""Convolutional discriminator (32x32 -> logit)."""
from tensorflow import keras
from tensorflow.keras import layers


class Discriminator(keras.Model):
    def __init__(self):
        super().__init__()
        self.blocks = []
        for c in (32, 64, 128, 128):
            self.blocks.append(keras.Sequential([
                layers.Conv2D(c, 4, strides=2, padding="same"),
                layers.LeakyReLU(negative_slope=0.2),
            ]))
        self.head = keras.Sequential([layers.Flatten(), layers.Dense(1)])

    def call(self, x, training=False):
        for block in self.blocks:
            x = block(x, training=training)
        return self.head(x)

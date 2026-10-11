"""Train the StyleGAN-style model on CIFAR-10.

Usage:
    python train.py                       # defaults from config.py
    python train.py --epochs 30 --limit 50000
    python train.py --synthetic --epochs 1 --limit 256   # quick smoke test, no download
"""
import argparse
import os
import random
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from tensorflow import keras

import config as C
from data import load_cifar10, make_dataset
from models import Discriminator, StyleGenerator
from utils import plot_grid, plot_interpolation, plot_losses, plot_style_mixing


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=C.EPOCHS)
    p.add_argument("--limit", type=int, default=C.TRAIN_LIMIT, help="number of training images")
    p.add_argument("--batch", type=int, default=C.BATCH_SIZE)
    p.add_argument("--synthetic", action="store_true", help="random data instead of CIFAR-10 (smoke test)")
    p.add_argument("--ckpt_dir", default=C.CHECKPOINT_DIR)
    p.add_argument("--out_dir", default=C.OUTPUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(C.SEED); np.random.seed(C.SEED); tf.random.set_seed(C.SEED)
    print("TensorFlow:", tf.__version__, "| devices:", [d.device_type for d in tf.config.list_physical_devices()])

    images = load_cifar10(args.limit, synthetic=args.synthetic)
    dataset = make_dataset(images, args.batch)
    print(f"Training images: {len(images):,} | shape: {images.shape[1:]}")

    G, D = StyleGenerator(), Discriminator()
    _ = G(tf.zeros([1, C.Z_DIM])); _ = D(tf.zeros([1, 32, 32, 3]))
    print(f"Generator parameters: {G.count_params():,} | discriminator: {D.count_params():,}")

    g_opt = keras.optimizers.Adam(C.LEARNING_RATE, beta_1=C.BETA_1, beta_2=C.BETA_2)
    d_opt = keras.optimizers.Adam(C.LEARNING_RATE, beta_1=C.BETA_1, beta_2=C.BETA_2)

    @tf.function
    def train_step(real):
        z = tf.random.normal([tf.shape(real)[0], C.Z_DIM])
        with tf.GradientTape() as dt:
            fake = tf.stop_gradient(G(z, training=True))
            d_loss = (tf.reduce_mean(tf.nn.softplus(-D(real, training=True)))
                      + tf.reduce_mean(tf.nn.softplus(D(fake, training=True))))
        d_opt.apply_gradients(zip(dt.gradient(d_loss, D.trainable_variables), D.trainable_variables))
        with tf.GradientTape() as gt:
            g_loss = tf.reduce_mean(tf.nn.softplus(-D(G(z, training=True), training=True)))
        g_opt.apply_gradients(zip(gt.gradient(g_loss, G.trainable_variables), G.trainable_variables))
        return d_loss, g_loss

    g_hist, d_hist, start = [], [], time.time()
    for epoch in range(1, args.epochs + 1):
        d_sum = g_sum = n = 0.0
        for real in dataset:
            dl, gl = train_step(real)
            d_sum += float(dl); g_sum += float(gl); n += 1
        d_hist.append(d_sum / n); g_hist.append(g_sum / n)
        print(f"Epoch {epoch:02d}/{args.epochs} | D {d_hist[-1]:.3f} | G {g_hist[-1]:.3f} | {time.time() - start:.0f}s")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    G.save_weights(os.path.join(args.ckpt_dir, "generator.weights.h5"))
    D.save_weights(os.path.join(args.ckpt_dir, "discriminator.weights.h5"))
    print(f"Saved weights to {args.ckpt_dir}/")

    fixed_z = tf.random.normal([64, C.Z_DIM], seed=C.SEED)
    plot_losses(g_hist, d_hist, os.path.join(args.out_dir, "training_losses.png"))
    plot_grid(G(fixed_z, training=False).numpy(), "Generated CIFAR-10 samples",
              os.path.join(args.out_dir, "generated_samples.png"))
    plot_style_mixing(G, os.path.join(args.out_dir, "style_mixing.png"))
    plot_interpolation(G, os.path.join(args.out_dir, "latent_interpolation.png"))


if __name__ == "__main__":
    main()

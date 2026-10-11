"""Plot helpers: sample grids, loss curves, style mixing, latent interpolation."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import NUM_STYLES, Z_DIM


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_losses(g_hist, d_hist, path):
    fig = plt.figure(figsize=(7, 3))
    plt.plot(g_hist, label="Generator")
    plt.plot(d_hist, label="Discriminator")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(alpha=0.25)
    plt.title("Training losses")
    _save(fig, path)


def plot_grid(images, title, path, nrow=8, figsize=(8, 8)):
    images = (np.asarray(images) + 1) / 2
    fig, axes = plt.subplots(int(np.ceil(len(images) / nrow)), nrow, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    for ax, im in zip(axes, images):
        ax.imshow(np.clip(im, 0, 1)); ax.axis("off")
    for ax in axes[len(images):]:
        ax.axis("off")
    fig.suptitle(title); plt.tight_layout()
    _save(fig, path)


def plot_style_mixing(G, path, cutoffs=(2, 4, 6)):
    z_a, z_b = tf.random.normal([1, Z_DIM]), tf.random.normal([1, Z_DIM])
    w_a, w_b = G.mapping(z_a), G.mapping(z_b)
    images, labels = [G(z_a), G(z_b)], ["Source A", "Source B"]
    for cutoff in cutoffs:
        ws = tf.concat([tf.repeat(w_a[:, None, :], cutoff, axis=1),
                        tf.repeat(w_b[:, None, :], NUM_STYLES - cutoff, axis=1)], axis=1)
        images.append(G((z_a, ws))); labels.append(f"Mix at layer {cutoff}")
    fig, axes = plt.subplots(1, len(images), figsize=(12, 3))
    for ax, im, label in zip(axes, images, labels):
        ax.imshow((im[0].numpy() + 1) / 2); ax.set_title(label, fontsize=9); ax.axis("off")
    fig.suptitle("Style mixing: A supplies early styles, B supplies later styles")
    plt.tight_layout()
    _save(fig, path)


def plot_interpolation(G, path, steps=9):
    z1, z2 = tf.random.normal([1, Z_DIM]), tf.random.normal([1, Z_DIM])
    t = tf.linspace(0.0, 1.0, steps)[:, None]
    path_imgs = G((1 - t) * z1 + t * z2, training=False)
    fig, axes = plt.subplots(1, steps, figsize=(14, 2.2))
    for i, (ax, im) in enumerate(zip(axes, path_imgs)):
        ax.imshow((im.numpy() + 1) / 2); ax.set_title(f"{i / (steps - 1):.2f}", fontsize=9); ax.axis("off")
    fig.suptitle("Latent interpolation from z1 to z2")
    plt.tight_layout()
    _save(fig, path)

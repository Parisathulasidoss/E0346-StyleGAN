"""Generate samples, style-mixing and interpolation figures from saved weights.

Usage:
    python generate.py --checkpoint checkpoints/generator.weights.h5 --out_dir outputs
"""
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

import config as C
from models import StyleGenerator
from utils import plot_grid, plot_interpolation, plot_style_mixing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=os.path.join(C.CHECKPOINT_DIR, "generator.weights.h5"))
    p.add_argument("--out_dir", default=C.OUTPUT_DIR)
    p.add_argument("--seed", type=int, default=C.SEED)
    args = p.parse_args()

    tf.random.set_seed(args.seed)
    G = StyleGenerator()
    _ = G(tf.zeros([1, C.Z_DIM]))  # build variables before loading
    G.load_weights(args.checkpoint)

    z = tf.random.normal([64, C.Z_DIM], seed=args.seed)
    plot_grid(G(z, training=False).numpy(), "Generated CIFAR-10 samples",
              os.path.join(args.out_dir, "generated_samples.png"))
    plot_style_mixing(G, os.path.join(args.out_dir, "style_mixing.png"))
    plot_interpolation(G, os.path.join(args.out_dir, "latent_interpolation.png"))


if __name__ == "__main__":
    main()

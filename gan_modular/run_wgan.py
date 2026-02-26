import argparse
import os

import numpy as np

from config import WGANConfig, X_COLS, Y_COLS
from data import load_and_preprocess, sample_generated_dataset, save_generated_outputs, save_scalers
from utils import set_seed
from wgan_gp import build_wgan_critic, build_wgan_generator, train_wgan_gp


def parse_args():
    parser = argparse.ArgumentParser(description="Train modular WGAN-GP and generate datasets.")
    parser.add_argument("--file-path", default="corrected_channel_dataset.xlsx")
    parser.add_argument("--model-dir", default="wgan_models")
    parser.add_argument("--output-dir", default="generated_datasets_wgan")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--critic-steps", type=int, default=5)
    parser.add_argument("--gp-lambda", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-freq", type=int, default=25)
    parser.add_argument("--num-samples", type=int, default=20000)
    parser.add_argument("--gen-batch-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = WGANConfig(
        file_path=args.file_path,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        critic_steps=args.critic_steps,
        gp_lambda=args.gp_lambda,
        lr=args.lr,
        log_freq=args.log_freq,
        num_samples=args.num_samples,
        gen_batch_size=args.gen_batch_size,
        seed=args.seed,
    )
    set_seed(cfg.seed)

    bundle = load_and_preprocess(cfg.file_path, X_COLS, Y_COLS, random_state=cfg.seed)
    x_train_s = bundle["x_train_s"]
    y_train_s = bundle["y_train_s"]
    x_test_s = bundle["x_test_s"]
    y_test_s = bundle["y_test_s"]
    y_train = bundle["y_train"]
    y_test = bundle["y_test"]

    input_dim = x_train_s.shape[1]
    output_dim = y_train_s.shape[1]

    generator = build_wgan_generator(cfg.latent_dim, input_dim, output_dim)
    critic = build_wgan_critic(input_dim, output_dim)

    print(generator.summary())
    print(critic.summary())

    train_wgan_gp(
        generator=generator,
        critic=critic,
        x_train_s=x_train_s,
        y_train_s=y_train_s,
        x_test_s=x_test_s,
        y_test_s=y_test_s,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        latent_dim=cfg.latent_dim,
        critic_steps=cfg.critic_steps,
        gp_lambda=cfg.gp_lambda,
        lr=cfg.lr,
        log_freq=cfg.log_freq,
    )

    os.makedirs(cfg.model_dir, exist_ok=True)
    generator.save(os.path.join(cfg.model_dir, "generator_final.h5"))
    critic.save(os.path.join(cfg.model_dir, "critic_final.h5"))
    save_scalers(cfg.model_dir, bundle["scaler_x"], bundle["scaler_y"])

    generated_data = sample_generated_dataset(
        generator_predict_fn=lambda noise, cond: generator([noise, cond], training=False).numpy(),
        scaler_y=bundle["scaler_y"],
        x_train_s=x_train_s,
        latent_dim=cfg.latent_dim,
        num_samples=cfg.num_samples,
        gen_batch_size=cfg.gen_batch_size,
    )

    real_full = np.vstack([y_train, y_test]).astype(np.float32)
    out_paths = save_generated_outputs(cfg.output_dir, generated_data, real_full, Y_COLS)
    print("Saved outputs:")
    for p in out_paths:
        print(p)


if __name__ == "__main__":
    main()

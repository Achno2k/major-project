import argparse
import os

import numpy as np

from config import CGANConfig, X_COLS, Y_COLS
from cgan import build_cgan_discriminator, build_cgan_generator, train_cgan
from data import load_and_preprocess, sample_generated_dataset, save_generated_outputs, save_scalers
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train modular CGAN and generate datasets.")
    parser.add_argument("--file-path", default="corrected_channel_dataset.xlsx")
    parser.add_argument("--model-dir", default="cgan_models")
    parser.add_argument("--output-dir", default="generated_datasets_cgan")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-freq", type=int, default=25)
    parser.add_argument("--num-samples", type=int, default=20000)
    parser.add_argument("--gen-batch-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta-1", type=float, default=0.5)
    parser.add_argument("--label-smooth-real", type=float, default=0.9)
    parser.add_argument("--label-smooth-fake", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = CGANConfig(
        file_path=args.file_path,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        lr=args.lr,
        log_freq=args.log_freq,
        num_samples=args.num_samples,
        gen_batch_size=args.gen_batch_size,
        seed=args.seed,
        beta_1=args.beta_1,
        label_smooth_real=args.label_smooth_real,
        label_smooth_fake=args.label_smooth_fake,
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

    generator = build_cgan_generator(cfg.latent_dim, input_dim, output_dim)
    discriminator = build_cgan_discriminator(input_dim, output_dim)

    print(generator.summary())
    print(discriminator.summary())

    train_cgan(
        generator=generator,
        discriminator=discriminator,
        x_train_s=x_train_s,
        y_train_s=y_train_s,
        x_test_s=x_test_s,
        y_test_s=y_test_s,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        latent_dim=cfg.latent_dim,
        lr=cfg.lr,
        beta_1=cfg.beta_1,
        label_smooth_real=cfg.label_smooth_real,
        label_smooth_fake=cfg.label_smooth_fake,
        log_freq=cfg.log_freq,
    )

    os.makedirs(cfg.model_dir, exist_ok=True)
    generator.save(os.path.join(cfg.model_dir, "generator_final.h5"))
    discriminator.save(os.path.join(cfg.model_dir, "discriminator_final.h5"))
    save_scalers(cfg.model_dir, bundle["scaler_x"], bundle["scaler_y"])

    generated_data = sample_generated_dataset(
        generator_predict_fn=lambda noise, cond: generator.predict([noise, cond], verbose=0),
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

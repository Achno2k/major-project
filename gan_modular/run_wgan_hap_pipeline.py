import argparse
import os

from config import WGANConfig
from data import load_and_preprocess, sample_generated_dataset, save_generated_outputs, save_scalers
from run_hap_analysis import run_selection_ec, split_parts_1d
from utils import set_seed
from wgan_gp import build_wgan_critic, build_wgan_generator, train_wgan_gp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import X_COLS, Y_COLS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train WGAN-GP, generate synthetic HAP channel data, and save the same HAP plots as CGAN."
    )
    parser.add_argument("--file-path", default="../corrected_channel_dataset.xlsx")
    parser.add_argument("--model-dir", default="../generated_models_wgan")
    parser.add_argument("--output-dir", default="../generated_datasets_wgan")
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
    parser.add_argument("--n-haps", type=int, default=4)
    parser.add_argument("--n-trials", type=int, default=50000)
    parser.add_argument("--pt-min", type=float, default=0.1)
    parser.add_argument("--pt-max", type=float, default=2.0)
    parser.add_argument("--pt-points", type=int, default=10)
    parser.add_argument("--path-losses", default="0.20,0.12,0.09,0.05")
    parser.add_argument("--snr-threshold", type=float, default=5.0)
    parser.add_argument(
        "--rf-attenuation-shift-db",
        type=float,
        default=-20.0,
        help=(
            "Post-generation RF attenuation calibration in dB. Negative values model "
            "lower-loss WGAN channels for HAP selection."
        ),
    )
    return parser.parse_args()


def calibrate_wgan_outputs(generated_data: np.ndarray, rf_shift_db: float) -> np.ndarray:
    calibrated = generated_data.copy()
    calibrated[:, -1] = calibrated[:, -1] + rf_shift_db
    return calibrated


def save_hap_outputs(args, generated_data, real_data):
    os.makedirs(args.output_dir, exist_ok=True)

    real_rf_db = real_data[:, -1].astype(np.float64).ravel()
    wgan_rf_db = generated_data[:, -1].astype(np.float64).ravel()
    real_parts = split_parts_1d(real_rf_db, k=args.n_haps)
    wgan_parts = split_parts_1d(wgan_rf_db, k=args.n_haps)

    path_losses = np.array([float(x.strip()) for x in args.path_losses.split(",")], dtype=np.float64)
    if path_losses.size < args.n_haps:
        raise ValueError(f"Need at least {args.n_haps} path loss values, got {path_losses.size}")
    path_losses = path_losses[: args.n_haps]

    k_b = 1.38064852e-23
    temperature = 290.0
    bandwidth = 20e6
    noise_power = k_b * temperature * bandwidth

    pt_values = np.linspace(args.pt_min, args.pt_max, args.pt_points)
    rng = np.random.default_rng(args.seed)

    out_real, ec_real = run_selection_ec(
        real_parts,
        pt_values,
        path_losses,
        args.n_trials,
        noise_power,
        args.snr_threshold,
        rng,
    )
    out_wgan, ec_wgan = run_selection_ec(
        wgan_parts,
        pt_values,
        path_losses,
        args.n_trials,
        noise_power,
        args.snr_threshold,
        rng,
    )

    metrics = pd.DataFrame(
        {
            "Pt_W": pt_values,
            "Outage_Real": out_real,
            "Outage_GAN": out_wgan,
            "EC_Real": ec_real,
            "EC_GAN": ec_wgan,
        }
    )
    metrics_csv = os.path.join(args.output_dir, "hap_selection_metrics.csv")
    metrics.to_csv(metrics_csv, index=False)

    outage_plot = os.path.join(args.output_dir, "hap_outage_probability_real_vs_gan.png")
    plt.figure(figsize=(8, 5))
    plt.semilogy(pt_values, out_real, "o-", label="Real (selection)")
    plt.semilogy(pt_values, out_wgan, "s-", label="WGAN (selection)")
    plt.xlabel("Transmit Power (W)")
    plt.ylabel("Outage Probability")
    plt.title("Outage Probability (Real vs WGAN) - using RF loss column only")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outage_plot, dpi=200)
    plt.close()

    ec_plot = os.path.join(args.output_dir, "hap_ergodic_capacity_real_vs_gan.png")
    plt.figure(figsize=(8, 5))
    plt.plot(pt_values, ec_real, "o-", label="Real (selection)")
    plt.plot(pt_values, ec_wgan, "s-", label="WGAN (selection)")
    plt.xlabel("Transmit Power (W)")
    plt.ylabel("Ergodic Capacity (bits/s/Hz)")
    plt.title("Ergodic Capacity (Real vs WGAN) - using RF loss column only")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ec_plot, dpi=200)
    plt.close()

    print("Saved WGAN HAP outputs:")
    print(metrics_csv)
    print(outage_plot)
    print(ec_plot)


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

    generator = build_wgan_generator(cfg.latent_dim, x_train_s.shape[1], y_train_s.shape[1])
    critic = build_wgan_critic(x_train_s.shape[1], y_train_s.shape[1])

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
    generated_data = calibrate_wgan_outputs(generated_data, args.rf_attenuation_shift_db)
    real_full = np.vstack([y_train, y_test]).astype(np.float32)
    print(f"Applied WGAN RF attenuation calibration: {args.rf_attenuation_shift_db:+.2f} dB")
    for path in save_generated_outputs(cfg.output_dir, generated_data, real_full, Y_COLS):
        print(path)

    save_hap_outputs(args, generated_data, real_full)


if __name__ == "__main__":
    main()

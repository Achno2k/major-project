import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HAP selection analysis on generated GAN/WGAN datasets."
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Directory containing gan_dataset.csv and real_dataset.csv",
    )
    parser.add_argument("--gan-file", default="gan_dataset.csv")
    parser.add_argument("--real-file", default="real_dataset.csv")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for plots/metrics (default: dataset-dir)",
    )
    parser.add_argument(
        "--synthetic-label",
        default="GAN",
        help="Label to use for the generated dataset in plots and logs.",
    )
    parser.add_argument("--rf-col", type=int, default=-1, help="RF attenuation column index")
    parser.add_argument("--n-haps", type=int, default=4)
    parser.add_argument("--n-trials", type=int, default=50000)
    parser.add_argument("--pt-min", type=float, default=0.1)
    parser.add_argument("--pt-max", type=float, default=2.0)
    parser.add_argument("--pt-points", type=int, default=10)
    parser.add_argument(
        "--path-losses",
        type=str,
        default="0.20,0.12,0.09,0.05",
        help="Comma-separated per-HAP linear path gains",
    )
    parser.add_argument("--snr-threshold", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def split_parts_1d(array_1d: np.ndarray, k: int):
    n = array_1d.size
    sizes = [(n // k) + (1 if i < n % k else 0) for i in range(k)]
    parts = []
    idx = 0
    for s in sizes:
        parts.append(array_1d[idx : idx + s])
        idx += s
    return parts


def run_selection_ec(parts_list, pt_values, path_losses, n_trials, noise_power, snr_threshold, rng):
    n_haps = len(parts_list)
    outage = np.zeros(len(pt_values))
    erg_cap = np.zeros(len(pt_values))

    for pi, pt in enumerate(pt_values):
        idxs = [rng.integers(0, parts_list[h].size, size=n_trials) for h in range(n_haps)]
        loss_db_mat = np.column_stack([parts_list[h][idxs[h]] for h in range(n_haps)])

        loss_db_mat_clipped = np.clip(loss_db_mat, -500.0, 500.0)
        gain_lin = 10.0 ** (-loss_db_mat_clipped / 10.0)
        gain_lin = gain_lin * path_losses[np.newaxis, :]

        pr = pt * gain_lin
        snrs = pr / (noise_power + 1e-300)
        snr_sel = np.max(snrs, axis=1)

        outage[pi] = np.mean(snr_sel < snr_threshold)
        erg_cap[pi] = np.mean(np.log2(1.0 + snr_sel))

    return outage, erg_cap


def main():
    args = parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else dataset_dir
    os.makedirs(out_dir, exist_ok=True)

    gan_path = os.path.join(dataset_dir, args.gan_file)
    real_path = os.path.join(dataset_dir, args.real_file)
    if not os.path.exists(gan_path):
        raise FileNotFoundError(f"GAN file not found: {gan_path}")
    if not os.path.exists(real_path):
        raise FileNotFoundError(f"Real file not found: {real_path}")

    gan = pd.read_csv(gan_path).to_numpy(dtype=np.float64)
    real = pd.read_csv(real_path).to_numpy(dtype=np.float64)
    print(f"Loaded shapes: {args.synthetic_label}", gan.shape, "REAL", real.shape)

    real_rf_db = real[:, args.rf_col].astype(np.float64).ravel()
    gan_rf_db = gan[:, args.rf_col].astype(np.float64).ravel()
    print(f"Means dB -> real: {real_rf_db.mean()} {args.synthetic_label}: {gan_rf_db.mean()}")

    real_parts = split_parts_1d(real_rf_db, k=args.n_haps)
    gan_parts = split_parts_1d(gan_rf_db, k=args.n_haps)

    path_losses = np.array([float(x.strip()) for x in args.path_losses.split(",")], dtype=np.float64)
    if path_losses.size < args.n_haps:
        raise ValueError(
            f"Need at least {args.n_haps} path loss values, got {path_losses.size}"
        )
    path_losses = path_losses[: args.n_haps]
    print("Using path_losses:", path_losses)

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
    out_gan, ec_gan = run_selection_ec(
        gan_parts,
        pt_values,
        path_losses,
        args.n_trials,
        noise_power,
        args.snr_threshold,
        rng,
    )

    sel = len(pt_values) // 2
    print("\nSummary at Pt =", pt_values[sel], "W")
    print("Outage real:", out_real[sel], "gan:", out_gan[sel])
    print("Ergodic capacity real:", ec_real[sel], "gan:", ec_gan[sel])
    print(
        "Avg EC improvement (gan vs real) in %:",
        100.0 * (np.mean(ec_gan) - np.mean(ec_real)) / (np.mean(ec_real) + 1e-12),
    )

    metrics = pd.DataFrame(
        {
            "Pt_W": pt_values,
            "Outage_Real": out_real,
            "Outage_GAN": out_gan,
            "EC_Real": ec_real,
            "EC_GAN": ec_gan,
        }
    )
    metrics_csv = os.path.join(out_dir, "hap_selection_metrics.csv")
    metrics.to_csv(metrics_csv, index=False)

    plt.figure(figsize=(8, 5))
    plt.semilogy(pt_values, out_real, "o-", label="Real (selection)")
    plt.semilogy(pt_values, out_gan, "s-", label=f"{args.synthetic_label} (selection)")
    plt.xlabel("Transmit Power (W)")
    plt.ylabel("Outage Probability")
    plt.title(f"Outage Probability (Real vs {args.synthetic_label}) - using RF loss column only")
    plt.grid(True)
    plt.legend()
    outage_plot = os.path.join(out_dir, "hap_outage_probability_real_vs_gan.png")
    plt.tight_layout()
    plt.savefig(outage_plot, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(pt_values, ec_real, "o-", label="Real (selection)")
    plt.plot(pt_values, ec_gan, "s-", label=f"{args.synthetic_label} (selection)")
    plt.xlabel("Transmit Power (W)")
    plt.ylabel("Ergodic Capacity (bits/s/Hz)")
    plt.title(f"Ergodic Capacity (Real vs {args.synthetic_label}) - using RF loss column only")
    plt.grid(True)
    plt.legend()
    ec_plot = os.path.join(out_dir, "hap_ergodic_capacity_real_vs_gan.png")
    plt.tight_layout()
    plt.savefig(ec_plot, dpi=200)
    plt.close()

    print("\nSaved outputs:")
    print(metrics_csv)
    print(outage_plot)
    print(ec_plot)


if __name__ == "__main__":
    main()

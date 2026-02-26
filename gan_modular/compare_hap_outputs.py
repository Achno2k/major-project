import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare HAP-analysis outputs from CGAN and WGAN runs."
    )
    parser.add_argument(
        "--cgan-dir",
        default="../generated_datasets_cgan",
        help="Directory containing CGAN hap_selection_metrics.csv",
    )
    parser.add_argument(
        "--wgan-dir",
        default="../generated_datasets_wgan",
        help="Directory containing WGAN hap_selection_metrics.csv",
    )
    parser.add_argument(
        "--metrics-file",
        default="hap_selection_metrics.csv",
        help="Metrics CSV filename produced by run_hap_analysis.py",
    )
    parser.add_argument(
        "--out-dir",
        default="../generated_datasets_comparison",
        help="Directory to save comparison plots and merged metrics",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cgan_dir = os.path.abspath(args.cgan_dir)
    wgan_dir = os.path.abspath(args.wgan_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cgan_csv = os.path.join(cgan_dir, args.metrics_file)
    wgan_csv = os.path.join(wgan_dir, args.metrics_file)
    if not os.path.exists(cgan_csv):
        raise FileNotFoundError(f"CGAN metrics file not found: {cgan_csv}")
    if not os.path.exists(wgan_csv):
        raise FileNotFoundError(f"WGAN metrics file not found: {wgan_csv}")

    cgan_df = pd.read_csv(cgan_csv)
    wgan_df = pd.read_csv(wgan_csv)

    required_cols = {"Pt_W", "Outage_Real", "Outage_GAN", "EC_Real", "EC_GAN"}
    if not required_cols.issubset(cgan_df.columns):
        raise ValueError(f"CGAN metrics missing required columns: {required_cols}")
    if not required_cols.issubset(wgan_df.columns):
        raise ValueError(f"WGAN metrics missing required columns: {required_cols}")

    merged = cgan_df[["Pt_W", "Outage_Real", "EC_Real"]].copy()
    merged = merged.rename(
        columns={"Outage_Real": "Outage_Real_CGAN", "EC_Real": "EC_Real_CGAN"}
    )
    merged["Outage_GAN_CGAN"] = cgan_df["Outage_GAN"].values
    merged["EC_GAN_CGAN"] = cgan_df["EC_GAN"].values
    merged["Outage_Real_WGAN"] = wgan_df["Outage_Real"].values
    merged["EC_Real_WGAN"] = wgan_df["EC_Real"].values
    merged["Outage_GAN_WGAN"] = wgan_df["Outage_GAN"].values
    merged["EC_GAN_WGAN"] = wgan_df["EC_GAN"].values

    merged_csv = os.path.join(out_dir, "hap_metrics_cgan_vs_wgan.csv")
    merged.to_csv(merged_csv, index=False)

    pt = cgan_df["Pt_W"].values

    plt.figure(figsize=(9, 5))
    plt.semilogy(pt, cgan_df["Outage_GAN"].values, "o-", label="CGAN")
    plt.semilogy(pt, wgan_df["Outage_GAN"].values, "s-", label="WGAN")
    plt.semilogy(pt, cgan_df["Outage_Real"].values, "k--", label="Real baseline")
    plt.xlabel("Transmit Power (W)")
    plt.ylabel("Outage Probability")
    plt.title("HAP Outage Comparison (CGAN vs WGAN)")
    plt.grid(True)
    plt.legend()
    outage_plot = os.path.join(out_dir, "compare_hap_outage_cgan_vs_wgan.png")
    plt.tight_layout()
    plt.savefig(outage_plot, dpi=220)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(pt, cgan_df["EC_GAN"].values, "o-", label="CGAN")
    plt.plot(pt, wgan_df["EC_GAN"].values, "s-", label="WGAN")
    plt.plot(pt, cgan_df["EC_Real"].values, "k--", label="Real baseline")
    plt.xlabel("Transmit Power (W)")
    plt.ylabel("Ergodic Capacity (bits/s/Hz)")
    plt.title("HAP Ergodic Capacity Comparison (CGAN vs WGAN)")
    plt.grid(True)
    plt.legend()
    ec_plot = os.path.join(out_dir, "compare_hap_ec_cgan_vs_wgan.png")
    plt.tight_layout()
    plt.savefig(ec_plot, dpi=220)
    plt.close()

    cgan_ec_mean = cgan_df["EC_GAN"].mean()
    wgan_ec_mean = wgan_df["EC_GAN"].mean()
    cgan_out_mean = cgan_df["Outage_GAN"].mean()
    wgan_out_mean = wgan_df["Outage_GAN"].mean()
    print("Comparison summary:")
    print(f"Average EC - CGAN: {cgan_ec_mean:.6f}, WGAN: {wgan_ec_mean:.6f}")
    print(f"Average Outage - CGAN: {cgan_out_mean:.6f}, WGAN: {wgan_out_mean:.6f}")
    print("\nSaved outputs:")
    print(merged_csv)
    print(outage_plot)
    print(ec_plot)


if __name__ == "__main__":
    main()


import argparse
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from tensorflow.keras.models import load_model

from config import X_COLS, Y_COLS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SHAP ExAI analysis for CGAN/WGAN generators."
    )
    parser.add_argument("--file-path", default="../corrected_channel_dataset.xlsx")
    parser.add_argument("--model", choices=["cgan", "wgan", "both"], default="both")
    parser.add_argument("--cgan-model-dir", default="../.run_artifacts/cgan_models")
    parser.add_argument("--wgan-model-dir", default="../.run_artifacts/wgan_models")
    parser.add_argument("--out-dir", default="../.run_artifacts/shap_exai")
    parser.add_argument("--background-size", type=int, default=20)
    parser.add_argument("--eval-size", type=int, default=30)
    parser.add_argument("--noise-samples", type=int, default=8)
    parser.add_argument("--nsamples", type=int, default=120)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--wgan-rf-attenuation-shift-db",
        type=float,
        default=-20.0,
        help="Same post-generation RF calibration used by the WGAN HAP pipeline.",
    )
    return parser.parse_args()


def load_scaler(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def model_dirs(args):
    if args.model == "cgan":
        return [("cgan", args.cgan_model_dir)]
    if args.model == "wgan":
        return [("wgan", args.wgan_model_dir)]
    return [("cgan", args.cgan_model_dir), ("wgan", args.wgan_model_dir)]


def write_architecture_summary(out_dir, model_name, generator, discriminator_or_critic_path):
    rows = [
        {
            "model": model_name,
            "component": "generator",
            "inputs": "noise vector + conditioning inputs",
            "input_shape": "; ".join(str(tuple(t.shape)) for t in generator.inputs),
            "output_shape": str(tuple(generator.output.shape)),
            "purpose": "Generates scaled channel attenuation outputs from latent noise and physical channel conditions.",
        }
    ]

    if os.path.exists(discriminator_or_critic_path):
        disc = load_model(discriminator_or_critic_path, compile=False)
        rows.append(
            {
                "model": model_name,
                "component": "discriminator" if model_name == "cgan" else "critic",
                "inputs": "channel outputs + conditioning inputs",
                "input_shape": "; ".join(str(tuple(t.shape)) for t in disc.inputs),
                "output_shape": str(tuple(disc.output.shape)),
                "purpose": (
                    "Classifies real/fake samples."
                    if model_name == "cgan"
                    else "Scores samples for Wasserstein distance with gradient penalty."
                ),
            }
        )

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{model_name}_architecture_summary.csv"), index=False)


def make_generator_predict_fn(
    generator,
    scaler_x,
    scaler_y,
    noises,
    model_name,
    wgan_rf_shift_db,
):
    def predict_from_original_x(x_original):
        x_original = np.asarray(x_original, dtype=np.float32)
        x_scaled = scaler_x.transform(x_original).astype(np.float32)
        preds = []
        for noise in noises:
            noise_batch = np.repeat(noise[np.newaxis, :], x_scaled.shape[0], axis=0)
            pred_scaled = generator.predict([noise_batch, x_scaled], verbose=0)
            pred = scaler_y.inverse_transform(pred_scaled)
            if model_name == "wgan":
                pred[:, -1] = pred[:, -1] + wgan_rf_shift_db
            preds.append(pred)
        return np.mean(preds, axis=0)

    return predict_from_original_x


def save_bar_plot(df, out_path, title):
    plot_df = df.sort_values("mean_abs_shap", ascending=True)
    plt.figure(figsize=(9, 5))
    plt.barh(plot_df["feature"], plot_df["mean_abs_shap"], color="#2f80ed")
    plt.xlabel("Mean |SHAP value|")
    plt.title(title)
    plt.grid(axis="x", color="#d0d0d0", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_heatmap(summary_df, out_path, title):
    pivot = summary_df.pivot(index="feature", columns="output", values="mean_abs_shap")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=True).index]
    plt.figure(figsize=(9, 5))
    im = plt.imshow(pivot.values, aspect="auto", cmap="viridis")
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    plt.colorbar(im, label="Mean |SHAP value|")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def run_for_model(args, model_name, model_dir, x_background, x_eval, noises):
    out_dir = os.path.join(args.out_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)

    generator_path = os.path.join(model_dir, "generator_final.h5")
    scaler_x_path = os.path.join(model_dir, "scaler_X.pkl")
    scaler_y_path = os.path.join(model_dir, "scaler_Y.pkl")
    if not os.path.exists(generator_path):
        raise FileNotFoundError(f"Generator not found: {generator_path}")
    if not os.path.exists(scaler_x_path) or not os.path.exists(scaler_y_path):
        raise FileNotFoundError(f"Scaler files not found in: {model_dir}")

    generator = load_model(generator_path, compile=False)
    scaler_x = load_scaler(scaler_x_path)
    scaler_y = load_scaler(scaler_y_path)
    disc_name = "discriminator_final.h5" if model_name == "cgan" else "critic_final.h5"
    write_architecture_summary(out_dir, model_name, generator, os.path.join(model_dir, disc_name))

    predict_all_outputs = make_generator_predict_fn(
        generator,
        scaler_x,
        scaler_y,
        noises,
        model_name,
        args.wgan_rf_attenuation_shift_db,
    )

    summary_rows = []
    for output_idx, output_name in enumerate(Y_COLS):
        def predict_one_output(x_original, idx=output_idx):
            return predict_all_outputs(x_original)[:, idx]

        explainer = shap.KernelExplainer(predict_one_output, x_background)
        shap_values = np.asarray(
            explainer.shap_values(x_eval, nsamples=args.nsamples),
            dtype=np.float64,
        )

        values_df = pd.DataFrame(shap_values, columns=X_COLS)
        values_path = os.path.join(out_dir, f"shap_values_{output_name}.csv")
        values_df.to_csv(values_path, index=False)

        output_summary = pd.DataFrame(
            {
                "model": model_name,
                "output": output_name,
                "feature": X_COLS,
                "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
                "mean_shap": np.mean(shap_values, axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        summary_rows.append(output_summary)

        save_bar_plot(
            output_summary,
            os.path.join(out_dir, f"shap_bar_{output_name}.png"),
            f"{model_name.upper()} SHAP Importance - {output_name}",
        )

        plt.figure(figsize=(9, 5))
        shap.summary_plot(
            shap_values,
            x_eval,
            feature_names=X_COLS,
            show=False,
            plot_size=None,
        )
        plt.title(f"{model_name.upper()} SHAP Dependency - {output_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"shap_summary_{output_name}.png"), dpi=220)
        plt.close()

    summary_df = pd.concat(summary_rows, ignore_index=True)
    summary_path = os.path.join(out_dir, "shap_feature_importance.csv")
    summary_df.to_csv(summary_path, index=False)
    save_heatmap(
        summary_df,
        os.path.join(out_dir, "shap_feature_importance_heatmap.png"),
        f"{model_name.upper()} SHAP Feature Importance Across Generated Outputs",
    )
    return summary_df


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    df = pd.read_excel(args.file_path)
    missing = [c for c in X_COLS + Y_COLS if c not in df.columns]
    if missing:
        raise ValueError("Missing expected columns in dataset: " + ", ".join(missing))

    x_original = df[X_COLS].to_numpy(dtype=np.float32)
    selected = rng.choice(x_original.shape[0], size=min(args.background_size + args.eval_size, x_original.shape[0]), replace=False)
    background_idx = selected[: args.background_size]
    eval_idx = selected[args.background_size : args.background_size + args.eval_size]
    x_background = x_original[background_idx]
    x_eval = x_original[eval_idx]
    noises = rng.normal(0, 1, size=(args.noise_samples, args.latent_dim)).astype(np.float32)

    pd.DataFrame(
        {
            "role": ["conditioning_input"] * len(X_COLS) + ["generated_output"] * len(Y_COLS),
            "column": X_COLS + Y_COLS,
        }
    ).to_csv(os.path.join(args.out_dir, "input_output_columns.csv"), index=False)

    all_summaries = []
    for model_name, model_dir in model_dirs(args):
        all_summaries.append(run_for_model(args, model_name, model_dir, x_background, x_eval, noises))

    combined = pd.concat(all_summaries, ignore_index=True)
    combined_path = os.path.join(args.out_dir, "shap_feature_importance_all_models.csv")
    combined.to_csv(combined_path, index=False)
    print("Saved SHAP ExAI outputs:")
    print(os.path.abspath(args.out_dir))
    print(os.path.abspath(combined_path))


if __name__ == "__main__":
    main()

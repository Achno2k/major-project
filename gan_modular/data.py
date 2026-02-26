import os
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess(
    file_path: str,
    x_cols,
    y_cols,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    df = pd.read_excel(file_path)
    missing = [c for c in x_cols + y_cols if c not in df.columns]
    if missing:
        raise ValueError("Missing expected columns in dataset: " + ", ".join(missing))

    x = df[x_cols].values.astype(np.float32)
    y = df[y_cols].values.astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_train_s = scaler_x.fit_transform(x_train).astype(np.float32)
    x_test_s = scaler_x.transform(x_test).astype(np.float32)
    y_train_s = scaler_y.fit_transform(y_train).astype(np.float32)
    y_test_s = scaler_y.transform(y_test).astype(np.float32)

    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "x_train_s": x_train_s,
        "x_test_s": x_test_s,
        "y_train_s": y_train_s,
        "y_test_s": y_test_s,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
    }


def save_scalers(model_dir: str, scaler_x: StandardScaler, scaler_y: StandardScaler):
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "scaler_X.pkl"), "wb") as f:
        pickle.dump(scaler_x, f)
    with open(os.path.join(model_dir, "scaler_Y.pkl"), "wb") as f:
        pickle.dump(scaler_y, f)


def save_generated_outputs(
    output_dir: str,
    generated_data: np.ndarray,
    real_data: np.ndarray,
    y_cols,
):
    os.makedirs(output_dir, exist_ok=True)
    gan_npy_path = os.path.join(output_dir, "gan_dataset.npy")
    gan_csv_path = os.path.join(output_dir, "gan_dataset.csv")
    real_npy_path = os.path.join(output_dir, "real_dataset.npy")
    real_csv_path = os.path.join(output_dir, "real_dataset.csv")

    np.save(gan_npy_path, generated_data)
    np.save(real_npy_path, real_data)
    pd.DataFrame(generated_data, columns=y_cols).to_csv(gan_csv_path, index=False)
    pd.DataFrame(real_data, columns=y_cols).to_csv(real_csv_path, index=False)

    return gan_npy_path, gan_csv_path, real_npy_path, real_csv_path


def sample_generated_dataset(
    generator_predict_fn,
    scaler_y: StandardScaler,
    x_train_s: np.ndarray,
    latent_dim: int,
    num_samples: int,
    gen_batch_size: int,
) -> np.ndarray:
    generated_parts = []
    full_batches = num_samples // gen_batch_size

    for _ in range(full_batches):
        noise = np.random.normal(0, 1, (gen_batch_size, latent_dim)).astype(np.float32)
        cond_idx = np.random.randint(0, x_train_s.shape[0], gen_batch_size)
        cond_input = x_train_s[cond_idx]
        gen_batch_s = generator_predict_fn(noise, cond_input)
        generated_parts.append(gen_batch_s)

    rem = num_samples - full_batches * gen_batch_size
    if rem > 0:
        noise = np.random.normal(0, 1, (rem, latent_dim)).astype(np.float32)
        cond_idx = np.random.randint(0, x_train_s.shape[0], rem)
        cond_input = x_train_s[cond_idx]
        gen_batch_s = generator_predict_fn(noise, cond_input)
        generated_parts.append(gen_batch_s)

    generated_scaled = np.vstack(generated_parts)
    return scaler_y.inverse_transform(generated_scaled).astype(np.float32)


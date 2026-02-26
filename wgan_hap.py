import argparse
import os
import pickle
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.optimizers import Adam


X_COLS = [
    "Visibility_km",
    "LWC_gm3",
    "CNC_cm3",
    "Temperature_C",
    "Frequency_GHz",
    "Path_km",
]

Y_COLS = [
    "FSO_Attenuation_Visibility_dB",
    "FSO_Attenuation_Cloud_Model_dB",
    "RF_Total_Attenuation_dB",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Conditional WGAN-GP for HAP channel data synthesis.")
    parser.add_argument("--file-path", default="corrected_channel_dataset.xlsx")
    parser.add_argument("--model-dir", default="wgan_models")
    parser.add_argument("--output-dir", default="generated_datasets")
    parser.add_argument("--epochs", type=int, default=5)
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_generator(latent_dim: int, input_dim: int, output_dim: int) -> Model:
    noise_input = Input(shape=(latent_dim,), name="noise_input")
    cond_input = Input(shape=(input_dim,), name="cond_input")
    x = layers.Concatenate()([noise_input, cond_input])
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    out = layers.Dense(output_dim, activation="linear")(x)
    return Model([noise_input, cond_input], out, name="wgan_generator")


def build_critic(input_dim: int, output_dim: int) -> Model:
    y_input = Input(shape=(output_dim,), name="y_input")
    cond_input = Input(shape=(input_dim,), name="cond_input")
    x = layers.Concatenate()([y_input, cond_input])
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="linear")(x)
    return Model([y_input, cond_input], out, name="wgan_discriminator")


def load_and_preprocess(file_path: str):
    print(f"Loading dataset from: {file_path}")
    df = pd.read_excel(file_path)

    missing = [c for c in X_COLS + Y_COLS if c not in df.columns]
    if missing:
        raise ValueError("Missing expected columns in dataset: " + ", ".join(missing))

    x = df[X_COLS].values.astype(np.float32)
    y = df[Y_COLS].values.astype(np.float32)
    print("Raw shapes -> X:", x.shape, "Y:", y.shape)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, shuffle=True
    )

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_train_s = scaler_x.fit_transform(x_train).astype(np.float32)
    x_test_s = scaler_x.transform(x_test).astype(np.float32)
    y_train_s = scaler_y.fit_transform(y_train).astype(np.float32)
    y_test_s = scaler_y.transform(y_test).astype(np.float32)

    print("Scaled shapes -> X_train_s:", x_train_s.shape, "Y_train_s:", y_train_s.shape)
    return x_train, x_test, y_train, y_test, x_train_s, x_test_s, y_train_s, y_test_s, scaler_x, scaler_y


def gradient_penalty(critic: Model, real_y: tf.Tensor, fake_y: tf.Tensor, cond_x: tf.Tensor) -> tf.Tensor:
    batch_size = tf.shape(real_y)[0]
    alpha = tf.random.uniform(shape=(batch_size, 1), minval=0.0, maxval=1.0)
    interpolated = alpha * real_y + (1.0 - alpha) * fake_y

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = critic([interpolated, cond_x], training=True)
    grads = tape.gradient(pred, interpolated)
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)
    return tf.reduce_mean(tf.square(grad_norm - 1.0))


def train_wgan_gp(
    generator: Model,
    critic: Model,
    x_train_s: np.ndarray,
    y_train_s: np.ndarray,
    x_test_s: np.ndarray,
    y_test_s: np.ndarray,
    epochs: int,
    batch_size: int,
    latent_dim: int,
    critic_steps: int,
    gp_lambda: float,
    lr: float,
    log_freq: int,
):
    n_train = x_train_s.shape[0]
    steps_per_epoch = max(1, n_train // batch_size)
    val_sample_size = min(256, x_test_s.shape[0])

    g_opt = Adam(learning_rate=lr, beta_1=0.0, beta_2=0.9)
    c_opt = Adam(learning_rate=lr, beta_1=0.0, beta_2=0.9)

    print(
        f"Training config -> epochs: {epochs}, batch_size: {batch_size}, "
        f"steps/epoch: {steps_per_epoch}, critic_steps: {critic_steps}, gp_lambda: {gp_lambda}"
    )

    for epoch in range(1, epochs + 1):
        critic_losses = []
        generator_losses = []
        gp_values = []

        for _ in range(steps_per_epoch):
            for _ in range(critic_steps):
                idx = np.random.randint(0, n_train, batch_size)
                cond_batch = tf.convert_to_tensor(x_train_s[idx], dtype=tf.float32)
                real_batch = tf.convert_to_tensor(y_train_s[idx], dtype=tf.float32)
                noise = tf.random.normal((batch_size, latent_dim))

                with tf.GradientTape() as tape:
                    fake_batch = generator([noise, cond_batch], training=True)
                    real_score = critic([real_batch, cond_batch], training=True)
                    fake_score = critic([fake_batch, cond_batch], training=True)
                    gp = gradient_penalty(critic, real_batch, fake_batch, cond_batch)
                    c_loss = (
                        tf.reduce_mean(fake_score)
                        - tf.reduce_mean(real_score)
                        + gp_lambda * gp
                    )

                c_grads = tape.gradient(c_loss, critic.trainable_variables)
                c_opt.apply_gradients(zip(c_grads, critic.trainable_variables))
                critic_losses.append(float(c_loss.numpy()))
                gp_values.append(float(gp.numpy()))

            idx = np.random.randint(0, n_train, batch_size)
            cond_batch = tf.convert_to_tensor(x_train_s[idx], dtype=tf.float32)
            noise = tf.random.normal((batch_size, latent_dim))

            with tf.GradientTape() as tape:
                fake_batch = generator([noise, cond_batch], training=True)
                fake_score = critic([fake_batch, cond_batch], training=True)
                g_loss = -tf.reduce_mean(fake_score)

            g_grads = tape.gradient(g_loss, generator.trainable_variables)
            g_opt.apply_gradients(zip(g_grads, generator.trainable_variables))
            generator_losses.append(float(g_loss.numpy()))

        val_idx = np.random.randint(0, x_test_s.shape[0], val_sample_size)
        val_x = x_test_s[val_idx]
        val_y = y_test_s[val_idx]
        val_noise = np.random.normal(0, 1, (val_x.shape[0], latent_dim)).astype(np.float32)
        val_pred = generator([val_noise, val_x], training=False).numpy()
        val_mse_scaled = float(np.mean(np.square(val_y - val_pred)))

        if epoch == 1 or epoch == epochs or (epoch % log_freq == 0):
            print(
                f"Epoch {epoch}/{epochs} | "
                f"critic_loss(avg)={np.mean(critic_losses):.4f} | "
                f"g_loss(avg)={np.mean(generator_losses):.4f} | "
                f"gp(avg)={np.mean(gp_values):.4f} | "
                f"val_mse(scaled)={val_mse_scaled:.6f}"
            )


def generate_dataset(
    generator: Model,
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
        gen_batch_s = generator([noise, cond_input], training=False).numpy()
        generated_parts.append(gen_batch_s)

    rem = num_samples - full_batches * gen_batch_size
    if rem > 0:
        noise = np.random.normal(0, 1, (rem, latent_dim)).astype(np.float32)
        cond_idx = np.random.randint(0, x_train_s.shape[0], rem)
        cond_input = x_train_s[cond_idx]
        gen_batch_s = generator([noise, cond_input], training=False).numpy()
        generated_parts.append(gen_batch_s)

    generated_scaled = np.vstack(generated_parts)
    print("Scaled generated dataset shape:", generated_scaled.shape)
    return scaler_y.inverse_transform(generated_scaled)


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    (
        x_train,
        x_test,
        y_train,
        y_test,
        x_train_s,
        x_test_s,
        y_train_s,
        y_test_s,
        scaler_x,
        scaler_y,
    ) = load_and_preprocess(args.file_path)

    input_dim = x_train_s.shape[1]
    output_dim = y_train_s.shape[1]
    generator = build_generator(args.latent_dim, input_dim, output_dim)
    critic = build_critic(input_dim, output_dim)

    print(generator.summary())
    print(critic.summary())

    train_wgan_gp(
        generator=generator,
        critic=critic,
        x_train_s=x_train_s,
        y_train_s=y_train_s,
        x_test_s=x_test_s,
        y_test_s=y_test_s,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        critic_steps=args.critic_steps,
        gp_lambda=args.gp_lambda,
        lr=args.lr,
        log_freq=args.log_freq,
    )

    generator_path = os.path.join(args.model_dir, "generator_final.h5")
    critic_path = os.path.join(args.model_dir, "critic_final.h5")
    generator.save(generator_path)
    critic.save(critic_path)

    with open(os.path.join(args.model_dir, "scaler_X.pkl"), "wb") as f:
        pickle.dump(scaler_x, f)
    with open(os.path.join(args.model_dir, "scaler_Y.pkl"), "wb") as f:
        pickle.dump(scaler_y, f)

    print("Saved generator, critic, and scalers to:", args.model_dir)

    generated_data = generate_dataset(
        generator=generator,
        scaler_y=scaler_y,
        x_train_s=x_train_s,
        latent_dim=args.latent_dim,
        num_samples=args.num_samples,
        gen_batch_size=args.gen_batch_size,
    )
    generated_data = np.asarray(generated_data, dtype=np.float32)

    gan_npy_path = os.path.join(args.output_dir, "gan_dataset.npy")
    gan_csv_path = os.path.join(args.output_dir, "gan_dataset.csv")
    real_npy_path = os.path.join(args.output_dir, "real_dataset.npy")
    real_csv_path = os.path.join(args.output_dir, "real_dataset.csv")

    np.save(gan_npy_path, generated_data)
    pd.DataFrame(generated_data, columns=Y_COLS).to_csv(gan_csv_path, index=False)

    real_full = np.vstack([y_train, y_test]).astype(np.float32)
    np.save(real_npy_path, real_full)
    pd.DataFrame(real_full, columns=Y_COLS).to_csv(real_csv_path, index=False)

    print("WGAN dataset saved:")
    print(gan_npy_path)
    print(gan_csv_path)
    print(real_npy_path)
    print(real_csv_path)


if __name__ == "__main__":
    main()

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam


def build_cgan_generator(latent_dim: int, input_dim: int, output_dim: int) -> Model:
    noise_input = Input(shape=(latent_dim,), name="noise_input")
    cond_input = Input(shape=(input_dim,), name="cond_input")
    x = layers.Concatenate()([noise_input, cond_input])
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    out = layers.Dense(output_dim, activation="linear")(x)
    return Model([noise_input, cond_input], out, name="cgan_generator")


def build_cgan_discriminator(input_dim: int, output_dim: int) -> Model:
    y_input = Input(shape=(output_dim,), name="y_input")
    cond_input = Input(shape=(input_dim,), name="cond_input")
    x = layers.Concatenate()([y_input, cond_input])
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return Model([y_input, cond_input], out, name="cgan_discriminator")


def train_cgan(
    generator: Model,
    discriminator: Model,
    x_train_s: np.ndarray,
    y_train_s: np.ndarray,
    x_test_s: np.ndarray,
    y_test_s: np.ndarray,
    epochs: int,
    batch_size: int,
    latent_dim: int,
    lr: float,
    beta_1: float,
    label_smooth_real: float,
    label_smooth_fake: float,
    log_freq: int,
):
    input_dim = x_train_s.shape[1]
    bce = BinaryCrossentropy()

    discriminator.trainable = True
    discriminator.compile(
        optimizer=Adam(learning_rate=lr, beta_1=beta_1),
        loss=bce,
        metrics=["accuracy"],
    )

    discriminator.trainable = False
    noise_in = Input(shape=(latent_dim,), name="cg_noise_in")
    cond_in = Input(shape=(input_dim,), name="cg_cond_in")
    gen_out = generator([noise_in, cond_in])
    disc_out = discriminator([gen_out, cond_in])
    combined = Model([noise_in, cond_in], disc_out, name="cgan_combined")
    combined.compile(optimizer=Adam(learning_rate=lr, beta_1=beta_1), loss=bce)
    discriminator.trainable = True

    steps_per_epoch = max(1, x_train_s.shape[0] // batch_size)
    val_sample_size = min(256, x_test_s.shape[0])

    for epoch in range(1, epochs + 1):
        d_losses = []
        g_losses = []
        for _ in range(steps_per_epoch):
            idx = np.random.randint(0, x_train_s.shape[0], batch_size)
            batch_x = x_train_s[idx]
            batch_y = y_train_s[idx]

            noise = np.random.normal(0, 1, (batch_size, latent_dim)).astype(np.float32)
            fake_y = generator.predict([noise, batch_x], verbose=0)

            real_labels = np.ones((batch_size, 1), dtype=np.float32) * label_smooth_real
            fake_labels = np.ones((batch_size, 1), dtype=np.float32) * label_smooth_fake

            d_loss_real = discriminator.train_on_batch([batch_y, batch_x], real_labels)
            d_loss_fake = discriminator.train_on_batch([fake_y, batch_x], fake_labels)
            d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])
            d_losses.append(float(d_loss))

            noise_g = np.random.normal(0, 1, (batch_size, latent_dim)).astype(np.float32)
            trick_labels = np.ones((batch_size, 1), dtype=np.float32) * label_smooth_real
            g_loss = combined.train_on_batch([noise_g, batch_x], trick_labels)
            g_losses.append(float(g_loss))

        val_idx = np.random.randint(0, x_test_s.shape[0], val_sample_size)
        val_x = x_test_s[val_idx]
        val_y = y_test_s[val_idx]
        noise_val = np.random.normal(0, 1, (val_x.shape[0], latent_dim)).astype(np.float32)
        val_pred = generator.predict([noise_val, val_x], verbose=0)
        val_mse = float(np.mean((val_y - val_pred) ** 2))

        if epoch == 1 or epoch == epochs or (epoch % log_freq == 0):
            print(
                f"Epoch {epoch}/{epochs} | d_loss(avg)={np.mean(d_losses):.4f} | "
                f"g_loss(avg)={np.mean(g_losses):.4f} | val_mse(scaled)={val_mse:.6f}"
            )


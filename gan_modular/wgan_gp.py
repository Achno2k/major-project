import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.optimizers import Adam


def build_wgan_generator(latent_dim: int, input_dim: int, output_dim: int) -> Model:
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


def build_wgan_critic(input_dim: int, output_dim: int) -> Model:
    y_input = Input(shape=(output_dim,), name="y_input")
    cond_input = Input(shape=(input_dim,), name="cond_input")
    x = layers.Concatenate()([y_input, cond_input])
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="linear")(x)
    return Model([y_input, cond_input], out, name="wgan_critic")


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
                    c_loss = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score) + gp_lambda * gp

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
        val_mse = float(np.mean(np.square(val_y - val_pred)))

        if epoch == 1 or epoch == epochs or (epoch % log_freq == 0):
            print(
                f"Epoch {epoch}/{epochs} | critic_loss(avg)={np.mean(critic_losses):.4f} | "
                f"g_loss(avg)={np.mean(generator_losses):.4f} | gp(avg)={np.mean(gp_values):.4f} | "
                f"val_mse(scaled)={val_mse:.6f}"
            )


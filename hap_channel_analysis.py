# -*- coding: utf-8 -*-
"""
HAP Channel Analysis with Conditional GAN
==========================================
This script performs:
1. Data cleaning and preprocessing for wireless channel data
2. Training a Conditional GAN (CGAN) to generate synthetic channel data
3. Performance analysis for High Altitude Platform (HAP) selection strategies
4. Generates graphs: Ergodic Capacity vs Pt, Outage Probability vs Pt, Throughput vs SNR

Author: Fixed and improved version
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import time
import warnings

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================
# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "output")
MODEL_DIR = os.path.join(OUT_DIR, "models")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Dataset path
INPUT_XL = os.path.join(SCRIPT_DIR, "corrected_channel_dataset.xlsx")

# Training hyperparameters
EPOCHS = 10         # Reduced for quick testing
BATCH_SIZE = 64
LATENT_DIM = 32
LEARNING_RATE = 1e-4  # Reduced for stability
LOG_FREQ = 10

# Monte Carlo simulation parameters
N_TRIALS = 100000
NUM_HAPS = 4

# Physical constants
K_BOLTZMANN = 1.38064852e-23  # Boltzmann constant (J/K)
TEMPERATURE = 290.0           # Kelvin
BANDWIDTH = 20e6              # 20 MHz

# Column definitions
X_COLS = [
    "Visibility_km",
    "LWC_gm3",
    "CNC_cm3",
    "Temperature_C",
    "Frequency_GHz",
    "Path_km"
]

Y_COLS = [
    "FSO_Attenuation_Visibility_dB",
    "FSO_Attenuation_Cloud_Model_dB",
    "RF_Total_Attenuation_dB"
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def load_and_clean_data(filepath):
    """
    Load dataset from Excel and clean RF_Total_Attenuation_dB column.
    Uses IQR-based outlier detection and KNN imputation.
    """
    print_section("DATA LOADING & CLEANING")
    
    print(f"Loading Excel: {filepath}")
    df = pd.read_excel(filepath)
    print(f"Loaded shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for required columns
    missing_cols = [c for c in X_COLS + Y_COLS if c not in df.columns]
    if missing_cols:
        raise RuntimeError(f"Missing required columns: {missing_cols}")
    
    # Original statistics
    print("\nOriginal RF_Total_Attenuation_dB statistics:")
    print(df["RF_Total_Attenuation_dB"].describe())
    
    # Clean RF_Total_Attenuation_dB
    rf = df["RF_Total_Attenuation_dB"].copy()
    
    # Mark invalid values (negative or extreme outliers)
    invalid_mask = (pd.isna(rf)) | (rf < 0.0) | (rf > 200.0)
    
    # IQR-based outlier detection
    q1, q3 = rf.quantile(0.25), rf.quantile(0.75)
    iqr = q3 - q1
    iqr_mask = (rf < (q1 - 3 * iqr)) | (rf > (q3 + 3 * iqr))
    
    combined_invalid = invalid_mask | iqr_mask
    print(f"\nMarked as invalid/outliers: {combined_invalid.sum()} / {rf.size}")
    
    df.loc[combined_invalid, "RF_Total_Attenuation_dB"] = np.nan
    
    # KNN Imputation for missing values
    if df["RF_Total_Attenuation_dB"].isna().sum() > 0:
        print("Performing KNN imputation...")
        impute_features = X_COLS + Y_COLS
        sub = df[impute_features].copy()
        
        scaler_for_knn = StandardScaler()
        sub_scaled = scaler_for_knn.fit_transform(sub.astype(float))
        
        imputer = KNNImputer(n_neighbors=6, weights="distance")
        sub_imputed_scaled = imputer.fit_transform(sub_scaled)
        sub_imputed = scaler_for_knn.inverse_transform(sub_imputed_scaled)
        
        df[impute_features] = sub_imputed
    
    # Clip to realistic range
    df["RF_Total_Attenuation_dB"] = df["RF_Total_Attenuation_dB"].clip(lower=20.0, upper=160.0)
    
    print("\nAfter cleaning RF_Total_Attenuation_dB statistics:")
    print(df["RF_Total_Attenuation_dB"].describe())
    
    return df


def build_generator(latent_dim, input_dim, output_dim):
    """Build the Generator network for CGAN."""
    noise_input = Input(shape=(latent_dim,), name="noise_input")
    cond_input = Input(shape=(input_dim,), name="cond_input")
    
    x = layers.Concatenate()([noise_input, cond_input])
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(output_dim, activation="linear")(x)
    
    return Model([noise_input, cond_input], out, name="generator")


def build_discriminator(input_dim, output_dim):
    """Build the Discriminator network for CGAN."""
    y_input = Input(shape=(output_dim,), name="y_input")
    cond_input = Input(shape=(input_dim,), name="cond_input")
    
    x = layers.Concatenate()([y_input, cond_input])
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    
    return Model([y_input, cond_input], out, name="discriminator")


def train_cgan(X_train_s, Y_train_s, X_test_s, Y_test_s, scaler_Y):
    """Train the Conditional GAN."""
    print_section("CGAN TRAINING")
    
    input_dim = X_train_s.shape[1]
    output_dim = Y_train_s.shape[1]
    
    # Build models
    generator = build_generator(LATENT_DIM, input_dim, output_dim)
    discriminator = build_discriminator(input_dim, output_dim)
    
    # Compile discriminator
    discriminator.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, beta_1=0.5),
        loss=BinaryCrossentropy(),
        metrics=["accuracy"]
    )
    
    # Build combined model (freeze discriminator during generator updates)
    discriminator.trainable = False
    noise_in = Input(shape=(LATENT_DIM,), name="cg_noise_in")
    cond_in = Input(shape=(input_dim,), name="cg_cond_in")
    gen_out = generator([noise_in, cond_in])
    disc_out = discriminator([gen_out, cond_in])
    combined = Model([noise_in, cond_in], disc_out, name="cgan_combined")
    combined.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, beta_1=0.5),
        loss=BinaryCrossentropy()
    )
    discriminator.trainable = True
    
    print("Generator and Discriminator built.")
    print(f"Input dim: {input_dim}, Output dim: {output_dim}, Latent dim: {LATENT_DIM}")
    
    # Training loop
    steps_per_epoch = max(1, X_train_s.shape[0] // BATCH_SIZE)
    
    # Use softer label smoothing
    real_label_val = 0.95
    fake_label_val = 0.05
    
    history = {'d_loss': [], 'g_loss': [], 'val_mse': []}
    
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        d_losses = []
        g_losses = []
        
        for step in range(steps_per_epoch):
            # Sample real data
            idx = np.random.randint(0, X_train_s.shape[0], BATCH_SIZE)
            batch_X = X_train_s[idx]
            batch_Y = Y_train_s[idx]
            
            # Generate fake data
            noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM)).astype(np.float32)
            fake_Y = generator.predict([noise, batch_X], verbose=0)
            
            # Labels with smoothing
            real_labels = np.ones((BATCH_SIZE, 1), dtype=np.float32) * real_label_val
            fake_labels = np.ones((BATCH_SIZE, 1), dtype=np.float32) * fake_label_val
            
            # Train discriminator on real and fake
            d_loss_real = discriminator.train_on_batch([batch_Y, batch_X], real_labels)
            d_loss_fake = discriminator.train_on_batch([fake_Y, batch_X], fake_labels)
            d_losses.append(0.5 * (d_loss_real[0] + d_loss_fake[0]))
            
            # Train generator through combined model
            noise_g = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM)).astype(np.float32)
            trick_labels = np.ones((BATCH_SIZE, 1), dtype=np.float32)
            g_loss = combined.train_on_batch([noise_g, batch_X], trick_labels)
            g_losses.append(g_loss)
        
        # Record history
        history['d_loss'].append(np.mean(d_losses))
        history['g_loss'].append(np.mean(g_losses))
        
        # Validation
        if epoch % LOG_FREQ == 0 or epoch == 1 or epoch == EPOCHS:
            val_idx = np.random.randint(0, X_test_s.shape[0], min(256, X_test_s.shape[0]))
            val_noise = np.random.normal(0, 1, (len(val_idx), LATENT_DIM)).astype(np.float32)
            val_pred_s = generator.predict([val_noise, X_test_s[val_idx]], verbose=0)
            val_mse = mean_squared_error(Y_test_s[val_idx], val_pred_s)
            history['val_mse'].append(val_mse)
            
            print(f"Epoch {epoch:3d}/{EPOCHS} | D_loss: {np.mean(d_losses):.4f} | "
                  f"G_loss: {np.mean(g_losses):.4f} | Val_MSE: {val_mse:.6f}")
        
        # Checkpoint
        if epoch % 50 == 0:
            generator.save(os.path.join(MODEL_DIR, f"generator_epoch{epoch}.h5"))
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f} seconds")
    
    # Save final models
    generator.save(os.path.join(MODEL_DIR, "generator_final.h5"))
    discriminator.save(os.path.join(MODEL_DIR, "discriminator_final.h5"))
    print(f"Models saved to: {MODEL_DIR}")
    
    return generator, discriminator, history


def generate_synthetic_data(generator, X_train_s, scaler_Y, num_samples=20000):
    """Generate synthetic data using trained generator."""
    print_section("GENERATING SYNTHETIC DATA")
    
    batch_size = 5000
    generated_parts = []
    
    for i in range(num_samples // batch_size):
        noise = np.random.normal(0, 1, (batch_size, LATENT_DIM)).astype(np.float32)
        cond_idx = np.random.randint(0, X_train_s.shape[0], batch_size)
        cond_input = X_train_s[cond_idx]
        gen_batch_s = generator.predict([noise, cond_input], verbose=0)
        generated_parts.append(gen_batch_s)
    
    # Handle remainder
    rem = num_samples - (num_samples // batch_size) * batch_size
    if rem > 0:
        noise = np.random.normal(0, 1, (rem, LATENT_DIM)).astype(np.float32)
        cond_idx = np.random.randint(0, X_train_s.shape[0], rem)
        cond_input = X_train_s[cond_idx]
        gen_batch_s = generator.predict([noise, cond_input], verbose=0)
        generated_parts.append(gen_batch_s)
    
    generated_scaled = np.vstack(generated_parts)
    generated_data = scaler_Y.inverse_transform(generated_scaled)
    
    print(f"Generated {generated_data.shape[0]} synthetic samples")
    
    return generated_data


def create_hap_partitions(rf_data, num_haps=4, method='random_fair'):
    """
    Create HAP partitions for fair comparison.
    
    Methods:
    - 'random_fair': Randomly shuffle and split - all HAPs have similar distributions
                     Any HAP can be best at any given instant (RECOMMENDED)
    - 'realistic_variation': Similar base distribution with slight per-HAP variations
    - 'sequential': Simple sequential split
    """
    n = rf_data.size
    rng = np.random.default_rng(42)
    
    if method == 'random_fair':
        # Randomly shuffle data and split equally among HAPs
        # This ensures all HAPs have similar statistical distributions
        # but different actual samples - any HAP can be best at any instant
        shuffled_indices = rng.permutation(n)
        shuffled_data = rf_data[shuffled_indices]
        
        n_per_hap = n // num_haps
        parts = []
        for i in range(num_haps):
            start_idx = i * n_per_hap
            end_idx = (i + 1) * n_per_hap if i < num_haps - 1 else n
            parts.append(shuffled_data[start_idx:end_idx].copy())
        
        return parts
    
    elif method == 'realistic_variation':
        # Each HAP gets randomly shuffled data with slight offset
        # Simulates HAPs at slightly different distances/conditions
        # Offsets are small so any HAP can still be best at any instant
        shuffled_indices = rng.permutation(n)
        shuffled_data = rf_data[shuffled_indices]
        
        n_per_hap = n // num_haps
        # Small offsets in dB (realistic variation between HAP positions)
        offsets = [0.0, 1.0, 2.0, 3.0]  # HAP 1 slightly better on average
        
        parts = []
        for i in range(num_haps):
            start_idx = i * n_per_hap
            end_idx = (i + 1) * n_per_hap if i < num_haps - 1 else n
            hap_data = shuffled_data[start_idx:end_idx].copy() + offsets[i]
            parts.append(hap_data)
        
        return parts
    
    elif method == 'sequential':
        sizes = [(n // num_haps) + (1 if i < n % num_haps else 0) for i in range(num_haps)]
        parts = []
        idx = 0
        for s in sizes:
            parts.append(rf_data[idx:idx + s].copy())
            idx += s
        return parts
    
    return None


def compute_channel_metrics(hap_parts, Pt_values, path_losses, n_trials, noise_power):
    """
    Compute outage probability and ergodic capacity using Monte Carlo simulation.
    Uses selection combining (best SNR selection).
    """
    num_haps = len(hap_parts)
    outage_prob = np.zeros(len(Pt_values))
    ergodic_cap = np.zeros(len(Pt_values))
    
    rng = np.random.default_rng(42)
    
    # SNR threshold for outage (5 dB in linear scale)
    snr_threshold_linear = 10 ** (5.0 / 10.0)
    
    for pi, Pt in enumerate(Pt_values):
        # Sample attenuation values for each HAP
        idxs = [rng.integers(0, hap_parts[h].size, size=n_trials) for h in range(num_haps)]
        loss_db_mat = np.column_stack([hap_parts[h][idxs[h]] for h in range(num_haps)])
        
        # Convert dB to linear gain
        loss_db_mat = np.clip(loss_db_mat, -200.0, 200.0)
        gain_lin = 10.0 ** (-loss_db_mat / 10.0)
        
        # Apply per-HAP path loss factors
        gain_lin = gain_lin * path_losses[np.newaxis, :num_haps]
        
        # Received power and SNR
        Pr = Pt * gain_lin
        SNRs = Pr / (noise_power + 1e-300)
        
        # Selection combining: choose max SNR across HAPs
        SNR_selected = np.max(SNRs, axis=1)
        
        # Compute metrics
        outage_prob[pi] = np.mean(SNR_selected < snr_threshold_linear)
        ergodic_cap[pi] = np.mean(np.log2(1.0 + SNR_selected))
    
    return outage_prob, ergodic_cap


def compute_throughput_vs_snr(hap_parts, path_losses, noise_power, Pt_ref=0.5):
    """
    Compute throughput vs SNR for intelligent and random selection strategies.
    """
    num_haps = len(hap_parts)
    n_trials = N_TRIALS
    
    rng = np.random.default_rng(12345)
    
    # Pre-allocate arrays
    inst_SNRs = np.zeros((n_trials, num_haps))
    inst_throughput = np.zeros_like(inst_SNRs)
    
    # Compute instantaneous SNR and throughput for each HAP
    for h in range(num_haps):
        arr = hap_parts[h]
        idxs = rng.integers(0, arr.size, size=n_trials)
        loss_db = arr[idxs]
        
        # Convert to linear gain
        gain_lin = 10.0 ** (-loss_db / 10.0) * path_losses[h]
        
        # Received power and SNR
        Pr = Pt_ref * gain_lin
        SNR_lin = Pr / (noise_power + 1e-300)
        
        # Throughput (Shannon capacity in bits/s/Hz converted to Mbps)
        throughput_mbps = BANDWIDTH * np.log2(1.0 + SNR_lin) / 1e6
        
        inst_SNRs[:, h] = SNR_lin
        inst_throughput[:, h] = throughput_mbps
    
    # Intelligent selection: max SNR per trial
    best_idx = np.argmax(inst_SNRs, axis=1)
    intelligent_throughput = inst_throughput[np.arange(n_trials), best_idx]
    intelligent_snr = inst_SNRs[np.arange(n_trials), best_idx]
    
    # Random selection
    random_choices = rng.integers(0, num_haps, size=n_trials)
    random_throughput = inst_throughput[np.arange(n_trials), random_choices]
    random_snr = inst_SNRs[np.arange(n_trials), random_choices]
    
    return {
        'intelligent_snr': intelligent_snr,
        'intelligent_throughput': intelligent_throughput,
        'random_snr': random_snr,
        'random_throughput': random_throughput
    }


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def setup_plot_style():
    """Set up matplotlib style for publication-quality plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.figsize': (10, 7),
        'lines.linewidth': 2,
        'lines.markersize': 8,
    })


def plot_ergodic_capacity_vs_power(Pt_values, ec_real, ec_gan_parts, save_path):
    """
    Plot 1: Ergodic Capacity vs Transmit Power
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
    
    # Plot real data
    ax.plot(Pt_values, ec_real, 'o-', color=colors[0], label='Real Data (Selection)', 
            linewidth=2.5, markersize=8)
    
    # Plot GAN parts (simulating different HAPs)
    markers = ['s', '^', 'D', 'v']
    for i, ec_p in enumerate(ec_gan_parts):
        ax.plot(Pt_values, ec_p, f'{markers[i]}--', color=colors[i+1], 
                label=f'HAP {i+1} (GAN)', linewidth=1.8, markersize=7, alpha=0.8)
    
    ax.set_xlabel('Transmit Power (W)', fontweight='bold')
    ax.set_ylabel('Ergodic Capacity (bits/s/Hz)', fontweight='bold')
    ax.set_title('Ergodic Capacity vs Transmit Power\n(Selection Combining)', fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([Pt_values[0], Pt_values[-1]])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_outage_probability_vs_power(Pt_values, out_real, out_gan_parts, save_path):
    """
    Plot 2: Outage Probability vs Transmit Power (semi-log)
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
    
    # Replace zeros with small value for log plot
    out_real_plot = np.where(out_real > 0, out_real, 1e-6)
    
    ax.semilogy(Pt_values, out_real_plot, 'o-', color=colors[0], 
                label='Real Data (Selection)', linewidth=2.5, markersize=8)
    
    markers = ['s', '^', 'D', 'v']
    for i, out_p in enumerate(out_gan_parts):
        out_p_plot = np.where(out_p > 0, out_p, 1e-6)
        ax.semilogy(Pt_values, out_p_plot, f'{markers[i]}--', color=colors[i+1],
                    label=f'HAP {i+1} (GAN)', linewidth=1.8, markersize=7, alpha=0.8)
    
    ax.set_xlabel('Transmit Power (W)', fontweight='bold')
    ax.set_ylabel('Outage Probability', fontweight='bold')
    ax.set_title('Outage Probability vs Transmit Power\n(SNR Threshold = 5 dB)', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim([Pt_values[0], Pt_values[-1]])
    ax.set_ylim([1e-4, 1.0])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_throughput_vs_snr(results, save_path):
    """
    Plot 3: Throughput vs SNR (comparing Intelligent vs Random selection)
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Convert SNR to dB for plotting
    snr_db_intelligent = 10 * np.log10(results['intelligent_snr'] + 1e-300)
    snr_db_random = 10 * np.log10(results['random_snr'] + 1e-300)
    
    # Create bins for SNR
    snr_bins = np.linspace(-10, 50, 31)
    snr_centers = (snr_bins[:-1] + snr_bins[1:]) / 2
    
    # Compute mean throughput per SNR bin
    intelligent_means = []
    random_means = []
    
    for i in range(len(snr_bins) - 1):
        # Intelligent selection
        mask_int = (snr_db_intelligent >= snr_bins[i]) & (snr_db_intelligent < snr_bins[i+1])
        if np.sum(mask_int) > 0:
            intelligent_means.append(np.mean(results['intelligent_throughput'][mask_int]))
        else:
            intelligent_means.append(np.nan)
        
        # Random selection
        mask_rand = (snr_db_random >= snr_bins[i]) & (snr_db_random < snr_bins[i+1])
        if np.sum(mask_rand) > 0:
            random_means.append(np.mean(results['random_throughput'][mask_rand]))
        else:
            random_means.append(np.nan)
    
    intelligent_means = np.array(intelligent_means)
    random_means = np.array(random_means)
    
    # Plot
    ax.plot(snr_centers, intelligent_means, 'o-', color='#2E86AB', 
            label='Intelligent Selection', linewidth=2.5, markersize=8)
    ax.plot(snr_centers, random_means, 's--', color='#F18F01', 
            label='Random Selection', linewidth=2.5, markersize=8)
    
    # Theoretical Shannon capacity for reference
    snr_lin = 10 ** (snr_centers / 10)
    theoretical = BANDWIDTH * np.log2(1 + snr_lin) / 1e6
    ax.plot(snr_centers, theoretical, 'k:', label='Shannon Limit', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('SNR (dB)', fontweight='bold')
    ax.set_ylabel('Throughput (Mbps)', fontweight='bold')
    ax.set_title('Throughput vs SNR\n(Intelligent vs Random HAP Selection)', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-10, 50])
    ax.set_ylim([0, None])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_training_history(history, save_path):
    """Plot GAN training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['d_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['d_loss'], 'b-', label='Discriminator Loss', linewidth=2)
    ax1.plot(epochs, history['g_loss'], 'r-', label='Generator Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title('GAN Training Loss', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation MSE plot
    val_epochs = [1] + list(range(LOG_FREQ, EPOCHS + 1, LOG_FREQ))
    if EPOCHS not in val_epochs:
        val_epochs.append(EPOCHS)
    ax2.plot(val_epochs[:len(history['val_mse'])], history['val_mse'], 'g-o', 
             label='Validation MSE', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('MSE (scaled)', fontweight='bold')
    ax2.set_title('Validation MSE', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_comparison_summary(results, save_path):
    """Plot summary comparison of intelligent vs random selection."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. CDF of throughput
    ax1 = axes[0]
    for arr, label, col in [(results['random_throughput'], "Random", "#F18F01"),
                            (results['intelligent_throughput'], "Intelligent", "#2E86AB")]:
        sorted_v = np.sort(arr)
        p = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        ax1.plot(sorted_v, p, label=label, color=col, linewidth=2.5)
    ax1.set_xlabel("Throughput (Mbps)", fontweight='bold')
    ax1.set_ylabel("CDF", fontweight='bold')
    ax1.set_title("Throughput CDF", fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    # 2. Boxplot
    ax2 = axes[1]
    bp = ax2.boxplot([results['random_throughput'], results['intelligent_throughput']], 
                     labels=["Random", "Intelligent"], patch_artist=True)
    colors = ['#F18F01', '#2E86AB']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel("Throughput (Mbps)", fontweight='bold')
    ax2.set_title("Throughput Distribution", fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # 3. Outage comparison bar
    ax3 = axes[2]
    thr_db = 5.0
    out_r = np.mean(10 * np.log10(results['random_snr'] + 1e-300) < thr_db)
    out_i = np.mean(10 * np.log10(results['intelligent_snr'] + 1e-300) < thr_db)
    bars = ax3.bar(["Random", "Intelligent"], [out_r, out_i], color=colors, alpha=0.8)
    ax3.set_ylim(0, 1)
    ax3.set_title(f"Outage Probability\n(SNR < {thr_db} dB)", fontweight='bold')
    ax3.set_ylabel("Outage Probability", fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, [out_r, out_i]):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                 f'{val:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print_section("HAP CHANNEL ANALYSIS WITH CONDITIONAL GAN")
    print(f"Output directory: {OUT_DIR}")
    print(f"Model directory: {MODEL_DIR}")
    
    # Setup plot style
    setup_plot_style()
    
    # Calculate noise power
    noise_power = K_BOLTZMANN * TEMPERATURE * BANDWIDTH
    print(f"Noise power: {noise_power:.2e} W")
    
    # Per-HAP path loss factors (equal for fair comparison - no HAP has inherent advantage)
    path_losses = np.array([0.10, 0.10, 0.10, 0.10])
    
    # Transmit power range
    Pt_values = np.linspace(0.1, 2.0, 15)
    
    # =========================================================================
    # STEP 1: Load and clean data
    # =========================================================================
    df = load_and_clean_data(INPUT_XL)
    
    # Save cleaned dataset
    clean_path = os.path.join(OUT_DIR, "cleaned_dataset.xlsx")
    df.to_excel(clean_path, index=False)
    print(f"Saved cleaned dataset: {clean_path}")
    
    # =========================================================================
    # STEP 2: Prepare data for CGAN
    # =========================================================================
    print_section("DATA PREPARATION")
    
    X = df[X_COLS].values.astype(np.float32)
    Y = df[Y_COLS].values.astype(np.float32)
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, shuffle=True
    )
    
    scaler_X = StandardScaler().fit(X_train)
    scaler_Y = StandardScaler().fit(Y_train)
    
    X_train_s = scaler_X.transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    Y_train_s = scaler_Y.transform(Y_train)
    Y_test_s = scaler_Y.transform(Y_test)
    
    print(f"Training samples: {X_train_s.shape[0]}")
    print(f"Test samples: {X_test_s.shape[0]}")
    
    # =========================================================================
    # STEP 3: Train CGAN
    # =========================================================================
    generator, discriminator, history = train_cgan(
        X_train_s, Y_train_s, X_test_s, Y_test_s, scaler_Y
    )
    
    # Plot training history
    plot_training_history(history, os.path.join(OUT_DIR, "training_history.png"))
    
    # =========================================================================
    # STEP 4: Generate synthetic data
    # =========================================================================
    gan_data = generate_synthetic_data(generator, X_train_s, scaler_Y, num_samples=20000)
    
    # Save datasets
    real_data = np.vstack([Y_train, Y_test])
    np.save(os.path.join(OUT_DIR, "real_dataset.npy"), real_data)
    np.save(os.path.join(OUT_DIR, "gan_dataset.npy"), gan_data)
    pd.DataFrame(real_data, columns=Y_COLS).to_csv(os.path.join(OUT_DIR, "real_dataset.csv"), index=False)
    pd.DataFrame(gan_data, columns=Y_COLS).to_csv(os.path.join(OUT_DIR, "gan_dataset.csv"), index=False)
    print(f"Saved real and GAN datasets")
    
    # =========================================================================
    # STEP 5: Create HAP partitions
    # =========================================================================
    print_section("HAP PARTITION ANALYSIS")
    
    # RF attenuation is the last column
    real_rf = real_data[:, -1].astype(np.float64).ravel()
    gan_rf = gan_data[:, -1].astype(np.float64).ravel()
    
    # Create fair HAP partitions (randomly shuffled - all HAPs have similar distributions)
    # Any HAP can be best at any given instant - no deliberate bias
    real_parts = create_hap_partitions(real_rf, num_haps=NUM_HAPS, method='random_fair')
    gan_parts = create_hap_partitions(gan_rf, num_haps=NUM_HAPS, method='random_fair')
    
    print("\nReal data HAP partitions:")
    for i, p in enumerate(real_parts, 1):
        print(f"  HAP {i}: samples={p.size}, mean={p.mean():.2f} dB, std={p.std():.2f} dB")
    
    print("\nGAN data HAP partitions:")
    for i, p in enumerate(gan_parts, 1):
        print(f"  HAP {i}: samples={p.size}, mean={p.mean():.2f} dB, std={p.std():.2f} dB")
    
    # =========================================================================
    # STEP 6: Compute performance metrics
    # =========================================================================
    print_section("COMPUTING PERFORMANCE METRICS")
    
    print("Computing metrics for real data...")
    out_real, ec_real = compute_channel_metrics(
        real_parts, Pt_values, path_losses, N_TRIALS, noise_power
    )
    
    print("Computing metrics for GAN data (per HAP)...")
    out_gan_parts = []
    ec_gan_parts = []
    
    for i, gan_part in enumerate(gan_parts):
        # Create sub-partitions for each GAN part
        sub_parts = create_hap_partitions(gan_part, num_haps=NUM_HAPS, method='sequential')
        out_p, ec_p = compute_channel_metrics(
            sub_parts, Pt_values, path_losses, N_TRIALS // 2, noise_power
        )
        out_gan_parts.append(out_p)
        ec_gan_parts.append(ec_p)
        print(f"  HAP {i+1} done")
    
    # Compute throughput vs SNR
    print("Computing throughput vs SNR comparison...")
    throughput_results = compute_throughput_vs_snr(real_parts, path_losses, noise_power)
    
    # =========================================================================
    # STEP 7: Generate all required plots
    # =========================================================================
    print_section("GENERATING PLOTS")
    
    # Plot 1: Ergodic Capacity vs Transmit Power
    plot_ergodic_capacity_vs_power(
        Pt_values, ec_real, ec_gan_parts,
        os.path.join(OUT_DIR, "1_ergodic_capacity_vs_power.png")
    )
    
    # Plot 2: Outage Probability vs Transmit Power
    plot_outage_probability_vs_power(
        Pt_values, out_real, out_gan_parts,
        os.path.join(OUT_DIR, "2_outage_probability_vs_power.png")
    )
    
    # Plot 3: Throughput vs SNR
    plot_throughput_vs_snr(
        throughput_results,
        os.path.join(OUT_DIR, "3_throughput_vs_snr.png")
    )
    
    # Bonus: Comparison summary plot
    plot_comparison_summary(
        throughput_results,
        os.path.join(OUT_DIR, "4_selection_comparison_summary.png")
    )
    
    # =========================================================================
    # STEP 8: Save numeric results
    # =========================================================================
    print_section("SAVING RESULTS")
    
    # Save main results
    results_df = pd.DataFrame({
        "Pt_W": Pt_values,
        "Outage_Real": out_real,
        "EC_Real": ec_real
    })
    for i, (out_p, ec_p) in enumerate(zip(out_gan_parts, ec_gan_parts), 1):
        results_df[f"Outage_HAP{i}"] = out_p
        results_df[f"EC_HAP{i}"] = ec_p
    
    results_df.to_csv(os.path.join(OUT_DIR, "performance_metrics.csv"), index=False)
    
    # Save throughput comparison
    np.save(os.path.join(OUT_DIR, "intelligent_throughput.npy"), throughput_results['intelligent_throughput'])
    np.save(os.path.join(OUT_DIR, "random_throughput.npy"), throughput_results['random_throughput'])
    
    # Print summary statistics
    print_section("SUMMARY STATISTICS")
    
    mean_int = np.mean(throughput_results['intelligent_throughput'])
    mean_rand = np.mean(throughput_results['random_throughput'])
    improvement = 100 * (mean_int - mean_rand) / (mean_rand + 1e-30)
    
    print(f"Mean Throughput (Intelligent): {mean_int:.4f} Mbps")
    print(f"Mean Throughput (Random):      {mean_rand:.4f} Mbps")
    print(f"Improvement:                   {improvement:.2f}%")
    
    # Statistical test
    t_stat, p_val = stats.ttest_rel(
        throughput_results['intelligent_throughput'],
        throughput_results['random_throughput']
    )
    print(f"\nPaired t-test: t = {t_stat:.3f}, p = {p_val:.3e}")
    
    wins = np.sum(throughput_results['intelligent_throughput'] > throughput_results['random_throughput'])
    print(f"Intelligent beats Random: {wins}/{N_TRIALS} trials ({100*wins/N_TRIALS:.2f}%)")
    
    print_section("COMPLETE")
    print(f"All outputs saved to: {OUT_DIR}")
    print("\nGenerated files:")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()


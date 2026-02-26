from dataclasses import dataclass


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


@dataclass
class TrainConfig:
    file_path: str = "corrected_channel_dataset.xlsx"
    model_dir: str = "generated_models"
    output_dir: str = "generated_datasets"
    epochs: int = 5
    batch_size: int = 64
    latent_dim: int = 32
    lr: float = 1e-4
    log_freq: int = 25
    num_samples: int = 20000
    gen_batch_size: int = 5000
    seed: int = 42


@dataclass
class WGANConfig(TrainConfig):
    critic_steps: int = 5
    gp_lambda: float = 10.0
    model_dir: str = "wgan_models"
    output_dir: str = "generated_datasets_wgan"


@dataclass
class CGANConfig(TrainConfig):
    beta_1: float = 0.5
    label_smooth_real: float = 0.9
    label_smooth_fake: float = 0.0
    model_dir: str = "cgan_models"
    output_dir: str = "generated_datasets_cgan"

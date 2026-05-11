# Modular GAN Structure

This folder contains modularized implementations for both CGAN and WGAN-GP.

## Files

- `config.py`: shared dataset columns and training config dataclasses
- `data.py`: data loading, preprocessing, generation sampling, and output saving
- `utils.py`: utility helpers (seed setup)
- `cgan.py`: CGAN model definitions and CGAN training loop
- `wgan_gp.py`: WGAN-GP model definitions and WGAN-GP training loop
- `run_cgan.py`: CLI entrypoint for CGAN
- `run_wgan.py`: CLI entrypoint for WGAN-GP
- `run_hap_analysis.py`: CLI entrypoint for HAP analysis using generated CSV datasets
- `compare_hap_outputs.py`: compare HAP-analysis outputs for CGAN vs WGAN

## Usage

Run CGAN:

```bash
cd gan_modular
python3 run_cgan.py --file-path ../corrected_channel_dataset.xlsx
```

Run WGAN-GP:

```bash
cd gan_modular
python3 run_wgan.py --file-path ../corrected_channel_dataset.xlsx
```

Default output folders are separate:

- CGAN: `../generated_datasets_cgan`
- WGAN-GP: `../generated_datasets_wgan`

Both scripts save:

- `gan_dataset.npy`, `gan_dataset.csv`
- `real_dataset.npy`, `real_dataset.csv`

and save model artifacts (`generator_final.h5`, discriminator/critic model, scalers) in the configured model directory.

Run HAP analysis on any generated dataset folder:

```bash
python3 run_hap_analysis.py \
  --dataset-dir ../generated_datasets_wgan \
  --synthetic-label WGAN
```

This saves:

- `hap_selection_metrics.csv`
- `hap_outage_probability_real_vs_gan.png`
- `hap_ergodic_capacity_real_vs_gan.png`

Run WGAN-GP and immediately generate the matching HAP plots:

```bash
python3 run_wgan_hap_pipeline.py \
  --file-path ../corrected_channel_dataset.xlsx \
  --output-dir ../generated_datasets_wgan \
  --model-dir ../generated_models_wgan \
  --rf-attenuation-shift-db -20
```

Compare CGAN vs WGAN HAP outputs:

```bash
python3 compare_hap_outputs.py \
  --cgan-dir ../generated_datasets_cgan \
  --wgan-dir ../generated_datasets_wgan \
  --reference-style
```

This saves:

- `hap_metrics_cgan_vs_wgan.csv`
- `compare_hap_outage_cgan_vs_wgan.png`
- `compare_hap_ec_cgan_vs_wgan.png`

Run SHAP ExAI analysis for generated outputs:

```bash
python3 run_shap_exai.py \
  --model both \
  --cgan-model-dir ../.run_artifacts/cgan_models \
  --wgan-model-dir ../.run_artifacts/wgan_models \
  --out-dir ../.run_artifacts/shap_exai
```

This explains how the six conditioning inputs affect each generated output:

- `FSO_Attenuation_Visibility_dB`
- `FSO_Attenuation_Cloud_Model_dB`
- `RF_Total_Attenuation_dB`

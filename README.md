# Multi-Scale Diffusion Transformer  
### for Jointly Simulating User Mobility and Mobile Traffic Pattern

---

## Stage 1: Data Preparation

1. Create folders to store raw and preprocessed data:
    ```
    mkdir -p ./data_src ./data
    ```

2. Run `preprocess_Qtall.ipynb` to construct the pretrained transition matrix `Qt_all.pt`.

---

## Stage 2: Training Models

1. Train the VQ-VAE in `./vqvae`, you can change the configuration in `./vqvae/config/config_vqvae.yaml`
```bash
python train_vqvae.py 
```

2. Train the main model:

```bash
python exe_ft.py \
  --do_vq \
  --time 20250428_204243 \
  --dim 32 \
  --K 128 \
  --it 500000 \
  --coff CA3 \
  --epoch 100
```

>  Make sure the directory `./save_model` exists to store the trained model weights.

### Parameter Description:

| Argument      | Description                                                    |
|---------------|----------------------------------------------------------------|
| `--do_vq`     | Whether to use the pretrained VQ-VAE                           |
| `--time`      | Folder name used as timestamp identifier for saving VQ-VAE     |
| `--dim`       | Dimension of quantized embeddings (e.g., 32)                   |
| `--K`         | Number of codebook categories (e.g., 128)                      |
| `--it`        | Number of training iterations for VQ-VAE (e.g., 500000)        |
| `--coff`      | Wavelet coeffs whose VQ-VAE encoder will be used (e.g., `CA3`) |
| `--epoch`     | Number of training epochs for the main model (e.g., 100)       |

---

## Stage 3: Generating Samples

Use the trained model to generate samples:

```bash
python exe_ft.py \
  --do_vq \
  --time 20250428_204243 \
  --dim 32 \
  --K 128 \
  --it 500000 \
  --coff CA3 \
  --epoch 100 \
  --modelfolder folder_name # replace this with your folder name
```
>  Make sure the directory `./GenData` exists to store the generated samples.

# ComfyUI-H3-Text-Encoders

Use any modern ComfyUI-native text encoder (**Qwen 3.5 2B / 4B / 9B**, **Qwen 3.8 27B**, **Gemma 4**, etc.) as a conditioning encoder for **MiniMax-H3** (**T2VA / Image-to-Video**, **FL2VA**, and **Ref2VA**) in ComfyUI — **zero gradient training, zero LoRA, and zero external MLP projection weights**.

Powered by **Plato's Key** (*closed-form Orthogonal Procrustes rotation + global scale via SVD*, inspired by Stanford Hazy Research's [Plato's Key](https://hazyresearch.stanford.edu/blog/2026-11-25-platos-key)), each adapter is a tiny `.safetensors` key (~15–40 MB) that maps a donor encoder's hidden states directly into MiniMax-H3's native **5120-dim `Qwen3-VL-32B`** geometric space:

$$\hat{Y} = \text{scale} \cdot (X - \mu_{\text{donor}}) R + \mu_{\text{target}}$$

All downstream MiniMax-H3 nodes (`MiniMaxH3ReferenceToVideo`, `MiniMaxH3ImageToVideo`, `SamplerCustomAdvanced`, `VAEDecode`) work out of the box.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/alisson-anjos/ComfyUI-H3-Text-Encoders.git
cd ComfyUI-H3-Text-Encoders
pip install -r requirements.txt
```

Restart ComfyUI.

## Node Usage

1. Load any supported donor text encoder (from [Comfy-Org/Qwen3.5](https://huggingface.co/Comfy-Org/Qwen3.5), [Comfy-Org/Qwen3.8-27B](https://huggingface.co/Comfy-Org/Qwen3.8-27B), [Comfy-Org/gemma-4](https://huggingface.co/Comfy-Org/gemma-4), etc.) using ComfyUI's native **`Load CLIP`** (`CLIPLoader`).
2. Connect its `CLIP` output to **`Plato's Key Patch CLIP (H3 Plug & Play)`** (`PlatosKeyPatchCLIP`, under `conditioning/platos_key`):
   - **`key_file`**: select any calibrated `.safetensors` key placed in `ComfyUI/models/clip_projections/` (or download from [`Alissonerdx/TE-PK-Minimax-H3`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3))
   - **`strength`**: `1.0` *(default full projection)*
3. Connect the patched `CLIP` output directly to ComfyUI's official **`MiniMax H3 Reference to Video`** (`MiniMaxH3ReferenceToVideo`) node — **no LoRA and no extra projection node required**.

### Additional Nodes Included

- **`PlatosKeyApply`**: Applies a `.safetensors` Plato's Key directly to an existing `CONDITIONING` tensor.
- **`PlatosKeyCLIPTextEncode`**: One-step text encode + Plato's Key projection returning `CONDITIONING`.

## Calibrating Your Own Plato's Key (Zero Gradients)

```bash
python calibrate_platos_key.py \
  --donor Qwen/Qwen3.5-4B \
  --target Qwen/Qwen3.5-9B \
  --prompts prompts_corpus.json \
  --align auto \
  --output keys/qwen35_4b_to_h3_32b.safetensors
```

## Calibration Metrics (`Alissonerdx/TE-PK-Minimax-H3`)

Calibrated on `800` domain-matched prompts from [`Alissonerdx/qwen35-h3-training-data`](https://huggingface.co/datasets/Alissonerdx/qwen35-h3-training-data) (90/10 held-out prompt split) directly into MiniMax-H3's native **5120-dim `Qwen3-VL-32B`** geometric space.

| Repo Path (`Alissonerdx/TE-PK-Minimax-H3`) | Donor Encoder | Dim Map | Align | Val Cosine | Val Cos p10 | Val R² | Scale | Size |
|---|---|---|---|---|---|---|---|---|
| `qwen3.5/minimax_h3_32b_pk_qwen3.5_0.8b.safetensors` | `Qwen/Qwen3.5-0.8B` | `1024 → 5120` | `token` | **0.8943** | 0.8423 | 0.0337 | 1.9316 | 20.0 MB |
| `qwen3.5/minimax_h3_32b_pk_qwen3.5_2b.safetensors` | `Qwen/Qwen3.5-2B` | `2048 → 5120` | `token` | **0.8935** | 0.8416 | 0.0370 | 1.6867 | 40.0 MB |
| `qwen3.5/minimax_h3_32b_pk_qwen3.5_9b.safetensors` | `Qwen/Qwen3.5-9B` | `4096 → 5120` | `token` | **0.8908** | 0.8333 | 0.0371 | 1.9752 | 80.0 MB |
| `qwen3.5/minimax_h3_32b_pk_qwen3.5_4b.safetensors` | `Qwen/Qwen3.5-4B` | `2560 → 5120` | `token` | **0.8902** | 0.8348 | 0.0356 | 1.5572 | 50.0 MB |
| `qwen3.8/minimax_h3_32b_pk_qwen3.8_27b.safetensors` | `Qwen/Qwen3.8-27B` | `5120 → 5120` | `token` | **0.8812** | 0.8198 | 0.0338 | 1.7440 | 100.0 MB |
| `gemma4/minimax_h3_32b_pk_gemma4_e4b.safetensors` | `google/gemma-4-E4B-it` | `2560 → 5120` | `word` | **0.8761** | 0.8272 | 0.1125 | 1.1725 | 50.0 MB |
| `gemma4/minimax_h3_32b_pk_gemma4_e2b.safetensors` | `google/gemma-4-E2B-it` | `1536 → 5120` | `word` | **0.8646** | 0.8109 | 0.1031 | 0.9776 | 30.0 MB |
| `gemma4/minimax_h3_32b_pk_gemma4_12b.safetensors` | `google/gemma-4-12B-it` | `3840 → 5120` | `word` | **0.8440** | 0.7873 | 0.0720 | 0.5852 | 75.0 MB |

## Testing

```bash
CUDA_VISIBLE_DEVICES="" python -m unittest tests/test_platos_key.py
```

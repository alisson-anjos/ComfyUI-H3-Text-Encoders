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
   - **`key_file`**: select any calibrated `.safetensors` key placed in `ComfyUI/models/clip_projections/` (or download from [`Alissonerdx/TEs-Minimax-H3`](https://huggingface.co/Alissonerdx/TEs-Minimax-H3))
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

## Calibration Metrics (`Alissonerdx/TEs-Minimax-H3`)

Calibrated on prompts from [`Alissonerdx/qwen35-h3-training-data`](https://huggingface.co/datasets/Alissonerdx/qwen35-h3-training-data) into MiniMax-H3's native **5120-dim 32B** geometric space.

| Donor Encoder | Target Space | Align Mode | Val Cosine Mean | Val R² | Key Size |
|---|---|---|---|---|---|
| *Calibration batch running on RunPod A40 — metrics table will be updated upon completion.* | `5120` (H3 32B) | `token` / `word` | — | — | — |

## Testing

```bash
CUDA_VISIBLE_DEVICES="" python -m unittest tests/test_platos_key.py
```

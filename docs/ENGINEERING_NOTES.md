# Engineering Notes & Calibration Guide

## Why Closed-Form Ridge Beats Orthogonal Procrustes on MiniMax-H3

MiniMax-H3's native **5,120-dim `Qwen3-VL-32B`** space (layer 50, unnormalized) has strong anisotropic outlier channels. Rigid isotropic Procrustes stops at `R² ≈ 0.04`, whereas closed-form Ridge scales each target channel independently:

$$\hat{Y} = (X - \mu_{\text{donor}}) W + \mu_{\text{target}}, \qquad W = (X_c^\top X_c + \lambda I)^{-1} X_c^\top Y_c$$

This achieves **`R² = 0.990–0.992`** (**147× lower MSE**) on same-tokenizer encoders (`Qwen 3.5` and `Qwen 3.8`) with zero gradient training.

## Key Variants Explained

| Variant | Method | Tokenizer | Node Version | Best For |
|---|---|---|---|---|
| `*_pk_ridge_*.safetensors` | 1:1 Token Closed-Form Ridge (zero training) | Same BPE (`Qwen 3.5` / `Qwen 3.8`) | `v0.1.0+` | Best overall quality, dialogue & multi-shot |
| `*_pk_lora_*.safetensors` | Forced Qwen-Seg + Multi-Layer Ridge + Bundled Rank-32 TE LoRA | Cross-family (`Gemma 4` / `Inkling-2`) | `v0.3.0+` | Best cross-family speech & multi-shot |
| `*_pk_ridge_charov_*.safetensors` | Exact Token-Level Character-Overlap `A @ H` Ridge (zero training) | Cross-family (`Gemma 4` / `Inkling-2`) | `v0.2.0+` | Pure zero-training linear cross-family |
| `*_pk_ridge_4layer_*.safetensors` | Legacy Word-Span Pooled Multi-Layer Ridge (zero training) | Cross-family (`Gemma 4` / `Inkling-2`) | `v0.1.8+` | Legacy compatibility / comparison |

## Exact ComfyUI CLIP Parity Details

`PlatosKeyPatchCLIP` automatically enforces:
- `options_default` tuple override & intermediate layer extraction (`donor_layers` / `donor_layer`)
- `RMSNorm` (`layer_norm_hidden_state`) parity matching calibration
- Raw prompt format (`llama_template="{}"`), no chat template injection
- LTXV `min_length=1` padding strip on Gemma 4
- Gemma 4 `<bos>` attention-sink retention during forward pass, then stripped before projection
- MiniMax-H3 `minimax_token_tags` and `5120`-dim `pooled_output` metadata

## Calibrate Your Own Key (Zero Gradients)

Convert any HuggingFace text encoder in minutes with [`calibrate_platos_key.py`](../calibrate_platos_key.py):

```bash
python calibrate_platos_key.py \
  --donor Qwen/Qwen3.5-2B \
  --target Qwen/Qwen3.5-9B \
  --method ridge \
  --align token \
  --output keys/my_ridge_key.safetensors
```

### Python API (`platos_key_core.py`)

```python
from platos_key_core import fit_ridge_key, evaluate_platos_key, save_platos_key

key = fit_ridge_key(x_donor_train, x_target_train, ridge_lambda=100.0)
metrics = evaluate_platos_key(x_donor_val, x_target_val, key)
save_platos_key(key, "ComfyUI/models/clip_projections/my_custom_key.safetensors")
```

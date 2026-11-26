# ComfyUI-H3-Text-Encoders

> [!IMPORTANT]
> **Current `v2 Ridge` keys are already working and ready to test today** (Qwen 3.5 `0.8B–9B` up to `96.26%` cosine / `99.49%` R², Gemma 4 `E2B–31B` up to `92.54%` cosine).
> - **Active improvement (`v3` in progress):** we are recalibrating all keys on our **full ~20k-prompt dataset** directly against MiniMax-H3's native **`Qwen3-VL-32B`** teacher to push fine-grained prompt adherence even higher.
> - **Recommendation:** always prefer the **`Ridge`** keys (`*_pk_ridge_*.safetensors`) and keep `ComfyUI-H3-Text-Encoders` updated.

Use any modern ComfyUI-native text encoder (**Qwen 3.5 2B / 4B / 9B**, **Qwen 3.8 27B**, **Gemma 4**, etc.) as a conditioning encoder for **MiniMax-H3** (**T2VA / Image-to-Video**, **FL2VA**, and **Ref2VA**) in ComfyUI — **zero gradient training, zero LoRA, and zero external MLP projection weights**.

Powered by **Plato's Key** (*closed-form Orthogonal Procrustes rotation + global scale via SVD*, inspired by Stanford Hazy Research's [Plato's Key](https://hazyresearch.stanford.edu/blog/2026-11-25-platos-key)), each adapter is a tiny `.safetensors` key (~15–40 MB) that maps a donor encoder's hidden states directly into MiniMax-H3's native **5120-dim `Qwen3-VL-32B`** geometric space:

$$\hat{Y} = \text{scale} \cdot (X - \mu_{\text{donor}}) R + \mu_{\text{target}}$$

All downstream MiniMax-H3 nodes (`MiniMaxH3ReferenceToVideo`, `MiniMaxH3ImageToVideo`, `SamplerCustomAdvanced`, `VAEDecode`) work out of the box.

## Visual Comparison (Zero-Gradient Ridge, No LoRA, No MLP)

[![MiniMax-H3 Plato's Key Ridge Comparison](showcase/H3_PlatosKey_X_Demo_Final_thumb.jpg)](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/showcase/H3_PlatosKey_X_Demo_Final.mp4)

- [Watch / download side-by-side comparison video (`showcase/H3_PlatosKey_X_Demo_Final.mp4`)](showcase/H3_PlatosKey_X_Demo_Final.mp4)

## Step-by-Step Setup (Where to Place Files)

1. **Install this custom node**:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/alisson-anjos/ComfyUI-H3-Text-Encoders.git
cd ComfyUI-H3-Text-Encoders
pip install -r requirements.txt
```

2. **Download a Text Encoder** (pick any supported model) and place it in **`ComfyUI/models/text_encoders/`**:
   - **Qwen 3.5 (`0.8B / 2B / 4B / 9B`)**: [Comfy-Org/Qwen3.5](https://huggingface.co/Comfy-Org/Qwen3.5) → `ComfyUI/models/text_encoders/`
   - **Gemma 4 (`E2B / E4B / 12B / 31B`)**: [Comfy-Org/gemma-4](https://huggingface.co/Comfy-Org/gemma-4) → `ComfyUI/models/text_encoders/`
   - **Qwen 3.8 (`27B`)**: [Comfy-Org/Qwen3.8-27B](https://huggingface.co/Comfy-Org/Qwen3.8-27B) → `ComfyUI/models/text_encoders/`

3. **Download the matching Plato's Key `.safetensors`** (prefer **`*_pk_ridge_*.safetensors`**) from [**`Alissonerdx/TE-PK-Minimax-H3`**](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3) and place it in **`ComfyUI/models/clip_projections/`**:
   ```text
   ComfyUI/
   └── models/
       ├── text_encoders/
       │   └── qwen3.5_9b_int8_convrot.safetensors        # (or 2B / 4B / Gemma 4)
       └── clip_projections/
           └── minimax_h3_32b_pk_ridge_qwen3.5_9b.safetensors
   ```

4. **Restart ComfyUI** and wire the workflow:
   - **`Load CLIP`** (`CLIPLoader`, type `minimax` for Qwen 3.5) → **`Plato's Key Patch CLIP (H3 Plug & Play)`** (`key_file`: select the matching `.safetensors`, `strength`: `1.00`, `apply_to_pooled`: `False`) → **`MiniMax H3 Reference to Video`** or **`MiniMax H3 Image to Video`** — **no LoRA and no extra projection node required**.

### Additional Nodes Included

- **`PlatosKeyApply`**: Applies a `.safetensors` Plato's Key directly to an existing `CONDITIONING` tensor.
- **`PlatosKeyCLIPTextEncode`**: One-step text encode + Plato's Key projection returning `CONDITIONING`.

### Procrustes vs Ridge vs Norm — Which Key Should I Use?

All adapters are solved in **closed form (zero gradient descent, zero LoRA)**:

- **`Ridge` (`*_pk_ridge_*.safetensors`) — Recommended for MiniMax-H3**: Solves closed-form Ridge regression $(X^\top X + \lambda I)^{-1} X^\top Y$. Unlike isotropic Procrustes, Ridge scales each of the 5,120 target channels independently, accurately matching MiniMax-H3's anisotropic outlier channels (highest **Cosine** and **R²**). Tip: setting `strength = 1.05–1.10` gives the variance-matched (`ridge_norm`) effect directly at inference time.
- **`Procrustes` (`*_pk_*.safetensors`)**: Pure Stanford *Plato's Key* — a strict orthogonal rotation ($R^\top R = I$) plus a single global scale. Preserves exact donor angles, with lower R² on outlier channels.
- **`_norm` variants**: Rescale the projected vector to match `1.00×` target norm/variance, compensating for slight Ridge shrinkage (`~0.92×`).
- **What do `(-1)` and `(-2)` mean in the table?**
  - **`-1` (last layer)**: final transformer hidden state (post-`RMSNorm`), optimal for Qwen 3.5 (`94.9%–96.3%` cosine).
  - **`-2` (penultimate layer)**: second-to-last hidden state (before final LM head norm), optimal for cross-family Gemma 4 (`90.45%` vs `87.8%` at `-1`).
  - **Automatic in ComfyUI**: every `.safetensors` key stores its calibrated layer in its header, and `PlatosKeyPatchCLIP` configures the CLIP layer & norm automatically — you do **not** need `CLIPSetLastLayer`.

## Calibration & Conversion Guide (Create Your Own Key — Zero Gradients)

You can convert **any HuggingFace / ComfyUI-native text encoder** into a MiniMax-H3 adapter in minutes using [`calibrate_platos_key.py`](calibrate_platos_key.py) and [`platos_key_core.py`](platos_key_core.py) — no gradient training, no LoRA, and no backpropagation required.

### 1. Install Calibration Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Conversion Script

- **Same-tokenizer family** (e.g., `Qwen 3.5 2B / 4B` → `Qwen 3.5 9B` / H3 32B): uses exact 1:1 `--align token`:
```bash
python calibrate_platos_key.py \
  --donor Qwen/Qwen3.5-4B \
  --target Qwen/Qwen3.5-9B \
  --prompts prompts_corpus.json \
  --align token \
  --output keys/minimax_h3_32b_pk_qwen3.5_4b.safetensors
```

- **Cross-tokenizer family** (e.g., `Gemma 4` → H3 32B): automatically aligns shared word spans via `--align word`:
```bash
python calibrate_platos_key.py \
  --donor google/gemma-4-E4B-it \
  --target Qwen/Qwen3.5-9B \
  --prompts prompts_corpus.json \
  --align word \
  --output keys/minimax_h3_32b_pk_gemma4_e4b.safetensors
```

### 3. Closed-Form Ridge Solver (Python API)

For maximum **Cosine** and **R²** on anisotropic target spaces (like MiniMax-H3's 5,120-dim 32B space), use `fit_ridge_key` from [`platos_key_core.py`](platos_key_core.py):

```python
from platos_key_core import fit_ridge_key, evaluate_platos_key, save_platos_key

key = fit_ridge_key(
    x_donor_train,          # [N, d_donor]
    x_target_train,         # [N, 5120]
    ridge_lambda=10.0,      # sweep [0.01, 0.1, 1.0, 10.0, 100.0] on held-out val split
    norm_variance=False,    # True for ridge_norm
)
metrics = evaluate_platos_key(x_donor_val, x_target_val, key)
save_platos_key(key, "ComfyUI/models/clip_projections/my_custom_key.safetensors")
```

### 4. Use in ComfyUI

Copy the resulting `.safetensors` file into `ComfyUI/models/clip_projections/`, restart ComfyUI, and select it in **`Plato's Key Patch CLIP (H3 Plug & Play)`**.

## Calibration Metrics (`Alissonerdx/TE-PK-Minimax-H3`)

> **Recommendation:** Always prefer the **`Ridge`** keys (`*_pk_ridge_*.safetensors`) for MiniMax-H3 — they accurately match H3's anisotropic 5,120-dim outlier channels (**Cosine 0.90–0.96+**, **R² up to 0.995**) and deliver dramatically stronger prompt adherence than isotropic Procrustes.

Calibrated on `800` domain-matched MiniMax-H3 video prompts (90/10 held-out prompt split) directly into MiniMax-H3's native **5120-dim `Qwen3-VL-32B`** geometric space.

| Encoder (Key File) | Dims | Method | **Cosine** | **R²** | Size |
|---|---:|:---:|---:|---:|---:|
| [Qwen3.5-9B](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/blob/main/qwen3.5/minimax_h3_32b_pk_ridge_qwen3.5_9b.safetensors) | `4096→5120` | Ridge (`-1`) | **0.9626** | **0.9935** | 80 MB |
| [Qwen3.5-2B](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/blob/main/qwen3.5/minimax_h3_32b_pk_ridge_qwen3.5_2b.safetensors) | `2048→5120` | Ridge (`-1`) | **0.9620** | **0.9949** | 40 MB |
| [Qwen3.5-4B](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/blob/main/qwen3.5/minimax_h3_32b_pk_ridge_qwen3.5_4b.safetensors) | `2560→5120` | Ridge (`-1`) | **0.9502** | **0.9917** | 50 MB |
| [Qwen3.5-0.8B](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/blob/main/qwen3.5/minimax_h3_32b_pk_ridge_qwen3.5_0.8b.safetensors) | `1024→5120` | Ridge (`-1`) | **0.9492** | **0.9931** | 20 MB |
| [Gemma-4-31B](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/blob/main/gemma4/minimax_h3_32b_pk_ridge_gemma4_31b_layer-2.safetensors) | `5376→5120` | Ridge (`-2`) | **0.9254** | **0.8055** | 105 MB |
| [Gemma-4-E4B](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/blob/main/gemma4/minimax_h3_32b_pk_ridge_gemma4_e4b_layer-2.safetensors) | `2560→5120` | Ridge (`-2`) | **0.9204** | **0.7766** | 50 MB |
| [Gemma-4-E2B](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/blob/main/gemma4/minimax_h3_32b_pk_ridge_gemma4_e2b_layer-2.safetensors) | `1536→5120` | Ridge (`-2`) | **0.9045** | **0.7177** | 30 MB |
| [Gemma-4-12B](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/blob/main/gemma4/minimax_h3_32b_pk_ridge_gemma4_12b_layer-2.safetensors) | `3840→5120` | Ridge (`-2`) | **0.8909** | **0.6580** | 75 MB |
| [Qwen3.8-27B](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/blob/main/qwen3.8/minimax_h3_32b_pk_qwen3.8_27b.safetensors) | `5120→5120` | Procrustes (`-1`) | 0.8812 | 0.0338 | 100 MB |

## Testing

```bash
CUDA_VISIBLE_DEVICES="" python -m unittest tests/test_platos_key.py
```

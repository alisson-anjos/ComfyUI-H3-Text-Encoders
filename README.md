# ComfyUI-H3-Text-Encoders

> [!IMPORTANT]
> **NEW (`v0.1.7`) — `v4 Clean Sink-Free Multi-Layer (4-Layer Concat) Ridge` for Google Gemma 4 is LIVE (`26B-A4B MoE: 95.24% cos`, `E4B: 95.14% cos`, `E2B: 94.93% cos`, `12B: 93.42% cos`)**, alongside **`v3 Direct-32B Streaming Ridge`** for all **Qwen 3.5 & Qwen 3.8 27B** encoders (`0.985–0.992 R²`) and the **`Plato's Key Load Any HF Text Encoder (H3)`** node!
> - **Finishing now:** `Inkling-2-Mini` (4-Layer) and `Gemma-4-31B` (4-Layer Clean).
> - **Recommendation:** use **`*_pk_ridge_*.safetensors`** for Qwen 3.5 / 3.8 and **`*_pk_ridge_4layer_*.safetensors`** for Gemma 4 & cross-family models (`strength = 1.0`).

Use modern ComfyUI-native text encoders (**Qwen 3.5 0.8B / 2B / 4B / 9B**, **Qwen 3.8 27B**, **Google Gemma 4**) — or **any HuggingFace LLM/VLM** (like **Inkling-2-Mini** or **Gemma 4 26B-A4B**) — as a conditioning encoder for **MiniMax-H3** (**T2VA / Image-to-Video**, **FL2VA**, and **Ref2VA**) in ComfyUI — **zero gradient training, zero LoRA, and zero external MLP weights**.

Powered by **Plato's Key** (*closed-form Ridge & Multi-Layer Ridge*, inspired by Stanford Hazy Research's [Plato's Key](https://hazyresearch.stanford.edu/blog/2026-11-25-platos-key)), each adapter is a compact `.safetensors` key (~20–157 MB) that projects a donor encoder's hidden states directly into MiniMax-H3's native **5,120-dim `Qwen3-VL-32B`** geometric space:

$$\hat{Y} = (X - \mu_{\text{donor}}) W + \mu_{\text{target}}, \qquad W = (X_c^\top X_c + \lambda I)^{-1} X_c^\top Y_c$$

All downstream MiniMax-H3 nodes (`MiniMaxH3ReferenceToVideo`, `MiniMaxH3ImageToVideo`, `SamplerCustomAdvanced`, `VAEDecode`) work out of the box.

## Key Engineering Breakthroughs

1. **Closed-Form Ridge >> Orthogonal Procrustes on MiniMax-H3:** MiniMax-H3's native 5,120-dim `Qwen3-VL-32B` space (layer 50, unnormalized) has strong anisotropic outlier channels. Rigid isotropic Procrustes stops at `R² ≈ 0.04`, whereas closed-form Ridge scales each channel independently, reaching **`R² = 0.990–0.992`** (**147× lower MSE**) on Qwen 3.5 & Qwen 3.8.
2. **Multi-Layer Plato's Key (`4-Layer Concat`) Unlocks Cross-Family Gemma 4:** Single-layer linear maps across different architectures/tokenizers (Google Gemma 4 → Qwen3-VL-32B) hit a ceiling (~70% cos). Concatenating **4 evenly-spaced transformer layers** (e.g., `[-2, -10, -18, -26]` → `10,240 → 5,120`) before a single closed-form Ridge projection jumps Gemma 4 to **`93.4–95.2%` cosine** (`26B-A4B MoE: 95.24%`, `E4B: 95.14%`, `E2B: 94.93%`, `12B: 93.42%`) with **zero gradient training and zero extra forward passes** in ComfyUI!
3. **Token-0 Attention Sink Filtering (`v4` Clean 20k):** Unnormalized `Qwen3-VL-32B` layer 50 puts a massive attention sink on token 0 (`rms ≈ 18.25, max > 3,000` vs `rms ≈ 3.42` for normal text). Dropping bin 0 during cross-family bin-pooled calibration eliminates a 33× Frobenius blow-up in `W`, cutting MSE by **42×** (`59.2 → 1.40`) and lifting `Gemma 4 E4B` to **`95.14%` cosine (`p10 = 87.91%`)**!
4. **Exact ComfyUI CLIP Parity (`v0.1.5+`):** Automatically enforces `options_default` tuple override, layer & `RMSNorm` parity, `llama_template="{}"`, LTXV `min_length=1` padding strip, Gemma 4 `<bos>` attention-sink retention (stripping token 0 post-encode), and `minimax_token_tags`.
5. **Universal HuggingFace Loader (`v0.1.6`):** **`Plato's Key Load Any HF Text Encoder (H3)`** loads *any* `transformers` LLM/VLM (even models without a native ComfyUI `CLIPLoader`, like `thinkingmachines/Inkling-2-Mini`) with automatic single- or multi-layer projection and automatic CPU offload to keep 100% of VRAM free for the H3 DiT.

## Visual Comparison (Zero-Gradient Ridge, No LoRA, No MLP)

[![MiniMax-H3 Plato's Key Explainer & 3-Way Comparison](showcase/H3_PlatosKey_X_Demo_Final_thumb.jpg)](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/showcase/H3_PlatosKey_X_Demo_Final.mp4)

- [Watch / download full explainer + comparison video (`showcase/H3_PlatosKey_X_Demo_Final.mp4`)](showcase/H3_PlatosKey_X_Demo_Final.mp4)

### 4-Way Ref2VA — Native 32B vs Qwen 3.5 9B/2B vs Google Gemma 4 E4B (4-Layer, `strength = 1.0`)

![4-Way Comparison including Google Gemma 4 E4B 4-Layer](showcase/H3_PlatosKey_4Way_With_Gemma4_4Layer.jpg)

## Pre-Calibrated Weights & Direct Download Links

All `.safetensors` Plato's Key adapters live in [**`Alissonerdx/TE-PK-Minimax-H3`**](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3) (calibrated on 20,029 prompts directly to Native `Qwen3-VL-32B` Layer 50).

### Text Encoder Downloads & Matching Plato's Keys

| Encoder | ComfyUI Loader | Text Encoder Download (HuggingFace) | Method | **Val Cos** | **Key File (`Alissonerdx/TE-PK-Minimax-H3`)** |
|---|---|---|:---:|---:|---|
| **Qwen 3.5 9B** | `CLIPLoader` + `PlatosKeyPatchCLIP` | [**`Comfy-Org/Qwen3.5`**](https://huggingface.co/Comfy-Org/Qwen3.5/tree/main/text_encoders) (`qwen3.5_9b_int8_convrot.safetensors` `9.8 GB` / `bf16` `19.3 GB`) | Ridge (`-1`) | **93.05%** (`R² 0.9905`) | [`minimax_h3_32b_pk_ridge_qwen3.5_9b.safetensors`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/qwen3.5/minimax_h3_32b_pk_ridge_qwen3.5_9b.safetensors) |
| **Qwen 3.5 2B** | `CLIPLoader` + `PlatosKeyPatchCLIP` | [**`Comfy-Org/Qwen3.5`**](https://huggingface.co/Comfy-Org/Qwen3.5/tree/main/text_encoders) (`qwen3.5_2b_bf16.safetensors` `4.6 GB`) | Ridge (`-1`) | **92.57%** (`R² 0.9916`) | [`minimax_h3_32b_pk_ridge_qwen3.5_2b.safetensors`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/qwen3.5/minimax_h3_32b_pk_ridge_qwen3.5_2b.safetensors) |
| **Qwen 3.5 4B** | `CLIPLoader` + `PlatosKeyPatchCLIP` | [**`Comfy-Org/Qwen3.5`**](https://huggingface.co/Comfy-Org/Qwen3.5/tree/main/text_encoders) (`qwen3.5_4b_bf16.safetensors` `9.3 GB`) | Ridge (`-1`) | **91.82%** (`R² 0.9902`) | [`minimax_h3_32b_pk_ridge_qwen3.5_4b.safetensors`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/qwen3.5/minimax_h3_32b_pk_ridge_qwen3.5_4b.safetensors) |
| **Qwen 3.5 0.8B** | `PlatosKeyHFTextEncoderLoader` | [**`Qwen/Qwen3.5-0.8B`**](https://huggingface.co/Qwen/Qwen3.5-0.8B) | Ridge (`-1`) | **91.11%** (`R² 0.9901`) | [`minimax_h3_32b_pk_ridge_qwen3.5_0.8b.safetensors`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/qwen3.5/minimax_h3_32b_pk_ridge_qwen3.5_0.8b.safetensors) |
| **Qwen 3.8 27B** | `CLIPLoader` + `PlatosKeyPatchCLIP` | [**`Comfy-Org/Qwen3.8-27B`**](https://huggingface.co/Comfy-Org/Qwen3.8-27B/tree/main/text_encoders) (`qwen3.8_27b_w4a8.safetensors` `17.3 GB`) | Ridge (`-1`) | **89.94%** (`R² 0.9846`) | [`minimax_h3_32b_pk_ridge_qwen3.8_27b.safetensors`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/qwen3.8/minimax_h3_32b_pk_ridge_qwen3.8_27b.safetensors) |
| **Gemma 4 26B-A4B (MoE)** | `PlatosKeyHFTextEncoderLoader` | [**`google/gemma-4-26B-A4B-it`**](https://huggingface.co/google/gemma-4-26B-A4B-it) | **4L Ridge** | **95.24%** (`p10 87.9%`) | [`minimax_h3_32b_pk_ridge_4layer_gemma4_26b_a4b.safetensors`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_ridge_4layer_gemma4_26b_a4b.safetensors) |
| **Gemma 4 E4B** ⭐ | `CLIPLoader` + `PlatosKeyPatchCLIP` | [**`Comfy-Org/gemma-4`**](https://huggingface.co/Comfy-Org/gemma-4/tree/main/text_encoders) (`gemma4_e4b_it_int8_convrot.safetensors` `8.1 GB` / `fp8` `9.1 GB` / `bf16` `16.0 GB`) | **4L Ridge** | **95.14%** (`p10 87.9%`) | [`minimax_h3_32b_pk_ridge_4layer_gemma4_e4b.safetensors`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_ridge_4layer_gemma4_e4b.safetensors) |
| **Gemma 4 E2B** ⭐ | `CLIPLoader` + `PlatosKeyPatchCLIP` | [**`Comfy-Org/gemma-4`**](https://huggingface.co/Comfy-Org/gemma-4/tree/main/text_encoders) (`gemma4_e2b_it_int8_convrot.safetensors` `5.2 GB` / `bf16` `10.3 GB`) | **4L Ridge** | **94.93%** (`p10 87.5%`) | [`minimax_h3_32b_pk_ridge_4layer_gemma4_e2b.safetensors`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_ridge_4layer_gemma4_e2b.safetensors) |
| **Gemma 4 12B** | `CLIPLoader` + `PlatosKeyPatchCLIP` | [**`Comfy-Org/gemma-4`**](https://huggingface.co/Comfy-Org/gemma-4/tree/main/text_encoders) (`gemma4_12b_int8_convrot.safetensors` `12.1 GB` / `bf16` `24.0 GB`) | **4L Ridge** | **93.42%** (`p10 82.8%`) | [`minimax_h3_32b_pk_ridge_4layer_gemma4_12b.safetensors`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_ridge_4layer_gemma4_12b.safetensors) |
| **Gemma 4 31B** | `PlatosKeyHFTextEncoderLoader` | [**`google/gemma-4-31B-it`**](https://huggingface.co/google/gemma-4-31B-it) | Ridge (`-2` / 4L soon) | **75.83%** (1L) | [`minimax_h3_32b_pk_ridge_gemma4_31b_layer-2.safetensors`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_ridge_gemma4_31b_layer-2.safetensors) |
| **Inkling-2-Mini** | `PlatosKeyHFTextEncoderLoader` | [**`thinkingmachines/Inkling-2-Mini`**](https://huggingface.co/thinkingmachines/Inkling-2-Mini) | **4L Ridge** | *calibrating* | *uploading soon* |

> [!TIP]
> **Plug & Play recommendation:** For native ComfyUI `CLIPLoader` (single `.safetensors` in `ComfyUI/models/text_encoders/`), start with **`Gemma 4 E4B`** (`8.1 GB int8` or `9.1 GB fp8`, **95.14% cos**), **`Gemma 4 E2B`** (`5.2 GB int8`, **94.93% cos**), or **`Qwen 3.5 9B / 2B`**. For sharded HuggingFace models (`google/gemma-4-31B-it`, `google/gemma-4-26B-A4B-it`, `thinkingmachines/Inkling-2-Mini`), use our **`Plato's Key Load Any HF Text Encoder (H3)`** node directly.

## Step-by-Step Setup (Where to Place Files)

1. **Install this custom node**:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/alisson-anjos/ComfyUI-H3-Text-Encoders.git
cd ComfyUI-H3-Text-Encoders
pip install -r requirements.txt
```

2. **Download a Text Encoder** from the table above and place single-file `.safetensors` models in **`ComfyUI/models/text_encoders/`** (or skip manual downloads for any HuggingFace repo by using **`Plato's Key Load Any HF Text Encoder (H3)`**).

3. **Download the matching `.safetensors` key** from [**`Alissonerdx/TE-PK-Minimax-H3`**](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3) and place it in **`ComfyUI/models/clip_projections/`**:
```text
ComfyUI/models/
├── text_encoders/
│   ├── qwen3.5_9b_int8_convrot.safetensors
│   └── gemma4_e4b_it_fp8_scaled.safetensors
└── clip_projections/
    ├── minimax_h3_32b_pk_ridge_qwen3.5_9b.safetensors
    └── minimax_h3_32b_pk_ridge_4layer_gemma4_e4b.safetensors
```

4. **Restart ComfyUI** and wire:
   - **Native ComfyUI encoders:** `Load CLIP` → **`Plato's Key Patch CLIP (H3 Plug & Play)`** (`strength = 1.00`, `apply_to_pooled = False`) → `MiniMax H3 Reference to Video` / `MiniMax H3 Image to Video`.
   - **Any HuggingFace model:** **`Plato's Key Load Any HF Text Encoder (H3)`** (`model_id`, `key_file`, `offload_to_cpu = True`) → `MiniMax H3 Reference to Video` / `MiniMax H3 Image to Video`.

### Included ComfyUI Nodes (`conditioning/platos_key`)

- **`Plato's Key Patch CLIP (H3 Plug & Play)`**: Drop-in wrapper for native ComfyUI `CLIPLoader` (auto layer, multi-layer `donor_layers`, `RMSNorm`, `<bos>`, and `minimax_token_tags` parity).
- **`Plato's Key Load Any HF Text Encoder (H3)`**: Loads any `transformers` model from HuggingFace or disk, applies single- or multi-layer Plato's Key, and auto-offloads to CPU after encoding.
- **`Plato's Key Apply (Procrustes)`**: Projects an existing `CONDITIONING` tensor directly.
- **`Plato's Key CLIP Text Encode`**: One-step text encode + Plato's Key projection.

## Calibration & Conversion Script (Create Your Own Key — Zero Gradients)

Convert any HuggingFace text encoder in minutes with [`calibrate_platos_key.py`](calibrate_platos_key.py) (`--method ridge` by default, with automatic held-out `λ` selection):

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

## License

Apache 2.0 — see [LICENSE](LICENSE). Inspired by Stanford Hazy Research's [Plato's Key](https://hazyresearch.stanford.edu/blog/2026-11-25-platos-key).

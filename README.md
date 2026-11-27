# ComfyUI-H3-Text-Encoders

> [!IMPORTANT]
> **NEW (`v0.3.0` — `pk_lora`: Plato's Key + Forced Qwen-Segmentation + Bundled TE LoRA) — 1 Single `.safetensors`, 1 Single Text Encoder, ZERO DiT Changes!**
> - **Major Breakthrough for Cross-Family Speech & Multi-Shot:** Pure linear cross-tokenizer keys (`charov` / `4layer`) hit a ceiling on complex spoken dialogue & multi-shot scenes (`cos ≈ 0.90`). Our new **`*_pk_lora_*.safetensors`** unified adapters combine **Forced Qwen-Token Segmentation (94.74% exact 1:1 token match)** + **Multi-Layer Plato's Key** + a **lightweight Rank-32 Text Encoder LoRA** (all bundled in **1 single `.safetensors` file**, works on `bf16` / `fp8` / `int8` / `nvfp4`, `0` permanent VRAM) — jumping live 160-token 3-shot cosine on `Gemma 4 E4B` from **`0.901 → 0.976` (`+30.9 pts` on worst token)** and unlocking Native-32B-matching 3-shot visuals & clean spoken dialogue!
> - **Recommendation:**
>   - **Qwen 3.5 / Qwen 3.8:** use **`*_pk_ridge_*.safetensors`** (`R² 0.985–0.992`, 1:1 shared BPE tokenizer, zero training).
>   - **Gemma 4 / Inkling-2-Mini:** use **`*_pk_lora_*.safetensors`** *(Best overall — speech + multi-shot, requires node **`v0.3.0+`**)*, or **`*_pk_ridge_charov_*.safetensors`** *(100% zero-training linear, `v0.2.0+`)*, or **`*_pk_ridge_4layer_*.safetensors`** *(word-span pooled, `v0.1.8+`)*. All variants stay live side-by-side at `strength = 1.0`.

> [!WARNING]
> **Update your custom node (`v0.3.0+` for `pk_lora`, `v0.2.0+` for `charov`)!** Run `cd ComfyUI/custom_nodes/ComfyUI-H3-Text-Encoders && git pull` (or update via ComfyUI Manager) and restart ComfyUI. `v0.3.0` automatically detects `alignment="forced_qwen_seg"` and `has_te_lora="true"` inside a single `.safetensors` file and applies temporary forward hooks on any quantization (`bf16`, `fp8_scaled`, `int8_convrot`, `nvfp4`) with zero VRAM leak.

> [!NOTE]
> **Honest Note on Spoken Dialogue & Multi-Shot Scenes (Cross-Family Encoders):**
> - **For 100%-crisp spoken dialogue, lip-sync & complex multi-shot choreography:** **Qwen 3.5 (`9B` / `2B` / `4B` / `0.8B`)** and **Qwen 3.8 `27B`** remain **#1**, because they share the exact BPE tokenizer 1:1 with native `Qwen3-VL-32B`.
> - **Cross-family `charov` keys (`Gemma 4` / `Inkling-2-Mini`):** deliver a **major leap over `4layer`** on spoken dialogue intelligibility and multi-shot scene transitions (for example, `Gemma 4 E4B` jumped from `79.7%` to **`93.6%` live token-by-token cosine** in ComfyUI `fp8_scaled`, turning garbled audio into clear English sentences). However, a purely linear cross-tokenizer projection still does not 100% match native 32B on every complex multi-shot or fast-dialogue prompt — a lightweight, single-encoder residual refiner is in active R&D to close the remaining gap.

Use modern ComfyUI-native text encoders (**Qwen 3.5 0.8B / 2B / 4B / 9B**, **Qwen 3.8 27B**, **Google Gemma 4**) — or **any HuggingFace LLM/VLM** (like **Inkling-2-Mini** or **Gemma 4 26B-A4B**) — as a conditioning encoder for **MiniMax-H3** (**T2VA / Image-to-Video**, **FL2VA**, and **Ref2VA**) in ComfyUI — **zero gradient training, zero LoRA, and zero external MLP weights**.

Powered by **Plato's Key** (*closed-form Ridge & Multi-Layer Ridge*, inspired by Stanford Hazy Research's [Plato's Key](https://hazyresearch.stanford.edu/blog/2026-11-25-platos-key)), each adapter is a compact `.safetensors` key (~20–275 MB) that projects a donor encoder's hidden states directly into MiniMax-H3's native **5,120-dim `Qwen3-VL-32B`** geometric space:

$$\hat{Y} = (X - \mu_{\text{donor}}) W + \mu_{\text{target}}, \qquad W = (X_c^\top X_c + \lambda I)^{-1} X_c^\top Y_c$$

All downstream MiniMax-H3 nodes (`MiniMaxH3ReferenceToVideo`, `MiniMaxH3ImageToVideo`, `SamplerCustomAdvanced`, `VAEDecode`) work out of the box.

## How `pk_lora` (`v0.3.0`) & `charov` (`v0.2.0`) Work

Cross-family models tokenize the same prompt into completely different subword sequences (for example, Gemma 4 SentencePiece vs Qwen3 BPE).

- **`pk_lora` (`v0.3.0+` — Recommended for Speech & Multi-Shot):** First tokenizes the prompt with Qwen3 BPE, then encodes each Qwen subword separately with the donor tokenizer (**Forced Qwen-Token Segmentation** — **94.74% exact 1:1 match** on Gemma 4, zero cross-boundary bleed), applies a bundled **Rank-32 Text Encoder LoRA** via zero-leak forward hooks (works on `bf16` / `fp8` / `int8` / `nvfp4`), and projects 6 concatenated layers via **Multi-Layer Plato's Key** trained jointly with MiniMax-H3's `condition_proj` loss (`5×` weight on spoken dialogue quotes). Everything ships inside **1 unified `.safetensors` file** — still **1 single Text Encoder** and **zero DiT changes**!

![Plato's Key + Forced Qwen Segmentation + Bundled TE LoRA Architecture](showcase/H3_PlatosKey_v030_PK_LoRA_Diagram.png)

### 3-Way Multi-Shot + Spoken Dialogue Showcase (`Native 32B` vs `charov` Linear vs `PK + LoRA`)

[![MiniMax-H3 3-Way PK+LoRA Multi-Shot + Speech Showcase](showcase/H3_PlatosKey_PK_LoRA_3Way_Showcase_thumb.jpg)](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/showcase/H3_PlatosKey_PK_LoRA_3Way_Showcase.mp4)

- [Watch / download the 3-way Multi-Shot + Speech comparison (`showcase/H3_PlatosKey_PK_LoRA_3Way_Showcase.mp4`)](showcase/H3_PlatosKey_PK_LoRA_3Way_Showcase.mp4) — **fixed seed, 1 single Text Encoder, zero DiT changes**, showing **`Native Qwen3-VL-32B`** | **`Gemma 4 E4B charov (Linear, cos 0.901)`** | **`Gemma 4 E4B PK + LoRA (cos 0.976, AUDIO)`** executing all 3 shots (sunlit café + spoken dialogue → grab jacket & walk to exit → golden cobblestone street).

- **`charov` (`v0.2.0+` — 100% Zero-Training Linear):** Uses a deterministic **character-overlap resampling matrix** $A \in \mathbb{R}^{T_{\text{qwen}} \times T_{\text{donor}}}$ (row-stochastic, $A \mathbf{1} = \mathbf{1}$) that **commutes exactly** with closed-form Ridge:

$$A \cdot \big((H - \mu_{\text{donor}}) W + \mu_{\text{target}}\big) \equiv (A H - \mu_{\text{donor}}) W + \mu_{\text{target}}$$

In ComfyUI `v0.2.0+`, the node runs this resampling in `<1ms` on CPU (`0 VRAM`). See [`CHAR_OVERLAP_V6.md`](CHAR_OVERLAP_V6.md) for details.

![Plato's Key Char-Overlap Alignment Diagram](showcase/H3_PlatosKey_v6_CharOverlap_Diagram.png)

![Plato's Key Char-Overlap Pipeline Flow](showcase/H3_PlatosKey_v6_Pipeline_Flow.png)

## Key Engineering Breakthroughs

1. **Closed-Form Ridge >> Orthogonal Procrustes on MiniMax-H3:** MiniMax-H3's native 5,120-dim `Qwen3-VL-32B` space (layer 50, unnormalized) has strong anisotropic outlier channels. Rigid isotropic Procrustes stops at `R² ≈ 0.04`, whereas closed-form Ridge scales each channel independently, reaching **`R² = 0.990–0.992`** (**147× lower MSE**) on Qwen 3.5 & Qwen 3.8.
2. **Exact Token-Level `charov` (`A @ H`) + BOS Parity (`v0.2.0`):** Calibrating directly on exact individual Qwen3-VL-32B layer 49 tokens via character-overlap matrix `A @ H` — with explicit `<bos>` parity, ComfyUI-matched intermediate layers (replacing divergent `final_norm`), and quantization-robust FP64 Ridge solving — lifts live ComfyUI `fp8_scaled` token-by-token cosine on `Gemma 4 E4B` from `79.7%` to **`93.6%` (`p10 = 90.4%`)** with **zero gradient training**. *(Note: `charov` `R²` is measured on exact individual subword tokens, which is a much stricter metric than word-averaged `4layer` `R²`.)*
3. **Word-Span Aligned Multi-Layer Plato's Key (`4layer`):** Concatenating 5–6 transformer layers including `layer 0` (`offset_mapping` word-span alignment) remains available alongside `charov` for users on `v0.1.8+` or comparing styles.
4. **Token-0 Attention Sink Filtering & Exact ComfyUI CLIP Parity:** Automatically enforces `options_default` tuple override, layer & `RMSNorm` parity, `llama_template="{}"`, LTXV `min_length=1` padding strip, Gemma 4 `<bos>` attention-sink retention, and `minimax_token_tags`.
5. **Universal HuggingFace Loader:** **`Plato's Key Load Any HF Text Encoder (H3)`** loads *any* `transformers` LLM/VLM (even models without a native ComfyUI `CLIPLoader`, like `thinkingmachines/Inkling-2-Mini`) with automatic single-layer, multi-layer, and `charov` support plus CPU offload.

## Visual Comparison (Zero-Gradient Ridge, No LoRA, No MLP)

[![MiniMax-H3 Plato's Key Explainer & 3-Way Comparison](showcase/H3_PlatosKey_X_Demo_Final_thumb.jpg)](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/showcase/H3_PlatosKey_X_Demo_Final.mp4)

- [Watch / download full explainer + comparison video (`showcase/H3_PlatosKey_X_Demo_Final.mp4`)](showcase/H3_PlatosKey_X_Demo_Final.mp4)

### 4-Way Ref2VA — Native 32B vs Qwen 3.5 9B/2B vs Google Gemma 4 E4B (4-Layer, `strength = 1.0`)

![4-Way Comparison including Google Gemma 4 E4B 4-Layer](showcase/H3_PlatosKey_4Way_With_Gemma4_4Layer.jpg)

### Multi-Layer 4-Way Video Showcase (Gemma 4 Before/After + Parisian Café Ref2VA + Swiss Alps Train T2VA)

[![Multi-Layer 4-Way Showcase Video](showcase/H3_PlatosKey_MultiLayer_4Way_Showcase_thumb.jpg)](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/showcase/H3_PlatosKey_MultiLayer_4Way_Showcase.mp4)

- [Watch / download the 3-segment 4-way comparison video (`showcase/H3_PlatosKey_MultiLayer_4Way_Showcase.mp4`)](showcase/H3_PlatosKey_MultiLayer_4Way_Showcase.mp4) — featuring **Gemma 4 E4B 1-Layer vs 4-Layer Clean**, **Parisian Café Ref2VA (`Native 32B` | `Qwen 3.5 9B` | `Gemma 4 E4B 4L` | `Gemma 4 E2B 4L`)**, and **Swiss Alps Steam Train Pure T2VA (`Native 32B` | `Qwen 3.5 9B` | `Qwen 3.5 2B` | `Gemma 4 E4B 4L`)** at `strength = 1.0`.

### 10-Second 6-Way Action Stress Test (3 Model Families: Qwen + Google Gemma 4 + Inkling-2-Mini)

[![10s 6-Way Action Showcase Video](showcase/H3_PlatosKey_10s_6Way_Action_Showcase_thumb.jpg)](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/showcase/H3_PlatosKey_10s_6Way_Action_Showcase.mp4)

- [Watch / download the 20s 6-way action showcase (`showcase/H3_PlatosKey_10s_6Way_Action_Showcase.mp4`)](showcase/H3_PlatosKey_10s_6Way_Action_Showcase.mp4) — **fixed seed, `strength = 1.0`, full prompts on screen**, comparing **`Native 32B`** | **`Qwen 3.5 9B`** | **`Gemma 4 31B`** | **`Gemma 4 26B-A4B MoE`** | **`Gemma 4 E4B`** | **`Inkling-2-Mini`** across **10s Tokyo Neon Rain (Ref2VA)** and **10s Medieval Blacksmith Forge (Pure T2VA)**.

## Pre-Calibrated Weights & Direct Download Links

All `.safetensors` Plato's Key adapters live in [**`Alissonerdx/TE-PK-Minimax-H3`**](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3).

### Same-Tokenizer Encoders (Qwen 3.5 / Qwen 3.8 — 1:1 Token Ridge)

| Encoder | ComfyUI Loader | Text Encoder Download (HuggingFace) | **Val Metrics (`Cos` / `R²`)** | **Key File (`Alissonerdx/TE-PK-Minimax-H3`)** |
|---|---|---|---:|---|
| **Qwen 3.5 9B** ⭐ | `CLIPLoader` + `PlatosKeyPatchCLIP` | [**`Comfy-Org/Qwen3.5`**](https://huggingface.co/Comfy-Org/Qwen3.5/tree/main/text_encoders) (`qwen3.5_9b_int8_convrot.safetensors` `9.8 GB` / `bf16` `19.3 GB`) | **93.05%** (`R² 0.9905`) | [`minimax_h3_32b_pk_ridge_qwen3.5_9b.safetensors`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/qwen3.5/minimax_h3_32b_pk_ridge_qwen3.5_9b.safetensors) |
| **Qwen 3.5 2B** | `CLIPLoader` + `PlatosKeyPatchCLIP` | [**`Comfy-Org/Qwen3.5`**](https://huggingface.co/Comfy-Org/Qwen3.5/tree/main/text_encoders) (`qwen3.5_2b_bf16.safetensors` `4.6 GB`) | **92.57%** (`R² 0.9916`) | [`minimax_h3_32b_pk_ridge_qwen3.5_2b.safetensors`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/qwen3.5/minimax_h3_32b_pk_ridge_qwen3.5_2b.safetensors) |
| **Qwen 3.5 4B** | `CLIPLoader` + `PlatosKeyPatchCLIP` | [**`Comfy-Org/Qwen3.5`**](https://huggingface.co/Comfy-Org/Qwen3.5/tree/main/text_encoders) (`qwen3.5_4b_bf16.safetensors` `9.3 GB`) | **91.82%** (`R² 0.9902`) | [`minimax_h3_32b_pk_ridge_qwen3.5_4b.safetensors`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/qwen3.5/minimax_h3_32b_pk_ridge_qwen3.5_4b.safetensors) |
| **Qwen 3.5 0.8B** | `PlatosKeyHFTextEncoderLoader` | [**`Qwen/Qwen3.5-0.8B`**](https://huggingface.co/Qwen/Qwen3.5-0.8B) | **91.11%** (`R² 0.9901`) | [`minimax_h3_32b_pk_ridge_qwen3.5_0.8b.safetensors`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/qwen3.5/minimax_h3_32b_pk_ridge_qwen3.5_0.8b.safetensors) |
| **Qwen 3.8 27B** | `CLIPLoader` + `PlatosKeyPatchCLIP` | [**`Comfy-Org/Qwen3.8-27B`**](https://huggingface.co/Comfy-Org/Qwen3.8-27B/tree/main/text_encoders) (`qwen3.8_27b_w4a8.safetensors` `17.3 GB`) | **89.94%** (`R² 0.9846`) | [`minimax_h3_32b_pk_ridge_qwen3.8_27b.safetensors`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/qwen3.8/minimax_h3_32b_pk_ridge_qwen3.8_27b.safetensors) |

### Cross-Family Encoders (`pk_lora` Unified vs `charov` Zero-Training vs `4layer` Legacy)

> **Which file should I download?**
> - **`*_pk_lora_*.safetensors`** *(Best for Speech & Multi-Shot, requires node `v0.3.0+`)*: 1 unified `.safetensors` bundling **Forced Qwen-Segmentation (94.74% 1:1)** + **Multi-Layer Plato's Key** + **Rank-32 TE LoRA**.
> - **`*_pk_ridge_charov_*.safetensors`** *(100% Zero-Training, requires node `v0.2.0+`)*: Exact token-level **Character-Overlap (`A @ H`)** linear Ridge.
> - **`*_pk_ridge_4layer_*.safetensors`** *(Legacy Word-Span, works on `v0.1.8+`)*: Word-span pooled linear Ridge.

| Encoder | ComfyUI Loader | **`pk_lora` Unified (`v0.3.0+`, Best Speech + Multi-Shot)** | **`charov` Zero-Train (`v0.2.0+`)** | **`4layer` Legacy** |
|---|---|---|---|---|
| **Gemma 4 E4B** 🏆⭐ | `CLIPLoader` + `PlatosKeyPatchCLIP` ([`Comfy-Org/gemma-4`](https://huggingface.co/Comfy-Org/gemma-4/tree/main/text_encoders)) | [`pk_lora_gemma4_e4b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_lora_gemma4_e4b.safetensors) (**Live cos 97.6%**, `H3 R² 0.894`, `quotes cos 95.2%`) | [`charov_gemma4_e4b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_ridge_charov_gemma4_e4b.safetensors) (`95.59%` / `R² 0.816`) | [`4layer_gemma4_e4b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_ridge_4layer_gemma4_e4b.safetensors) (`94.86%`) |
| **Gemma 4 E2B** ⭐ | `CLIPLoader` + `PlatosKeyPatchCLIP` ([`Comfy-Org/gemma-4`](https://huggingface.co/Comfy-Org/gemma-4/tree/main/text_encoders)) | *Training now — rolling out shortly!* | [`charov_gemma4_e2b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_ridge_charov_gemma4_e2b.safetensors) (`94.77%` / `R² 0.781`) | [`4layer_gemma4_e2b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_ridge_4layer_gemma4_e2b.safetensors) (`94.46%`) |
| **Gemma 4 12B** | `CLIPLoader` + `PlatosKeyPatchCLIP` ([`Comfy-Org/gemma-4`](https://huggingface.co/Comfy-Org/gemma-4/tree/main/text_encoders)) | *In queue — rolling out shortly!* | [`charov_gemma4_12b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_ridge_charov_gemma4_12b.safetensors) (`95.26%` / `R² 0.805`) | [`4layer_gemma4_12b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_ridge_4layer_gemma4_12b.safetensors) (`93.67%`) |
| **Inkling-2-Mini** | `PlatosKeyHFTextEncoderLoader` ([`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini)) | *In queue — rolling out shortly!* | [`charov_inkling2_mini`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/inkling2/minimax_h3_32b_pk_ridge_charov_inkling2_mini.safetensors) (`93.52%` / `R² 0.731`) | [`4layer_inkling2_mini`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/inkling2/minimax_h3_32b_pk_ridge_4layer_inkling2_mini.safetensors) (`96.23%`) |
| **Gemma 4 26B-A4B (MoE)** | `PlatosKeyHFTextEncoderLoader` ([`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)) | *In queue — rolling out shortly!* | [`charov_gemma4_26b_a4b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_ridge_charov_gemma4_26b_a4b.safetensors) (`95.32%` / `R² 0.808`) | [`4layer_gemma4_26b_a4b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_ridge_4layer_gemma4_26b_a4b.safetensors) (`95.43%`) |
| **Gemma 4 31B** 🏆 | `CLIPLoader` + `PlatosKeyPatchCLIP` ([`text_encoders/`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/tree/main/text_encoders)) | *In queue — rolling out shortly!* | [`charov_gemma4_31b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_ridge_charov_gemma4_31b.safetensors) (`95.90%` / `R² 0.831`) | [`4layer_gemma4_31b`](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3/resolve/main/gemma4/minimax_h3_32b_pk_ridge_4layer_gemma4_31b.safetensors) (`95.97%`) |

> [!TIP]
> **Plug & Play recommendation:** Start with **`Qwen 3.5 9B`** (best all-around + #1 for exact spoken dialogue & multi-shot), or try **`Gemma 4 E4B charov`** (`8.1 GB int8` / `9.1 GB fp8`) or **`Gemma 4 31B charov`** (`19.1 GB nvfp4/w4a8`) for strong cross-family visuals & improved speech!

## Step-by-Step Setup (Where to Place Files)

1. **Install or update this custom node (`v0.2.0+` required for `charov`)**:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/alisson-anjos/ComfyUI-H3-Text-Encoders.git
cd ComfyUI-H3-Text-Encoders
pip install -r requirements.txt
```

2. **Download a Text Encoder** from the tables above and place single-file `.safetensors` models in **`ComfyUI/models/text_encoders/`** (or skip manual downloads for any HuggingFace repo by using **`Plato's Key Load Any HF Text Encoder (H3)`**).

3. **Download the matching `.safetensors` key** from [**`Alissonerdx/TE-PK-Minimax-H3`**](https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3) and place it in **`ComfyUI/models/clip_projections/`**:
```text
ComfyUI/models/
├── text_encoders/
│   ├── qwen3.5_9b_int8_convrot.safetensors
│   └── gemma4_e4b_it_fp8_scaled.safetensors
└── clip_projections/
    ├── minimax_h3_32b_pk_ridge_qwen3.5_9b.safetensors
    └── minimax_h3_32b_pk_ridge_charov_gemma4_e4b.safetensors
```

4. **Restart ComfyUI** and wire:
   - **Native ComfyUI encoders:** `Load CLIP` → **`Plato's Key Patch CLIP (H3 Plug & Play)`** (`strength = 1.00`, `apply_to_pooled = False`) → `MiniMax H3 Reference to Video` / `MiniMax H3 Image to Video`.
   - **Any HuggingFace model:** **`Plato's Key Load Any HF Text Encoder (H3)`** (`model_id`, `key_file`, `offload_to_cpu = True`) → `MiniMax H3 Reference to Video` / `MiniMax H3 Image to Video`.

### Included ComfyUI Nodes (`conditioning/platos_key`)

- **`Plato's Key Patch CLIP (H3 Plug & Play)`**: Drop-in wrapper for native ComfyUI `CLIPLoader` (auto layer, multi-layer `donor_layers`, **`charov` `A @ H` resampling**, `RMSNorm`, `<bos>`, and `minimax_token_tags` parity).
- **`Plato's Key Load Any HF Text Encoder (H3)`**: Loads any `transformers` model from HuggingFace or disk, applies single-layer, multi-layer, or **`charov`** Plato's Key, and auto-offloads to CPU after encoding.
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

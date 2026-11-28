"""ComfyUI custom nodes for Plato's Key hidden-state transfer."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .platos_key_core import PlatosKey, apply_platos_key, load_platos_key

try:
    import folder_paths  # type: ignore
except ImportError:
    folder_paths = None

LOCAL_KEYS_DIR = Path(__file__).resolve().parent / "keys"
_KEY_CACHE: Dict[Tuple[str, float], PlatosKey] = {}
_LORA_CACHE: Dict[Tuple[str, float], Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], float]] = {}
_VISION_CACHE: Dict[Tuple[str, float], Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
_QWEN_TARGET_TOK: List[Any] = [None]
_H3_TOKEN0_SINK_CACHE: List[Optional[torch.Tensor]] = [None]
_H3_TOKEN0_SINK_LOADED: List[bool] = [False]
_VISION_WARNED: List[bool] = [False]


def _load_vision_proj(path: Path) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Load optional bundled vision projection (`vision_R`, `vision_donor_mean`, `vision_target_mean`)."""
    resolved = str(path.resolve())
    mtime = os.path.getmtime(resolved)
    ck = (resolved, mtime)
    if ck in _VISION_CACHE:
        return _VISION_CACHE[ck]

    from safetensors import safe_open
    res = None
    with safe_open(resolved, framework="pt", device="cpu") as sf:
        ks = set(sf.keys())
        if {"vision_R", "vision_donor_mean", "vision_target_mean"}.issubset(ks):
            res = (
                sf.get_tensor("vision_R"),
                sf.get_tensor("vision_donor_mean"),
                sf.get_tensor("vision_target_mean"),
            )
    for old_k in [k for k in _VISION_CACHE if k[0] == resolved]:
        del _VISION_CACHE[old_k]
    _VISION_CACHE[ck] = res
    return res


def _build_h3_vision_template(
    ref_items: Optional[List[Dict[str, Any]]] = None,
    images: Optional[List[Any]] = None,
    picture_label: str = "<Picture %d>: ",
) -> Tuple[str, List[Any]]:
    """Convert MiniMax-H3 `minimax_ref_items` or `images` into a VLM `llama_template` + flat `images` list."""
    parts: List[str] = []
    vis_imgs: List[Any] = []
    if ref_items:
        img_idx = 0
        aud_idx = 0
        for item in ref_items:
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            if itype == "image" and item.get("data") is not None:
                img_idx += 1
                parts.append(f"<Picture {img_idx}>: <|vision_start|><|image_pad|><|vision_end|>")
                vis_imgs.append(item["data"])
            elif itype == "audio":
                aud_idx += 1
                parts.append(f"<Audio {aud_idx}>: ")
    elif images:
        for i, img in enumerate(images, 1):
            if img is None:
                continue
            lbl = (picture_label % i) if "%d" in picture_label else f"<Picture {i}>: "
            parts.append(f"{lbl}<|vision_start|><|image_pad|><|vision_end|>")
            vis_imgs.append(img)
    return "".join(parts) + "{}", vis_imgs


def _get_h3_token0_sink() -> Optional[torch.Tensor]:
    """Lazily load the 20 KB universal Qwen3-VL-32B L49 Token-0 Attention Sink vector ([5120])."""
    if _H3_TOKEN0_SINK_LOADED[0]:
        return _H3_TOKEN0_SINK_CACHE[0]
    _H3_TOKEN0_SINK_LOADED[0] = True
    sink_path = LOCAL_KEYS_DIR / "h3_qwen32b_token0_sink.safetensors"
    if not sink_path.is_file():
        return None
    try:
        from safetensors import safe_open
        with safe_open(str(sink_path), framework="pt", device="cpu") as sf:
            keys = list(sf.keys())
            k = "token0_sink" if "token0_sink" in keys else (keys[0] if keys else None)
            if k is not None:
                t = sf.get_tensor(k).float().reshape(-1)
                if t.numel() == 5120:
                    _H3_TOKEN0_SINK_CACHE[0] = t
    except Exception as exc:
        print(f"[Plato's Key] Could not load token0_sink: {exc}")
    return _H3_TOKEN0_SINK_CACHE[0]


def _inject_h3_token0_sink(
    cond_list: List[List[Any]],
    d_target: int,
    strength: float = 1.0,
) -> List[List[Any]]:
    """Restore Qwen3-VL-32B's universal Token-0 Attention Sink (rms ≈ 224, cos = 0.99998 across prompts)
    whenever a projected 5120-dim H3 conditioning tensor has an attenuated token 0 (rms < 50).
    Eliminates clipped/unfinished first-word speech at the start of generated videos across ALL keys
    (Qwen 3.5, Qwen 3.8, Gemma 4, Inkling-2-Mini) with 0 training and <0.01ms overhead.
    """
    if d_target != 5120 or strength <= 0.0:
        return cond_list
    sink = _get_h3_token0_sink()
    if sink is None:
        return cond_list
    out: List[List[Any]] = []
    for cond_tensor, meta in cond_list:
        if (
            isinstance(cond_tensor, torch.Tensor)
            and cond_tensor.ndim == 3
            and cond_tensor.shape[1] >= 2
            and cond_tensor.shape[2] == 5120
        ):
            tok0_rms = float(cond_tensor[:, 0, :].float().pow(2).mean().sqrt().item())
            if tok0_rms < 50.0:
                cond_tensor = cond_tensor.clone()
                cond_tensor[:, 0, :] = sink.to(device=cond_tensor.device, dtype=cond_tensor.dtype)
        out.append([cond_tensor, meta])
    return out


def _load_te_lora_pairs(path: Path) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], float]:
    """Load optional bundled Text Encoder LoRA (`te_lora.<module>.lora_{A,B}`) from a unified Plato's Key file."""
    resolved = str(path.resolve())
    mtime = os.path.getmtime(resolved)
    ck = (resolved, mtime)
    if ck in _LORA_CACHE:
        return _LORA_CACHE[ck]

    from safetensors import safe_open
    pairs: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    scale = 1.0
    with safe_open(resolved, framework="pt", device="cpu") as sf:
        meta = dict(sf.metadata() or {})
        if meta.get("has_te_lora", "").lower() == "true":
            rank = float(meta.get("lora_rank", "32"))
            alpha = float(meta.get("lora_alpha", str(rank)))
            scale = alpha / max(rank, 1.0)
            a_map: Dict[str, torch.Tensor] = {}
            b_map: Dict[str, torch.Tensor] = {}
            for k in sf.keys():
                if k.startswith("te_lora."):
                    body = k[len("te_lora."):]
                    if body.endswith(".lora_A"):
                        a_map[body[:-7]] = sf.get_tensor(k)
                    elif body.endswith(".lora_B"):
                        b_map[body[:-7]] = sf.get_tensor(k)
            for mod_name, a_t in a_map.items():
                if mod_name in b_map:
                    pairs[mod_name] = (a_t, b_map[mod_name])
    for old_k in [k for k in _LORA_CACHE if k[0] == resolved]:
        del _LORA_CACHE[old_k]
    _LORA_CACHE[ck] = (pairs, scale)
    return pairs, scale


def _apply_te_lora_hooks(root_module: Any, lora_pairs: Dict[str, Tuple[torch.Tensor, torch.Tensor]], scale: float):
    """Context manager that temporarily applies bundled TE LoRA via forward hooks.
    Works identically on bf16, fp16, fp8_scaled, int8_convrot, w4a8, and nvfp4 without dequantizing base weights.
    """
    from contextlib import contextmanager
    import torch.nn.functional as _F

    @contextmanager
    def _ctx():
        if not lora_pairs or root_module is None or not hasattr(root_module, "named_modules"):
            yield
            return
        skip_parts = ("vision_model", "audio_model", "visual", "vision_tower", "audio_tower")
        handles = []
        for full_name, mod in root_module.named_modules():
            if any(sp in full_name for sp in skip_parts):
                continue
            matched = None
            if full_name in lora_pairs:
                matched = lora_pairs[full_name]
            else:
                for suffix, pair in lora_pairs.items():
                    if full_name.endswith("." + suffix):
                        matched = pair
                        break
            if matched is None:
                continue
            a_cpu, b_cpu = matched

            def _make_hook(a_t, b_t, sc):
                def _hook(_mod, inputs, output):
                    if not inputs or not isinstance(inputs[0], torch.Tensor) or not isinstance(output, torch.Tensor):
                        return output
                    x = inputs[0]
                    a_dev = a_t.to(device=x.device, dtype=x.dtype)
                    b_dev = b_t.to(device=x.device, dtype=x.dtype)
                    delta = _F.linear(_F.linear(x, a_dev), b_dev) * sc
                    return output + delta.to(dtype=output.dtype)
                return _hook

            handles.append(mod.register_forward_hook(_make_hook(a_cpu, b_cpu, scale)))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    return _ctx()


def _unwrap_donor_tokenizer(tok_obj: Any):
    """Unwrap a ComfyUI CLIPTokenizer or HF tokenizer to an object with .encode(piece)."""
    if tok_obj is None:
        return None
    c = tok_obj
    clip_key = getattr(tok_obj, "clip", None)
    if isinstance(clip_key, str) and hasattr(tok_obj, clip_key):
        c = getattr(tok_obj, clip_key)
    while hasattr(c, "tokenizer"):
        c = c.tokenizer
    return c


def _forced_qwen_encode(text: str, donor_tok: Any) -> Optional[Tuple[List[int], List[int]]]:
    """Forced Qwen-Segmentation: tokenize text with Qwen first, then encode each Qwen subword piece
    separately with the donor tokenizer so every Qwen token boundary is an exact donor token boundary.
    Returns (donor_token_ids_without_bos, q_last_indices_0based_without_bos).
    """
    if not text or donor_tok is None:
        return None
    try:
        qtok = _get_qwen_target_tokenizer()
        q_ids = qtok(text, add_special_tokens=False)["input_ids"]
    except Exception:
        return None
    if not q_ids:
        return None

    def _enc_piece(piece: str) -> List[int]:
        try:
            res = donor_tok.encode(piece, add_special_tokens=False)
        except TypeError:
            res = donor_tok.encode(piece)
        return list(res.ids) if hasattr(res, "ids") else list(res)

    d_seq: List[int] = []
    q_last: List[int] = []
    fallback_space = _enc_piece(" ") or [0]
    for tid in q_ids:
        piece = qtok.decode([tid])
        d_sub = _enc_piece(piece) or fallback_space
        d_seq.extend(d_sub)
        q_last.append(len(d_seq) - 1)
    return d_seq, q_last


def _get_qwen_target_tokenizer():
    """Lazily load Qwen3-VL / Qwen2.5 fast tokenizer (0 VRAM, <1ms CPU) for Char-Overlap A@H alignment."""
    if _QWEN_TARGET_TOK[0] is not None:
        return _QWEN_TARGET_TOK[0]
    from transformers import AutoTokenizer

    # Priority 1: ComfyUI's built-in offline qwen25_tokenizer (identical BPE vocab/merges to Qwen3-VL-32B)
    try:
        import comfy.text_encoders as _cte  # type: ignore
        q_dir = Path(_cte.__file__).resolve().parent / "qwen25_tokenizer"
        if q_dir.is_dir() and (q_dir / "vocab.json").is_file():
            _QWEN_TARGET_TOK[0] = AutoTokenizer.from_pretrained(str(q_dir), use_fast=True)
            return _QWEN_TARGET_TOK[0]
    except Exception:
        pass

    # Priority 2: HuggingFace lightweight tokenizer.json (~11 MB, cached locally)
    _QWEN_TARGET_TOK[0] = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-32B-Instruct", use_fast=True)
    return _QWEN_TARGET_TOK[0]


def _char_overlap_matrix(
    q_offsets: List[Tuple[int, int]],
    d_offsets: List[Tuple[int, int]],
) -> torch.Tensor:
    """Build exact [T_q, T_d] row-stochastic character-overlap resampling matrix A."""
    tq, td = len(q_offsets), len(d_offsets)
    A = torch.zeros((tq, td), dtype=torch.float32)
    j_start = 0
    for i, (qs, qe) in enumerate(q_offsets):
        if qe <= qs:
            j = min(td - 1, max(0, int(round(i * (td - 1) / max(1, tq - 1)))))
            A[i, j] = 1.0
            continue
        while j_start < td and d_offsets[j_start][1] <= qs:
            j_start += 1
        tot = 0.0
        j = j_start
        while j < td and d_offsets[j][0] < qe:
            ds, de = d_offsets[j]
            if de > ds:
                ov = min(qe, de) - max(qs, ds)
                if ov > 0:
                    A[i, j] = float(ov)
                    tot += float(ov)
            j += 1
        if tot > 0:
            A[i] /= tot
        else:
            A[i, min(td - 1, max(0, j_start))] = 1.0
    return A


def _extract_donor_offsets(tok_obj: Any, text: str) -> Optional[List[Tuple[int, int]]]:
    """Extract character offset_mapping from a ComfyUI CLIP tokenizer or HF fast tokenizer."""
    if not text or tok_obj is None:
        return None
    candidates = [tok_obj]
    clip_key = getattr(tok_obj, "clip", None)
    if isinstance(clip_key, str) and hasattr(tok_obj, clip_key):
        candidates.insert(0, getattr(tok_obj, clip_key))
    for c in candidates:
        inner = getattr(c, "tokenizer", c)
        # ComfyUI _Gemma4Tokenizer wraps a Rust tokenizers.Tokenizer at inner.tokenizer
        rust_tok = getattr(inner, "tokenizer", None)
        if rust_tok is not None and hasattr(rust_tok, "encode"):
            try:
                enc = rust_tok.encode(text, add_special_tokens=False)
                if hasattr(enc, "offsets") and enc.offsets:
                    return [(int(s), int(e)) for s, e in enc.offsets]
            except Exception:
                pass
        # Fast HuggingFace tokenizer
        try:
            enc = inner(text, return_offsets_mapping=True, add_special_tokens=False)
            if isinstance(enc, dict) and "offset_mapping" in enc and enc["offset_mapping"]:
                return [(int(s), int(e)) for s, e in enc["offset_mapping"]]
        except Exception:
            pass
    return None


def _resample_char_overlap_cond(
    cond_list: List[List[Any]],
    text: str,
    donor_offsets: Optional[List[Tuple[int, int]]],
) -> List[List[Any]]:
    """Resample projected [B, T_donor, D] conditioning onto exact Qwen3-VL subword grid via A @ H."""
    if not text or not donor_offsets:
        return cond_list
    try:
        q_tok = _get_qwen_target_tokenizer()
        q_enc = q_tok(text, return_offsets_mapping=True, add_special_tokens=False)
        q_offsets = [(int(s), int(e)) for s, e in q_enc["offset_mapping"]]
    except Exception as exc:
        print(f"[Plato's Key] Char-Overlap tokenizer skipped: {exc}")
        return cond_list
    if not q_offsets:
        return cond_list

    A = _char_overlap_matrix(q_offsets, donor_offsets)  # [Tq, Td]
    out: List[List[Any]] = []
    for cond_tensor, meta in cond_list:
        new_meta = dict(meta) if isinstance(meta, dict) else {}
        if isinstance(cond_tensor, torch.Tensor) and cond_tensor.ndim == 3 and cond_tensor.shape[1] == len(donor_offsets):
            A_dev = A.to(device=cond_tensor.device, dtype=cond_tensor.dtype)
            cond_tensor = torch.matmul(A_dev, cond_tensor)  # [B, Tq, D]
            tq = cond_tensor.shape[1]
            if "attention_mask" in new_meta and isinstance(new_meta["attention_mask"], torch.Tensor):
                new_meta["attention_mask"] = torch.ones((cond_tensor.shape[0], tq), dtype=new_meta["attention_mask"].dtype, device=new_meta["attention_mask"].device)
            if "minimax_token_tags" in new_meta:
                new_meta["minimax_token_tags"] = torch.ones((cond_tensor.shape[0], tq), dtype=torch.long)
        out.append([cond_tensor, new_meta])
    return out


def _get_search_dirs() -> List[Path]:
    dirs: List[Path] = []
    if folder_paths is not None:
        try:
            dirs.append(Path(folder_paths.models_dir) / "clip_projections")
            dirs.append(Path(folder_paths.models_dir) / "platos_key")
        except Exception:
            pass
    dirs.append(LOCAL_KEYS_DIR)
    return dirs


def discover_key_files() -> List[str]:
    """List available `.safetensors` Plato's Key files across ComfyUI models and local keys/."""
    found: List[str] = []
    seen = set()
    for base in _get_search_dirs():
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*.safetensors")):
            if "token0_sink" in p.name:
                continue
            rel = str(p.relative_to(base))
            if rel not in seen:
                seen.add(rel)
                found.append(rel)
    return found if found else ["(none - set key_path_override)"]


def resolve_key_path(key_file: str, key_path_override: str = "") -> Path:
    """Resolve a Plato's Key file from explicit override path or known search directories."""
    override = (key_path_override or "").strip()
    if override:
        p = Path(override).expanduser()
        if p.is_file():
            return p
        raise FileNotFoundError(f"Plato's Key override path does not exist: {p}")

    direct = Path(key_file).expanduser()
    if direct.is_file():
        return direct

    for base in _get_search_dirs():
        candidate = base / key_file
        if candidate.is_file():
            return candidate

    searched = ", ".join(str(d) for d in _get_search_dirs())
    raise FileNotFoundError(
        f"Plato's Key '{key_file}' not found. Searched: [{searched}]. "
        "Place your .safetensors key in ComfyUI/models/clip_projections/ or set key_path_override."
    )


def get_cached_key(path: Path) -> PlatosKey:
    """Load and cache a Plato's Key by (resolved_path, mtime)."""
    resolved = str(path.resolve())
    mtime = os.path.getmtime(resolved)
    cache_key = (resolved, mtime)
    if cache_key not in _KEY_CACHE:
        # Evict stale entries for the same path
        for k in [k for k in _KEY_CACHE if k[0] == resolved]:
            del _KEY_CACHE[k]
        _KEY_CACHE[cache_key] = load_platos_key(resolved, device="cpu")
    return _KEY_CACHE[cache_key]


def _transform_conditioning(
    conditioning: List[List[Any]],
    key: PlatosKey,
    strength: float = 1.0,
    apply_to_pooled: bool = False,
) -> List[List[Any]]:
    """Apply Plato's Key to every [tensor, dict] entry in a ComfyUI CONDITIONING list."""
    out_conditioning: List[List[Any]] = []
    for cond_tensor, cond_dict in conditioning:
        mapped_tensor = apply_platos_key(cond_tensor, key, strength=strength)
        new_dict = dict(cond_dict) if isinstance(cond_dict, dict) else {}

        if apply_to_pooled and "pooled_output" in new_dict:
            pooled = new_dict["pooled_output"]
            if isinstance(pooled, torch.Tensor) and pooled.shape[-1] == key.d_donor:
                new_dict["pooled_output"] = apply_platos_key(pooled, key, strength=strength)

        out_conditioning.append([mapped_tensor, new_dict])
    return out_conditioning


class PlatosKeyApply:
    """Apply a calibrated Plato's Key (.safetensors) to ComfyUI CONDITIONING embeddings."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "key_file": (discover_key_files(),),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Blend strength. For equal dims: (1-s)*x + s*mapped. For cross-dim: target_mean + s*((x-donor_mean)@R*scale).",
                    },
                ),
            },
            "optional": {
                "key_path_override": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional absolute path to a .safetensors Plato's Key file (overrides key_file dropdown).",
                    },
                ),
                "apply_to_pooled": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Also apply Plato's Key to pooled_output if its dimension matches donor_dim.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/platos_key"

    def apply(
        self,
        conditioning: List[List[Any]],
        key_file: str,
        strength: float = 1.0,
        key_path_override: str = "",
        apply_to_pooled: bool = False,
    ) -> Tuple[List[List[Any]]]:
        key_path = resolve_key_path(key_file, key_path_override)
        key = get_cached_key(key_path)
        mapped = _transform_conditioning(
            conditioning=conditioning,
            key=key,
            strength=float(strength),
            apply_to_pooled=bool(apply_to_pooled),
        )
        return (mapped,)


class PlatosKeyCLIPTextEncode:
    """Encode text with a donor CLIP/text encoder and immediately project via Plato's Key."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "key_file": (discover_key_files(),),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                    },
                ),
            },
            "optional": {
                "key_path_override": ("STRING", {"default": "", "multiline": False}),
                "apply_to_pooled": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "conditioning/platos_key"

    def encode(
        self,
        clip: Any,
        text: str,
        key_file: str,
        strength: float = 1.0,
        key_path_override: str = "",
        apply_to_pooled: bool = False,
    ) -> Tuple[List[List[Any]]]:
        tokens = clip.tokenize(text)
        if hasattr(clip, "encode_from_tokens_scheduled"):
            conditioning = clip.encode_from_tokens_scheduled(tokens)
        else:
            output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
            cond = output.pop("cond")
            conditioning = [[cond, output]]

        key_path = resolve_key_path(key_file, key_path_override)
        key = get_cached_key(key_path)
        mapped = _transform_conditioning(
            conditioning=conditioning,
            key=key,
            strength=float(strength),
            apply_to_pooled=bool(apply_to_pooled),
        )
        return (mapped,)


class PlatosKeyPatchCLIP:
    """Wrap a ComfyUI CLIP object so downstream nodes (like MiniMaxH3ReferenceToVideo)
    automatically receive Plato's Key projected hidden states (e.g. 5120-dim H3 32B space)
    with zero LoRA and zero external projection weights."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "Donor text encoder loaded with ComfyUI's native CLIPLoader (e.g. Qwen 3.5, Qwen 3.8, Gemma 4)."}),
                "key_file": (discover_key_files(), {"tooltip": "Calibrated Plato's Key .safetensors file from ComfyUI/models/clip_projections/."}),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Projection strength (1.0 = exact calibrated map; <1.0 softer prompt conditioning; >1.0 stronger prompt adherence).",
                    },
                ),
            },
            "optional": {
                "key_path_override": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional absolute path to a .safetensors key outside ComfyUI/models/clip_projections/. Leave empty to use key_file.",
                    },
                ),
                "apply_to_pooled": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Also project pooled_output (for SDXL/Flux-style models). Keep False for MiniMax-H3.",
                    },
                ),
                "inject_token0_sink": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Restore Qwen3-VL-32B's universal 20 KB Token-0 Attention Sink vector (ON by default). Eliminates clipped/unfinished first-word speech at the start of generated videos across all H3 keys.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "patch"
    CATEGORY = "conditioning/platos_key"

    def patch(
        self,
        clip: Any,
        key_file: str,
        strength: float = 1.0,
        key_path_override: str = "",
        apply_to_pooled: bool = False,
        inject_token0_sink: bool = True,
    ) -> Tuple[Any]:
        key_path = resolve_key_path(key_file, key_path_override)
        key = get_cached_key(key_path)
        te_lora_pairs, te_lora_scale = _load_te_lora_pairs(key_path)
        vision_proj = _load_vision_proj(key_path)
        s_val = float(strength)
        p_val = bool(apply_to_pooled)
        sink_val = bool(inject_token0_sink)

        patched = clip.clone() if hasattr(clip, "clone") else clip

        multi_layers_str = key.metadata.get("donor_layers", "")
        if multi_layers_str:
            use_final_norm = False
            layer_spec = "all"
        else:
            tap_str = key.metadata.get("donor_layer") or key.metadata.get("tap_layer") or "-1"
            try:
                tap_idx = int(tap_str)
            except ValueError:
                tap_idx = -1
            use_final_norm = (tap_idx == -1)
            layer_spec = "last" if tap_idx == -1 else tap_idx
        if hasattr(patched, "clip_layer") and layer_spec != "all":
            try:
                patched.clip_layer(None if layer_spec == "last" else layer_spec)
            except Exception:
                pass

        def _find_inner_clip_models(csm):
            if csm is None:
                return []
            if hasattr(csm, "layer_norm_hidden_state") or hasattr(csm, "options_default"):
                return [csm]
            found, seen = [], set()
            if hasattr(csm, "modules"):
                for mod in csm.modules():
                    if mod is not csm and id(mod) not in seen:
                        if hasattr(mod, "layer_norm_hidden_state") or hasattr(mod, "options_default"):
                            seen.add(id(mod))
                            found.append(mod)
            return found

        from contextlib import contextmanager

        @contextmanager
        def _override_extraction():
            csm = getattr(patched, "cond_stage_model", None)
            targets = _find_inner_clip_models(csm)
            saved = []
            new_layer = "all" if layer_spec == "all" else ("last" if layer_spec == "last" else "hidden")
            new_layer_idx = None if layer_spec in ("all", "last") else layer_spec
            for t in targets:
                st = {}
                for attr in ("layer", "layer_idx", "layer_norm_hidden_state", "options_default"):
                    if hasattr(t, attr):
                        val = getattr(t, attr)
                        st[attr] = dict(val) if isinstance(val, dict) else val
                saved.append((t, st))
                if hasattr(t, "layer"):
                    t.layer = new_layer
                if hasattr(t, "layer_idx"):
                    t.layer_idx = new_layer_idx
                if hasattr(t, "layer_norm_hidden_state"):
                    t.layer_norm_hidden_state = use_final_norm
                if hasattr(t, "options_default"):
                    od_orig = t.options_default
                    if isinstance(od_orig, tuple) and len(od_orig) >= 2:
                        t.options_default = (new_layer, new_layer_idx) + od_orig[2:]
                    elif isinstance(od_orig, dict):
                        od = dict(od_orig)
                        od["layer"] = new_layer
                        od["layer_idx"] = new_layer_idx
                        od["layer_norm_hidden_state"] = use_final_norm
                        t.options_default = od
            try:
                yield
            finally:
                for t, st in saved:
                    for attr, val in st.items():
                        setattr(t, attr, val)

        in_proj = [False]
        use_char_overlap = (key.metadata.get("alignment") == "char_overlap_qwen")
        use_forced_seg = (key.metadata.get("alignment") == "forced_qwen_seg")
        last_char_ctx: List[Any] = [None, None]  # [text, donor_offsets]
        last_forced_q_last: List[Optional[List[int]]] = [None]  # 0-based indices after BOS strip
        last_vision_active: List[bool] = [False]
        captured_embeds_info: List[List[Dict[str, Any]]] = [[]]

        @contextmanager
        def _capture_vision_embeds_info(csm):
            captured_embeds_info[0] = []
            if not last_vision_active[0] or csm is None or not hasattr(csm, "modules"):
                yield
                return
            handles = []
            for mod in csm.modules():
                if hasattr(mod, "visual") and hasattr(mod, "forward"):
                    def _pre_hook(_m, _args, _kwargs):
                        ei = _kwargs.get("embeds_info")
                        if isinstance(ei, list) and ei:
                            captured_embeds_info[0] = list(ei)
                    handles.append(mod.register_forward_pre_hook(_pre_hook, with_kwargs=True))
            try:
                yield
            finally:
                for h in handles:
                    h.remove()

        orig_tok = getattr(patched, "tokenize", None)
        prepended_bos = [False]
        if orig_tok is not None:
            def _wrapped_tokenize(*args, **kwargs):
                # 2. Match MiniMax-H3 / calibration prompt format:
                #    - no chat template (`llama_template="{}"`)
                #    - disable LTX-style 1024-token left-padding (`min_length = 1`) on Gemma 4
                #    - KEEP `<bos>` (`start_token`) during the forward pass (Gemma 4 needs it as an
                #      attention sink, matching v3 calibration), then strip token 0 after encoding!
                tok_obj = getattr(patched, "tokenizer", None)
                prompt_txt = args[0] if (args and isinstance(args[0], str)) else kwargs.get("text", "")
                ref_items_in = kwargs.get("minimax_ref_items")
                images_in = kwargs.get("images")
                has_ref_visuals = bool(
                    (isinstance(ref_items_in, list) and any(isinstance(it, dict) and it.get("type") == "image" and it.get("data") is not None for it in ref_items_in))
                    or (isinstance(images_in, list) and any(im is not None for im in images_in))
                )
                last_vision_active[0] = False

                # VLM Vision path (v0.4.0+): if the key bundles `vision_R` and the donor supports `images=[...]`,
                # route reference/keyframe images directly through the donor's native Vision Tower!
                if has_ref_visuals and vision_proj is not None:
                    pic_lbl = kwargs.get("picture_label", "<Picture %d>: ")
                    vis_tpl, vis_imgs = _build_h3_vision_template(
                        ref_items=ref_items_in if isinstance(ref_items_in, list) else None,
                        images=images_in if isinstance(images_in, list) else None,
                        picture_label=pic_lbl if isinstance(pic_lbl, str) else "<Picture %d>: ",
                    )
                    if vis_imgs:
                        try:
                            kw_vis = {
                                k: v for k, v in kwargs.items()
                                if k not in ("minimax_ref_items", "images", "picture_label", "llama_template")
                            }
                            kw_vis["llama_template"] = vis_tpl
                            kw_vis["images"] = vis_imgs
                            res_vis = orig_tok(*args, **kw_vis)
                            last_vision_active[0] = True
                            last_forced_q_last[0] = None
                            prepended_bos[0] = False
                            return res_vis
                        except TypeError:
                            pass

                if has_ref_visuals and not _VISION_WARNED[0]:
                    _VISION_WARNED[0] = True
                    print(
                        "[Plato's Key] WARNING: This key/encoder is text-only — VLM vision tokens from "
                        "`minimax_ref_items`/`images` are skipped (Video VAE reference latents & keyframes "
                        "still work in the DiT, but multi-ref/RefPacks identity benefits from a vision-enabled "
                        "key such as Qwen 3.5 pk_lora v0.4.0+ or native Qwen3-VL-32B)."
                    )

                if use_char_overlap:
                    if isinstance(prompt_txt, str) and prompt_txt:
                        last_char_ctx[0] = prompt_txt
                        last_char_ctx[1] = _extract_donor_offsets(tok_obj, prompt_txt)
                    else:
                        last_char_ctx[0] = None
                        last_char_ctx[1] = None

                # Forced Qwen-Segmentation: build exact token dict directly so every Qwen token boundary
                # is a donor token boundary (94.74% 1:1 on Gemma 4!).
                if use_forced_seg and isinstance(prompt_txt, str) and prompt_txt and tok_obj is not None:
                    clip_key = getattr(tok_obj, "clip", None)
                    sub_tok = getattr(tok_obj, clip_key, tok_obj) if isinstance(clip_key, str) else tok_obj
                    donor_raw = _unwrap_donor_tokenizer(tok_obj)
                    forced = _forced_qwen_encode(prompt_txt, donor_raw)
                    if forced is not None:
                        d_seq, q_last = forced
                        start_tok = getattr(sub_tok, "start_token", None)
                        if start_tok is not None:
                            prepended_bos[0] = True
                            token_tuples = [(int(start_tok), 1.0)] + [(int(gid), 1.0) for gid in d_seq]
                        else:
                            prepended_bos[0] = False
                            token_tuples = [(int(gid), 1.0) for gid in d_seq]
                        last_forced_q_last[0] = q_last
                        dict_key = clip_key if isinstance(clip_key, str) else "gemma4"
                        return {dict_key: [token_tuples]}
                last_forced_q_last[0] = None
                saved_tok_attrs = []
                has_bos = False
                if tok_obj is not None:
                    clip_key = getattr(tok_obj, "clip", None)
                    candidates = [tok_obj]
                    if isinstance(clip_key, str) and hasattr(tok_obj, clip_key):
                        candidates.append(getattr(tok_obj, clip_key))
                    for c in candidates:
                        if hasattr(c, "min_length") and isinstance(c.min_length, int) and c.min_length > 1:
                            saved_tok_attrs.append((c, "min_length", c.min_length))
                            c.min_length = 1
                        if hasattr(c, "start_token") and c.start_token is not None:
                            has_bos = True
                prepended_bos[0] = has_bos
                try:
                    kw_no_tpl = dict(kwargs)
                    kw_no_tpl.setdefault("llama_template", "{}")
                    try:
                        return orig_tok(*args, **kw_no_tpl)
                    except TypeError:
                        # Non-MiniMax CLIPs (e.g. Gemma 4, Mistral, T5, generic Qwen) may reject
                        # MiniMax-specific kwargs like `minimax_ref_items`, `images`, or `llama_template`.
                        for drop_keys in (
                            ("minimax_ref_items", "images", "return_word_ids"),
                            ("minimax_ref_items", "images", "return_word_ids", "llama_template"),
                        ):
                            clean_kw = {k: v for k, v in kw_no_tpl.items() if k not in drop_keys}
                            try:
                                return orig_tok(*args, **clean_kw)
                            except TypeError:
                                continue
                        return orig_tok(*args[:1])
                finally:
                    for c, attr_name, old_val in saved_tok_attrs:
                        setattr(c, attr_name, old_val)
            patched.tokenize = _wrapped_tokenize

        def _strip_bos_cond(cond_list):
            if not prepended_bos[0]:
                return cond_list
            stripped = []
            for cond_tensor, meta in cond_list:
                new_meta = dict(meta) if isinstance(meta, dict) else {}
                if isinstance(cond_tensor, torch.Tensor):
                    if cond_tensor.ndim == 3 and cond_tensor.shape[1] > 1:
                        cond_tensor = cond_tensor[:, 1:, :]
                        am = new_meta.get("attention_mask")
                        if isinstance(am, torch.Tensor) and am.ndim >= 2 and am.shape[-1] > 1:
                            new_meta["attention_mask"] = am[..., 1:]
                    elif cond_tensor.ndim == 4 and cond_tensor.shape[2] > 1:
                        cond_tensor = cond_tensor[:, :, 1:, :]
                        am = new_meta.get("attention_mask")
                        if isinstance(am, torch.Tensor) and am.ndim >= 2 and am.shape[-1] > 1:
                            new_meta["attention_mask"] = am[..., 1:]
                stripped.append([cond_tensor, new_meta])
            return stripped

        def _ensure_h3_meta(cond_list):
            # 3. Emit MiniMax-H3 required metadata (`minimax_token_tags`, 5120-dim `pooled_output`)
            out = []
            for cond_tensor, meta in cond_list:
                new_meta = dict(meta) if isinstance(meta, dict) else {}
                if "minimax_token_tags" not in new_meta and isinstance(cond_tensor, torch.Tensor) and cond_tensor.ndim == 3:
                    new_meta["minimax_token_tags"] = torch.ones(
                        (cond_tensor.shape[0], cond_tensor.shape[1]), dtype=torch.long
                    )
                if not p_val and isinstance(cond_tensor, torch.Tensor) and cond_tensor.ndim == 3:
                    pooled = new_meta.get("pooled_output")
                    if pooled is None or (isinstance(pooled, torch.Tensor) and pooled.shape[-1] != key.d_target):
                        new_meta["pooled_output"] = torch.zeros(
                            (cond_tensor.shape[0], key.d_target),
                            dtype=cond_tensor.dtype,
                            device=cond_tensor.device,
                        )
                out.append([cond_tensor, new_meta])
            return out

        def _select_forced_seg_cond(cond_list):
            q_last = last_forced_q_last[0]
            if not use_forced_seg or not q_last:
                return cond_list
            selected = []
            for cond_tensor, meta in cond_list:
                new_meta = dict(meta) if isinstance(meta, dict) else {}
                if isinstance(cond_tensor, torch.Tensor) and cond_tensor.ndim == 3 and cond_tensor.shape[1] > max(q_last):
                    idx = torch.tensor(q_last, dtype=torch.long, device=cond_tensor.device)
                    cond_tensor = cond_tensor.index_select(1, idx)
                    tq = cond_tensor.shape[1]
                    if "attention_mask" in new_meta and isinstance(new_meta["attention_mask"], torch.Tensor):
                        new_meta["attention_mask"] = torch.ones((cond_tensor.shape[0], tq), dtype=new_meta["attention_mask"].dtype, device=new_meta["attention_mask"].device)
                    if "minimax_token_tags" in new_meta:
                        new_meta["minimax_token_tags"] = torch.ones((cond_tensor.shape[0], tq), dtype=torch.long)
                selected.append([cond_tensor, new_meta])
            return selected

        orig_sched = getattr(patched, "encode_from_tokens_scheduled", None)
        if orig_sched is not None:
            def _wrapped_sched(*args, **kwargs):
                if in_proj[0]:
                    return orig_sched(*args, **kwargs)
                in_proj[0] = True
                try:
                    csm = getattr(patched, "cond_stage_model", None)
                    with _override_extraction(), _apply_te_lora_hooks(csm, te_lora_pairs, te_lora_scale), _capture_vision_embeds_info(csm):
                        cond = orig_sched(*args, **kwargs)
                    cond = _strip_bos_cond(cond)
                    if last_vision_active[0] and vision_proj is not None and captured_embeds_info[0]:
                        v_R, v_dm, v_tm = vision_proj
                        mapped = []
                        for cond_tensor, cond_dict in cond:
                            new_dict = dict(cond_dict) if isinstance(cond_dict, dict) else {}
                            # Project text tokens with W_text (`apply_platos_key`)
                            txt_mapped = apply_platos_key(cond_tensor, key, strength=s_val)
                            seq_len = txt_mapped.shape[1]
                            tags = torch.ones(seq_len, dtype=torch.long)
                            for e in captured_embeds_info[0]:
                                if e.get("type") == "image":
                                    tags[max(0, e["index"] - 1):e["index"] + e["size"] + 1] = 0
                            vis_mask = (tags == 0)
                            if vis_mask.any():
                                # Extract exact multi-layer donor hidden states matching key.metadata["donor_layers"]
                                h_raw = cond_tensor
                                layers_meta = key.metadata.get("donor_layers")
                                if h_raw.ndim == 4 and layers_meta:
                                    import json as _json
                                    l_idx = [int(v) for v in _json.loads(layers_meta)]
                                    h_flat = torch.cat([h_raw[:, li] for li in l_idx], dim=-1)
                                else:
                                    h_flat = h_raw
                                if h_flat.shape[-1] == v_R.shape[0]:
                                    hv = h_flat[:, vis_mask, :].float()
                                    v_dm_d = v_dm.to(device=hv.device, dtype=torch.float32)
                                    v_R_d = v_R.to(device=hv.device, dtype=torch.float32)
                                    v_tm_d = v_tm.to(device=hv.device, dtype=torch.float32)
                                    vis_out = ((hv - v_dm_d) @ v_R_d) * s_val + v_tm_d
                                    txt_mapped = txt_mapped.clone()
                                    txt_mapped[:, vis_mask, :] = vis_out.to(dtype=txt_mapped.dtype, device=txt_mapped.device)
                            new_dict["minimax_token_tags"] = tags.unsqueeze(0)
                            mapped.append([txt_mapped, new_dict])
                    else:
                        mapped = _transform_conditioning(cond, key=key, strength=s_val, apply_to_pooled=p_val)
                        if use_forced_seg and last_forced_q_last[0]:
                            mapped = _select_forced_seg_cond(mapped)
                        elif use_char_overlap and last_char_ctx[0] and last_char_ctx[1]:
                            mapped = _resample_char_overlap_cond(mapped, last_char_ctx[0], last_char_ctx[1])
                    if sink_val:
                        mapped = _inject_h3_token0_sink(mapped, key.d_target, strength=s_val)
                    return _ensure_h3_meta(mapped)
                finally:
                    in_proj[0] = False
            patched.encode_from_tokens_scheduled = _wrapped_sched

        orig_enc = getattr(patched, "encode_from_tokens", None)
        if orig_enc is not None:
            def _wrapped_enc(*args, **kwargs):
                if in_proj[0]:
                    return orig_enc(*args, **kwargs)
                in_proj[0] = True
                try:
                    with _override_extraction():
                        out = orig_enc(*args, **kwargs)
                    if isinstance(out, dict) and "cond" in out:
                        out = dict(out)
                        out["cond"] = apply_platos_key(out["cond"], key, strength=s_val)
                        if p_val and isinstance(out.get("pooled_output"), torch.Tensor):
                            out["pooled_output"] = apply_platos_key(out["pooled_output"], key, strength=s_val)
                        return out
                    if isinstance(out, tuple) and len(out) >= 1 and isinstance(out[0], torch.Tensor):
                        c0 = apply_platos_key(out[0], key, strength=s_val)
                        if p_val and len(out) >= 2 and isinstance(out[1], torch.Tensor):
                            return (c0, apply_platos_key(out[1], key, strength=s_val)) + out[2:]
                        return (c0,) + out[1:]
                    if isinstance(out, torch.Tensor):
                        return apply_platos_key(out, key, strength=s_val)
                    return out
                finally:
                    in_proj[0] = False
            patched.encode_from_tokens = _wrapped_enc

        return (patched,)


class _HFPlatosKeyCLIP:
    """ComfyUI CLIP-compatible wrapper around any HuggingFace `transformers` text encoder + Plato's Key.

    Allows running ANY HuggingFace LLM/VLM/embedding model (even ones without a native ComfyUI CLIPLoader)
    directly inside MiniMax-H3 workflows, with automatic single-layer or multi-layer (`donor_layers`)
    projection and optional CPU offload after encoding to keep 100% of VRAM free for the H3 DiT.
    """

    def __init__(
        self,
        model_id: str,
        key: PlatosKey,
        strength: float = 1.0,
        dtype_str: str = "bf16",
        offload_to_cpu: bool = True,
        trust_remote_code: bool = True,
        te_lora_pairs: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        te_lora_scale: float = 1.0,
        inject_token0_sink: bool = True,
    ):
        self.model_id = model_id
        self.key = key
        self.strength = float(strength)
        self.dtype_str = dtype_str
        self.offload_to_cpu = bool(offload_to_cpu)
        self.trust_remote_code = bool(trust_remote_code)
        self.te_lora_pairs = te_lora_pairs or {}
        self.te_lora_scale = float(te_lora_scale)
        self.inject_token0_sink = bool(inject_token0_sink)
        self._tok = None
        self._model = None
        self._backbone = None

    def clone(self):
        c = _HFPlatosKeyCLIP(
            model_id=self.model_id,
            key=self.key,
            strength=self.strength,
            dtype_str=self.dtype_str,
            offload_to_cpu=self.offload_to_cpu,
            trust_remote_code=self.trust_remote_code,
            te_lora_pairs=self.te_lora_pairs,
            te_lora_scale=self.te_lora_scale,
            inject_token0_sink=self.inject_token0_sink,
        )
        c._tok = self._tok
        c._model = self._model
        c._backbone = self._backbone
        return c

    def _ensure_loaded(self):
        if self._backbone is not None:
            return
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        tdtype = dtype_map.get(self.dtype_str, torch.bfloat16)
        self._tok = AutoTokenizer.from_pretrained(self.model_id, use_fast=True, trust_remote_code=self.trust_remote_code)
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token or self._tok.unk_token
        try:
            m = AutoModel.from_pretrained(self.model_id, dtype=tdtype, trust_remote_code=self.trust_remote_code, low_cpu_mem_usage=True)
        except Exception:
            m = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=tdtype, trust_remote_code=self.trust_remote_code, low_cpu_mem_usage=True)
        m.eval()
        m.requires_grad_(False)
        for attr in ("visual", "vision_tower", "audio_tower", "multi_modal_projector", "lm_head"):
            if hasattr(m, "model") and hasattr(m.model, attr):
                delattr(m.model, attr)
            if hasattr(m, attr):
                delattr(m, attr)
        if hasattr(m, "model") and hasattr(m.model, "language_model"):
            bb = m.model.language_model
        elif hasattr(m, "language_model"):
            bb = m.language_model
        elif hasattr(m, "model"):
            bb = m.model
        else:
            bb = m
        self._model = m
        self._backbone = bb

    def tokenize(self, text: str, **kwargs):
        return {"hf_text": str(text)}

    @torch.no_grad()
    def encode_from_tokens_scheduled(self, tokens, uncond_eval=None, **kwargs):
        text = tokens.get("hf_text", "") if isinstance(tokens, dict) else str(tokens)
        self._ensure_loaded()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._backbone.to(device)
        try:
            use_char_ov = (self.key.metadata.get("alignment") == "char_overlap_qwen")
            use_forced_seg = (self.key.metadata.get("alignment") == "forced_qwen_seg")
            if use_forced_seg:
                forced = _forced_qwen_encode(text, self._tok)
                if forced is not None:
                    d_seq, q_last = forced
                    bos_id = getattr(self._tok, "bos_token_id", None)
                    use_bos = (bos_id is not None) and ("gemma" in self.model_id.lower())
                    full_seq = ([int(bos_id)] + d_seq) if use_bos else d_seq
                    offset = 1 if use_bos else 0
                    ids = torch.tensor([full_seq], dtype=torch.long, device=device)
                    mask = torch.ones_like(ids)
                    with _apply_te_lora_hooks(self._backbone, self.te_lora_pairs, self.te_lora_scale):
                        out = self._backbone(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
                    all_hs = out.hidden_states
                    sel = [idx + offset for idx in q_last]
                    layers_meta = self.key.metadata.get("donor_layers")
                    if layers_meta:
                        import json as _json
                        l_idx = [int(v) for v in _json.loads(layers_meta)]
                        h = torch.cat([all_hs[idx][0, sel].float().cpu() for idx in l_idx], dim=-1).unsqueeze(0)
                    else:
                        tap = int(self.key.metadata.get("donor_layer") or self.key.metadata.get("tap_layer") or -1)
                        h = all_hs[tap][0, sel].float().cpu().unsqueeze(0)
                    mapped = apply_platos_key(h, self.key, strength=self.strength)
                    meta = {
                        "pooled_output": torch.zeros((1, self.key.d_target), dtype=mapped.dtype),
                        "minimax_token_tags": torch.ones((1, mapped.shape[1]), dtype=torch.long),
                    }
                    out_c = [[mapped, meta]]
                    return _inject_h3_token0_sink(out_c, self.key.d_target, strength=self.strength) if self.inject_token0_sink else out_c

            tok_kw = dict(return_tensors="pt", truncation=True, max_length=512, return_special_tokens_mask=True)
            if use_char_ov:
                tok_kw["return_offsets_mapping"] = True
            enc = self._tok([text], **tok_kw)
            spec = enc.pop("special_tokens_mask")[0].bool()
            raw_offs = enc.pop("offset_mapping")[0].tolist() if "offset_mapping" in enc else None
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            with _apply_te_lora_hooks(self._backbone, self.te_lora_pairs, self.te_lora_scale):
                out = self._backbone(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
            all_hs = out.hidden_states
            valid = mask[0].bool().cpu() & (~spec)
            if not valid.any():
                valid = mask[0].bool().cpu()

            layers_meta = self.key.metadata.get("donor_layers")
            if layers_meta:
                import json as _json
                l_idx = [int(v) for v in _json.loads(layers_meta)]
                h = torch.cat([all_hs[idx][0].float().cpu()[valid] for idx in l_idx], dim=-1).unsqueeze(0)
            else:
                tap = int(self.key.metadata.get("donor_layer") or self.key.metadata.get("tap_layer") or -1)
                h = all_hs[tap][0].float().cpu()[valid].unsqueeze(0)

            mapped = apply_platos_key(h, self.key, strength=self.strength)
            meta = {
                "pooled_output": torch.zeros((1, self.key.d_target), dtype=mapped.dtype),
                "minimax_token_tags": torch.ones((1, mapped.shape[1]), dtype=torch.long),
            }
            cond_out = [[mapped, meta]]
            if use_char_ov and raw_offs is not None:
                valid_list = valid.tolist()
                donor_offs = [(int(s), int(e)) for (s, e), keep in zip(raw_offs, valid_list) if keep]
                cond_out = _resample_char_overlap_cond(cond_out, text, donor_offs)
            return _inject_h3_token0_sink(cond_out, self.key.d_target, strength=self.strength) if self.inject_token0_sink else cond_out
        finally:
            if self.offload_to_cpu and device.type == "cuda":
                self._backbone.to("cpu")
                torch.cuda.empty_cache()

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False, **kwargs):
        cond_list = self.encode_from_tokens_scheduled(tokens)
        c, meta = cond_list[0]
        if return_dict:
            out = dict(meta)
            out["cond"] = c
            return out
        if return_pooled:
            return c, meta["pooled_output"]
        return c


_HF_PRESETS = [
    "hf:thinkingmachines/Inkling-2-Mini",
    "hf:google/gemma-4-26B-A4B-it",
    "hf:google/gemma-4-31B-it",
    "hf:google/gemma-4-E4B-it",
    "hf:google/gemma-4-E2B-it",
    "hf:Qwen/Qwen3.5-0.8B",
]


def _text_encoder_roots() -> List[Path]:
    roots: List[Path] = []
    if folder_paths is not None:
        for key in ("text_encoders", "clip", "LLM"):
            try:
                for p in folder_paths.get_folder_paths(key):
                    pp = Path(p)
                    if pp.is_dir() and pp not in roots:
                        roots.append(pp)
            except Exception:
                pass
    if not roots:
        fallback = Path.cwd() / "models" / "text_encoders"
        if fallback.is_dir():
            roots.append(fallback)
    return roots


def discover_hf_text_encoders() -> List[str]:
    """Return every folder & model file inside ComfyUI's text_encoders directories + HF presets."""
    found: List[str] = []
    seen = set()
    for root in _text_encoder_roots():
        try:
            for item in sorted(root.iterdir()):
                if item.name.startswith("."):
                    continue
                if item.is_dir():
                    # Direct HF folder (has config.json or safetensors)
                    if (item / "config.json").is_file() or any(item.glob("*.safetensors")):
                        rel = item.name
                        if rel not in seen:
                            seen.add(rel)
                            found.append(rel)
                    else:
                        # One level deeper (e.g. google/gemma-4-31B-it or thinkingmachines/Inkling-2-Mini)
                        for sub in sorted(item.iterdir()):
                            if sub.is_dir() and ((sub / "config.json").is_file() or any(sub.glob("*.safetensors"))):
                                rel = f"{item.name}/{sub.name}"
                                if rel not in seen:
                                    seen.add(rel)
                                    found.append(rel)
                            elif sub.is_file() and sub.suffix.lower() in (".safetensors", ".bin", ".pt"):
                                rel = f"{item.name}/{sub.name}"
                                if rel not in seen:
                                    seen.add(rel)
                                    found.append(rel)
                elif item.is_file() and item.suffix.lower() in (".safetensors", ".bin", ".pt"):
                    rel = item.name
                    if rel not in seen:
                        seen.add(rel)
                        found.append(rel)
        except Exception:
            pass
    for preset in _HF_PRESETS:
        if preset not in seen:
            seen.add(preset)
            found.append(preset)
    return found or list(_HF_PRESETS)


def resolve_hf_text_encoder(selection: str, hf_repo_id_override: str = "") -> str:
    """Resolve a dropdown selection from ComfyUI's text_encoders folder or HF repo preset."""
    if hf_repo_id_override and hf_repo_id_override.strip():
        return hf_repo_id_override.strip()
    sel = (selection or "").strip()
    if sel.startswith("hf:"):
        return sel[3:].strip()
    for root in _text_encoder_roots():
        cand = root / sel
        if cand.exists():
            # If user picked a single .safetensors file inside a HF folder containing config.json, pass its folder
            if cand.is_file() and (cand.parent / "config.json").is_file():
                return str(cand.parent)
            return str(cand)
    return sel


class PlatosKeyHFTextEncoderLoader:
    """Load ANY text encoder from ComfyUI's `text_encoders` dropdown or HuggingFace via `transformers` and wrap it with a calibrated Plato's Key."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "text_encoder": (
                    discover_hf_text_encoders(),
                    {
                        "tooltip": "Select a model/folder from ComfyUI/models/text_encoders/ or an auto-download HuggingFace preset (hf:...).",
                    },
                ),
                "key_file": (discover_key_files(),),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "dtype": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "offload_to_cpu": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Offload text encoder to CPU right after encoding so 100% of VRAM is free for MiniMax-H3 DiT.",
                    },
                ),
            },
            "optional": {
                "hf_repo_id_override": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional: type any custom HuggingFace repo ID not listed in the dropdown.",
                    },
                ),
                "key_path_override": ("STRING", {"default": "", "multiline": False}),
                "inject_token0_sink": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Restore Qwen3-VL-32B's universal 20 KB Token-0 Attention Sink vector (ON by default). Eliminates clipped/unfinished first-word speech at the start of generated videos across all H3 keys.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load"
    CATEGORY = "conditioning/platos_key"

    def load(
        self,
        text_encoder: str,
        key_file: str,
        strength: float = 1.0,
        dtype: str = "bf16",
        offload_to_cpu: bool = True,
        hf_repo_id_override: str = "",
        key_path_override: str = "",
        inject_token0_sink: bool = True,
    ) -> Tuple[Any]:
        resolved_model = resolve_hf_text_encoder(text_encoder, hf_repo_id_override)
        key_path = resolve_key_path(key_file, key_path_override)
        key = get_cached_key(key_path)
        te_lora_pairs, te_lora_scale = _load_te_lora_pairs(key_path)
        clip = _HFPlatosKeyCLIP(
            model_id=resolved_model,
            key=key,
            strength=float(strength),
            dtype_str=dtype,
            offload_to_cpu=bool(offload_to_cpu),
            te_lora_pairs=te_lora_pairs,
            te_lora_scale=te_lora_scale,
            inject_token0_sink=bool(inject_token0_sink),
        )
        return (clip,)


NODE_CLASS_MAPPINGS = {
    "PlatosKeyApply": PlatosKeyApply,
    "PlatosKeyCLIPTextEncode": PlatosKeyCLIPTextEncode,
    "PlatosKeyPatchCLIP": PlatosKeyPatchCLIP,
    "PlatosKeyHFTextEncoderLoader": PlatosKeyHFTextEncoderLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PlatosKeyApply": "Plato's Key Apply (Procrustes)",
    "PlatosKeyCLIPTextEncode": "Plato's Key CLIP Text Encode",
    "PlatosKeyPatchCLIP": "Plato's Key Patch CLIP (H3 Plug & Play)",
    "PlatosKeyHFTextEncoderLoader": "Plato's Key Load Any HF Text Encoder (H3)",
}

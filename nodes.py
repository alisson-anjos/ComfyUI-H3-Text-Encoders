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
    ) -> Tuple[Any]:
        key_path = resolve_key_path(key_file, key_path_override)
        key = get_cached_key(key_path)
        s_val = float(strength)
        p_val = bool(apply_to_pooled)

        patched = clip.clone() if hasattr(clip, "clone") else clip

        orig_sched = getattr(patched, "encode_from_tokens_scheduled", None)
        if orig_sched is not None:
            def _wrapped_sched(*args, **kwargs):
                cond = orig_sched(*args, **kwargs)
                return _transform_conditioning(cond, key=key, strength=s_val, apply_to_pooled=p_val)
            patched.encode_from_tokens_scheduled = _wrapped_sched

        orig_enc = getattr(patched, "encode_from_tokens", None)
        if orig_enc is not None:
            def _wrapped_enc(*args, **kwargs):
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
            patched.encode_from_tokens = _wrapped_enc

        return (patched,)


NODE_CLASS_MAPPINGS = {
    "PlatosKeyApply": PlatosKeyApply,
    "PlatosKeyCLIPTextEncode": PlatosKeyCLIPTextEncode,
    "PlatosKeyPatchCLIP": PlatosKeyPatchCLIP,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PlatosKeyApply": "Plato's Key Apply (Procrustes)",
    "PlatosKeyCLIPTextEncode": "Plato's Key CLIP Text Encode",
    "PlatosKeyPatchCLIP": "Plato's Key Patch CLIP (H3 Plug & Play)",
}

#!/usr/bin/env python3
"""Calibrate a Plato's Key (.safetensors) between a donor and target text encoder.

Implements Stanford Hazy Research's "Plato's Key": closed-form Orthogonal Procrustes
rotation + global isotropic scale via SVD for zero-gradient hidden-state transfer.

Example (same tokenizer family, e.g. Qwen 3.5 1.5B -> Qwen 3.5 9B):
    python calibrate_platos_key.py \
        --donor Qwen/Qwen3.5-1.5B \
        --target Qwen/Qwen3.5-9B \
        --output keys/qwen35_1.5b_to_9b.safetensors \
        --sequential

Example (cross-family, e.g. Gemma -> Qwen 3.5 9B):
    python calibrate_platos_key.py \
        --donor google/gemma-2-2b \
        --target Qwen/Qwen3.5-9B \
        --align word \
        --output keys/gemma2_2b_to_qwen35_9b.safetensors \
        --sequential
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from platos_key_core import evaluate_platos_key, fit_platos_key, fit_ridge_key, save_platos_key

DEFAULT_CORPUS = Path(__file__).resolve().parent / "prompts_corpus.json"
WORD_SPAN_RE = re.compile(r"\S+")


def load_prompts(corpus_path: str | Path, max_prompts: Optional[int] = None) -> List[str]:
    """Load prompts from JSON corpus (list of strings, list of dicts, or {'prompts': ...})."""
    path = Path(corpus_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "prompts" in data:
        items = data["prompts"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"Unsupported corpus JSON structure in {path}")

    prompts: List[str] = []
    for item in items:
        if isinstance(item, str):
            prompts.append(item.strip())
        elif isinstance(item, dict) and "prompt" in item:
            prompts.append(str(item["prompt"]).strip())
        else:
            raise ValueError(f"Unsupported prompt entry: {type(item)}")

    prompts = [p for p in prompts if p]
    if max_prompts is not None and max_prompts > 0:
        prompts = prompts[:max_prompts]
    if len(prompts) < 2:
        raise ValueError(f"Need at least 2 prompts for calibration, found {len(prompts)}")
    return prompts


def parse_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    key = dtype_str.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype_str}'. Choose from: {sorted(mapping)}")
    return mapping[key]


def _word_spans(text: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in WORD_SPAN_RE.finditer(text)]


def _pool_word_spans(
    hidden: torch.Tensor,
    offsets: Sequence[Tuple[int, int]],
    spans: Sequence[Tuple[int, int]],
) -> Tuple[List[torch.Tensor], List[int]]:
    """Mean-pool subword hidden states into whitespace-delimited word spans."""
    vecs: List[torch.Tensor] = []
    valid_word_ids: List[int] = []

    for w_idx, (w_start, w_end) in enumerate(spans):
        tok_indices = [
            t_idx
            for t_idx, (t_start, t_end) in enumerate(offsets)
            if t_end > t_start and t_start < w_end and t_end > w_start
        ]
        if tok_indices:
            vecs.append(hidden[tok_indices].mean(dim=0))
            valid_word_ids.append(w_idx)
    return vecs, valid_word_ids


def detect_alignment_mode(
    donor_tokenizer: Any,
    target_tokenizer: Any,
    sample_prompts: Sequence[str],
    requested: str,
) -> str:
    """Resolve 'auto' alignment to 'token', 'word', or 'sentence'."""
    if requested != "auto":
        return requested

    probe = list(sample_prompts[: min(5, len(sample_prompts))])
    try:
        d_ids = [donor_tokenizer.encode(p, add_special_tokens=False) for p in probe]
        t_ids = [target_tokenizer.encode(p, add_special_tokens=False) for p in probe]
        if d_ids == t_ids:
            return "token"
    except Exception:
        pass

    if getattr(donor_tokenizer, "is_fast", False) and getattr(target_tokenizer, "is_fast", False):
        return "word"
    return "sentence"


def _extract_model_features(
    model_name_or_path: str,
    prompts: Sequence[str],
    layer_idx: int,
    align_mode: str,
    device: str,
    model_dtype: torch.dtype,
    batch_size: int,
    max_length: int,
    trust_remote_code: bool,
    skip_special_tokens: bool,
) -> List[Dict[str, Any]]:
    """Run a single encoder over all prompts and return CPU float32 per-prompt features."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = AutoModel.from_pretrained(
        model_name_or_path,
        torch_dtype=model_dtype,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    model.to(device)

    need_offsets = align_mode == "word" and getattr(tokenizer, "is_fast", False)
    results: List[Dict[str, Any]] = []

    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch_prompts = list(prompts[start : start + batch_size])
            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=need_offsets,
                return_special_tokens_mask=True,
            )

            offsets_batch = enc.pop("offset_mapping", None)
            special_mask_batch = enc.pop("special_tokens_mask", None)

            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[layer_idx].detach().to(device="cpu", dtype=torch.float32)
            attn_cpu = attention_mask.detach().to(device="cpu")
            ids_cpu = input_ids.detach().to(device="cpu")

            for i, prompt_text in enumerate(batch_prompts):
                valid_mask = attn_cpu[i].bool()
                if skip_special_tokens and special_mask_batch is not None:
                    non_special = ~special_mask_batch[i].bool()
                    if (valid_mask & non_special).any():
                        valid_mask = valid_mask & non_special

                h_valid = hidden_states[i][valid_mask]
                ids_valid = ids_cpu[i][valid_mask].tolist()

                entry: Dict[str, Any] = {
                    "hidden": h_valid,
                    "token_ids": ids_valid,
                }

                if align_mode == "word" and offsets_batch is not None:
                    raw_offsets = [
                        (int(a), int(b))
                        for a, b in offsets_batch[i][valid_mask].tolist()
                    ]
                    spans = _word_spans(prompt_text)
                    w_vecs, w_ids = _pool_word_spans(h_valid, raw_offsets, spans)
                    entry["word_vecs"] = w_vecs
                    entry["word_ids"] = w_ids
                elif align_mode == "sentence":
                    mean_vec = h_valid.mean(dim=0)
                    last_vec = h_valid[-1]
                    entry["sentence_vecs"] = torch.stack([mean_vec, last_vec], dim=0)

                results.append(entry)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def align_prompt_features(
    donor_feats: Sequence[Dict[str, Any]],
    target_feats: Sequence[Dict[str, Any]],
    align_mode: str,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Align extracted donor & target features per prompt.

    Returns a list of (X_donor_prompt, X_target_prompt) pairs so train/val splits
    can be performed strictly at the prompt level without leakage.
    """
    aligned_per_prompt: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for d_feat, t_feat in zip(donor_feats, target_feats):
        if align_mode == "token":
            dh = d_feat["hidden"]
            th = t_feat["hidden"]
            if dh.shape[0] == th.shape[0] and dh.shape[0] > 0:
                aligned_per_prompt.append((dh, th))
            else:
                # Fallback for rare tokenization discrepancy on a single prompt
                d_s = torch.stack([dh.mean(dim=0), dh[-1]], dim=0)
                t_s = torch.stack([th.mean(dim=0), th[-1]], dim=0)
                aligned_per_prompt.append((d_s, t_s))
        elif align_mode == "word":
            d_map = dict(zip(d_feat.get("word_ids", []), d_feat.get("word_vecs", [])))
            t_map = dict(zip(t_feat.get("word_ids", []), t_feat.get("word_vecs", [])))
            common_ids = sorted(set(d_map.keys()) & set(t_map.keys()))
            if common_ids:
                d_mat = torch.stack([d_map[i] for i in common_ids], dim=0)
                t_mat = torch.stack([t_map[i] for i in common_ids], dim=0)
                aligned_per_prompt.append((d_mat, t_mat))
        elif align_mode == "sentence":
            aligned_per_prompt.append((d_feat["sentence_vecs"], t_feat["sentence_vecs"]))
        else:
            raise ValueError(f"Unknown alignment mode: {align_mode}")

    if not aligned_per_prompt:
        raise RuntimeError(f"No aligned vectors produced with align_mode='{align_mode}'.")
    return aligned_per_prompt


def split_train_val(
    aligned_per_prompt: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    val_fraction: float = 0.10,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split aligned per-prompt tensors into train and held-out validation matrices."""
    indices = list(range(len(aligned_per_prompt)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_val = max(1, int(round(len(indices) * val_fraction)))
    if n_val >= len(indices):
        n_val = max(1, len(indices) // 5)

    val_idx = set(indices[:n_val])
    train_d, train_t, val_d, val_t = [], [], [], []
    for idx, (xd, xt) in enumerate(aligned_per_prompt):
        if idx in val_idx:
            val_d.append(xd)
            val_t.append(xt)
        else:
            train_d.append(xd)
            train_t.append(xt)

    return (
        torch.cat(train_d, dim=0),
        torch.cat(train_t, dim=0),
        torch.cat(val_d, dim=0),
        torch.cat(val_t, dim=0),
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate a Plato's Key (.safetensors) via closed-form Orthogonal Procrustes SVD."
    )
    parser.add_argument("--donor", required=True, help="Donor HF model ID or local path (e.g. Qwen/Qwen3.5-1.5B)")
    parser.add_argument("--target", required=True, help="Target HF model ID or local path (e.g. Qwen/Qwen3.5-9B)")
    parser.add_argument("--output", required=True, help="Output .safetensors path for the calibrated Plato's Key")
    parser.add_argument(
        "--corpus",
        default=str(DEFAULT_CORPUS),
        help=f"Path to prompts JSON corpus (default: {DEFAULT_CORPUS})",
    )
    parser.add_argument("--max-prompts", type=int, default=None, help="Optional cap on number of prompts")
    parser.add_argument(
        "--align",
        choices=["auto", "token", "word", "sentence"],
        default="auto",
        help="Token alignment strategy (default: auto)",
    )
    parser.add_argument("--donor-layer", type=int, default=-1, help="Donor hidden-state layer index (default: -1)")
    parser.add_argument("--target-layer", type=int, default=-1, help="Target hidden-state layer index (default: -1)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for hidden-state extraction")
    parser.add_argument("--max-length", type=int, default=256, help="Max token length per prompt")
    parser.add_argument("--val-split", type=float, default=0.10, help="Held-out prompt validation fraction (default: 0.10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for prompt split")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device")
    parser.add_argument("--dtype", default="bf16", help="Model inference dtype: fp32, fp16, bf16 (default: bf16)")
    parser.add_argument("--save-dtype", default="fp32", help="Saved key tensor dtype: fp32, fp16, bf16 (default: fp32)")
    parser.add_argument(
        "--sequential",
        action="store_true",
        default=True,
        help="Load donor and target sequentially to minimize peak VRAM (default: True)",
    )
    parser.add_argument(
        "--method",
        choices=["ridge", "procrustes"],
        default="ridge",
        help="Closed-form solver: 'ridge' (recommended for MiniMax-H3) or 'procrustes' (default: ridge)",
    )
    parser.add_argument(
        "--ridge-lambda",
        type=float,
        default=None,
        help="Explicit L2 regularization lambda for Ridge (omit to auto-select best on validation split)",
    )
    parser.add_argument("--keep-special-tokens", action="store_true", help="Include BOS/EOS tokens in alignment")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to transformers")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    prompts = load_prompts(args.corpus, max_prompts=args.max_prompts)
    print(f"[Plato's Key] Loaded {len(prompts)} prompts from {args.corpus}")

    donor_tok = AutoTokenizer.from_pretrained(args.donor, trust_remote_code=args.trust_remote_code)
    target_tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=args.trust_remote_code)
    align_mode = detect_alignment_mode(donor_tok, target_tok, prompts, args.align)
    print(f"[Plato's Key] Alignment mode: {align_mode} (requested: {args.align})")
    del donor_tok, target_tok

    model_dtype = parse_dtype(args.dtype)
    save_dtype = parse_dtype(args.save_dtype)

    print(f"[Plato's Key] Extracting donor hidden states ({args.donor}, layer={args.donor_layer})...")
    donor_feats = _extract_model_features(
        model_name_or_path=args.donor,
        prompts=prompts,
        layer_idx=args.donor_layer,
        align_mode=align_mode,
        device=args.device,
        model_dtype=model_dtype,
        batch_size=args.batch_size,
        max_length=args.max_length,
        trust_remote_code=args.trust_remote_code,
        skip_special_tokens=not args.keep_special_tokens,
    )

    print(f"[Plato's Key] Extracting target hidden states ({args.target}, layer={args.target_layer})...")
    target_feats = _extract_model_features(
        model_name_or_path=args.target,
        prompts=prompts,
        layer_idx=args.target_layer,
        align_mode=align_mode,
        device=args.device,
        model_dtype=model_dtype,
        batch_size=args.batch_size,
        max_length=args.max_length,
        trust_remote_code=args.trust_remote_code,
        skip_special_tokens=not args.keep_special_tokens,
    )

    aligned = align_prompt_features(donor_feats, target_feats, align_mode)
    x_train, y_train, x_val, y_val = split_train_val(aligned, val_fraction=args.val_split, seed=args.seed)
    print(
        f"[Plato's Key] Aligned vectors -> train: {tuple(x_train.shape)} -> {tuple(y_train.shape)}, "
        f"val: {tuple(x_val.shape)} -> {tuple(y_val.shape)}"
    )

    base_meta = {
        "donor_model": args.donor,
        "target_model": args.target,
        "donor_layer": str(args.donor_layer),
        "target_layer": str(args.target_layer),
        "align_mode": align_mode,
        "corpus_prompts": str(len(prompts)),
        "method": args.method,
        "add_special_tokens": "true" if args.keep_special_tokens else "false",
    }

    if args.method == "ridge":
        lambdas = [args.ridge_lambda] if args.ridge_lambda is not None else [1.0, 10.0, 100.0, 1000.0, 10000.0]
        best_key = None
        best_val = None
        for lam in lambdas:
            cand = fit_ridge_key(x_train, y_train, lam=lam, metadata=dict(base_meta), save_dtype=save_dtype)
            vm = evaluate_platos_key(x_val, y_val, cand)
            print(f"[Plato's Key] Ridge lam={lam:g} -> val_cos={vm['cosine_mean']:.5f} val_R2={vm['r2']:.5f}")
            if best_val is None or vm["r2"] > best_val["r2"]:
                best_key, best_val = cand, vm
        key = best_key
    else:
        key = fit_platos_key(
            x_train,
            y_train,
            metadata=base_meta,
            save_dtype=save_dtype,
        )

    train_metrics = evaluate_platos_key(x_train, y_train, key)
    val_metrics = evaluate_platos_key(x_val, y_val, key)

    # Record validation metrics in safetensors metadata
    key.metadata["val_cosine_mean"] = f"{val_metrics['cosine_mean']:.6f}"
    key.metadata["val_r2"] = f"{val_metrics['r2']:.6f}"
    key.metadata["val_mse"] = f"{val_metrics['mse']:.6f}"
    key.metadata["train_cosine_mean"] = f"{train_metrics['cosine_mean']:.6f}"
    key.metadata["train_r2"] = f"{train_metrics['r2']:.6f}"

    out_path = save_platos_key(key, args.output)
    print(f"[Plato's Key] Saved key to {out_path}")
    print(
        f"  Train | cos={train_metrics['cosine_mean']:.5f} (p10={train_metrics['cosine_p10']:.5f}) "
        f"| R^2={train_metrics['r2']:.5f} | rel_L2={train_metrics['rel_l2']:.5f}"
    )
    print(
        f"  Val   | cos={val_metrics['cosine_mean']:.5f} (p10={val_metrics['cosine_p10']:.5f}) "
        f"| R^2={val_metrics['r2']:.5f} | rel_L2={val_metrics['rel_l2']:.5f}"
    )


if __name__ == "__main__":
    main()

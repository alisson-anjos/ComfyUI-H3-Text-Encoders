"""Core mathematical primitives for Plato's Key (closed-form Orthogonal Procrustes + global scale).

Implements zero-gradient hidden-state transfer between text encoder representation spaces:
    y_hat = (x - donor_mean) @ R * scale + target_mean

where R is an orthogonal / semi-orthogonal matrix solving the Orthogonal Procrustes problem
via thin SVD of the centered cross-covariance matrix X_c^T Y_c, and `scale` is the closed-form
optimal global isotropic scale factor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from safetensors.torch import load_file, save_file


@dataclass
class PlatosKey:
    """Container for a calibrated Plato's Key affine-orthogonal transform."""

    R: torch.Tensor  # [d_donor, d_target]
    scale: torch.Tensor  # [1]
    donor_mean: torch.Tensor  # [d_donor]
    target_mean: torch.Tensor  # [d_target]
    metadata: Dict[str, str]

    @property
    def d_donor(self) -> int:
        return int(self.R.shape[0])

    @property
    def d_target(self) -> int:
        return int(self.R.shape[1])

    def to(self, device: torch.device | str, dtype: Optional[torch.dtype] = None) -> "PlatosKey":
        return PlatosKey(
            R=self.R.to(device=device, dtype=dtype),
            scale=self.scale.to(device=device, dtype=dtype),
            donor_mean=self.donor_mean.to(device=device, dtype=dtype),
            target_mean=self.target_mean.to(device=device, dtype=dtype),
            metadata=self.metadata,
        )


def fit_platos_key(
    x_donor: torch.Tensor,
    x_target: torch.Tensor,
    metadata: Optional[Dict[str, Any]] = None,
    save_dtype: torch.dtype = torch.float32,
) -> PlatosKey:
    """Solve closed-form Orthogonal Procrustes with centering and optimal global scale.

    Minimizes || scale * (X_donor - mu_d) @ R - (X_target - mu_t) ||_F^2
    subject to R having orthonormal rows (d_donor <= d_target) or orthonormal columns (d_donor >= d_target).

    Args:
        x_donor: Tensor of shape [N, d_donor].
        x_target: Tensor of shape [N, d_target].
        metadata: Optional metadata dictionary (values will be stringified for safetensors).
        save_dtype: Floating-point dtype for the returned key tensors.

    Returns:
        Calibrated PlatosKey instance.
    """
    if x_donor.ndim != 2 or x_target.ndim != 2:
        raise ValueError(
            f"Expected 2D tensors [N, D], got x_donor={tuple(x_donor.shape)}, x_target={tuple(x_target.shape)}"
        )
    if x_donor.shape[0] != x_target.shape[0]:
        raise ValueError(
            f"Sample count mismatch: x_donor has {x_donor.shape[0]}, x_target has {x_target.shape[0]}"
        )
    if x_donor.shape[0] < 2:
        raise ValueError("Need at least 2 aligned samples to fit Plato's Key.")

    # Compute SVD in float64 on CPU for maximum numerical accuracy
    xd = x_donor.detach().to(device="cpu", dtype=torch.float64)
    xt = x_target.detach().to(device="cpu", dtype=torch.float64)

    donor_mean = xd.mean(dim=0)
    target_mean = xt.mean(dim=0)

    xd_c = xd - donor_mean
    xt_c = xt - target_mean

    # Cross-covariance M = X_c^T @ Y_c, shape [d_donor, d_target]
    cross_cov = xd_c.T @ xt_c
    u, s, vh = torch.linalg.svd(cross_cov, full_matrices=False)
    R = u @ vh  # [d_donor, d_target]

    # Optimal isotropic scale: trace(S) / ||X_c @ R||_F^2
    # Note: when d_donor <= d_target, R @ R.T = I, so ||X_c @ R||_F^2 == ||X_c||_F^2.
    # Computing ||X_c @ R||_F^2 directly is exact for all (d_donor, d_target) regimes.
    if xd_c.shape[1] <= xt_c.shape[1]:
        denom = xd_c.pow(2).sum()
    else:
        denom = (xd_c @ R).pow(2).sum()

    if denom <= 1e-12:
        scale_val = 1.0
    else:
        scale_val = float((s.sum() / denom).item())

    str_meta: Dict[str, str] = {
        "format": "platos_key_v1",
        "d_donor": str(xd.shape[1]),
        "d_target": str(xt.shape[1]),
        "num_samples": str(xd.shape[0]),
        "scale": f"{scale_val:.8f}",
    }
    if metadata:
        for k, v in metadata.items():
            str_meta[str(k)] = v if isinstance(v, str) else json.dumps(v)

    return PlatosKey(
        R=R.to(dtype=save_dtype).contiguous(),
        scale=torch.tensor([scale_val], dtype=save_dtype),
        donor_mean=donor_mean.to(dtype=save_dtype).contiguous(),
        target_mean=target_mean.to(dtype=save_dtype).contiguous(),
        metadata=str_meta,
    )


def fit_ridge_key(
    x_donor: torch.Tensor,
    x_target: torch.Tensor,
    ridge_lambda: float = 10.0,
    norm_variance: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
    save_dtype: torch.dtype = torch.float16,
) -> PlatosKey:
    """Solve closed-form Ridge regression with centering (zero gradient descent).

    Minimizes || (X_donor - mu_d) @ W - (X_target - mu_t) ||_F^2 + lambda * ||W||_F^2.
    Matches anisotropic per-channel outlier variances of MiniMax-H3's 5120-dim 32B space.
    When `norm_variance=True` (`ridge_norm`), rescales `scale` so projected variance matches target variance.
    """
    if x_donor.ndim != 2 or x_target.ndim != 2:
        raise ValueError("Expected 2D tensors [N, D].")
    if x_donor.shape[0] != x_target.shape[0] or x_donor.shape[0] < 2:
        raise ValueError("Mismatched or insufficient aligned samples.")

    xd = x_donor.detach().to(device="cpu", dtype=torch.float64)
    xt = x_target.detach().to(device="cpu", dtype=torch.float64)

    donor_mean = xd.mean(dim=0)
    target_mean = xt.mean(dim=0)
    xd_c = xd - donor_mean
    xt_c = xt - target_mean

    d_in = xd_c.shape[1]
    gram = xd_c.T @ xd_c
    gram.diagonal().add_(float(ridge_lambda))
    rhs = xd_c.T @ xt_c
    W = torch.linalg.solve(gram, rhs)  # [d_donor, d_target]

    scale_val = 1.0
    if norm_variance:
        pred_norm = (xd_c @ W).pow(2).sum()
        tgt_norm = xt_c.pow(2).sum()
        if pred_norm > 1e-12:
            scale_val = float(torch.sqrt(tgt_norm / pred_norm).item())

    str_meta: Dict[str, str] = {
        "format": "platos_key_v1",
        "method": "ridge_norm" if norm_variance else "ridge",
        "ridge_lambda": str(ridge_lambda),
        "d_donor": str(d_in),
        "d_target": str(xt.shape[1]),
        "num_samples": str(xd.shape[0]),
        "scale": f"{scale_val:.8f}",
    }
    if metadata:
        for k, v in metadata.items():
            str_meta[str(k)] = v if isinstance(v, str) else json.dumps(v)

    return PlatosKey(
        R=W.to(dtype=save_dtype).contiguous(),
        scale=torch.tensor([scale_val], dtype=save_dtype),
        donor_mean=donor_mean.to(dtype=save_dtype).contiguous(),
        target_mean=target_mean.to(dtype=save_dtype).contiguous(),
        metadata=str_meta,
    )


def apply_platos_key(
    x: torch.Tensor,
    key: PlatosKey,
    strength: float = 1.0,
) -> torch.Tensor:
    """Apply a Plato's Key transform to hidden states of shape [..., d_donor].

    Args:
        x: Input hidden states with last dimension == key.d_donor.
        key: Calibrated PlatosKey.
        strength: Blend factor.
            - If d_donor == d_target: linear blend `(1 - strength) * x + strength * mapped`.
            - If d_donor != d_target: interpolates from `target_mean` (strength=0) to `mapped` (strength=1).

    Returns:
        Mapped hidden states of shape [..., d_target] with same device and dtype as `x`.
    """
    if x.shape[-1] != key.d_donor:
        raise ValueError(
            f"Input hidden dimension {x.shape[-1]} does not match Plato's Key donor dimension {key.d_donor}."
        )

    orig_dtype = x.dtype
    work_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    device = x.device

    R = key.R.to(device=device, dtype=work_dtype)
    scale = key.scale.to(device=device, dtype=work_dtype)
    donor_mean = key.donor_mean.to(device=device, dtype=work_dtype)
    target_mean = key.target_mean.to(device=device, dtype=work_dtype)

    x_w = x.to(dtype=work_dtype)
    projected = ((x_w - donor_mean) @ R) * scale

    if key.d_donor == key.d_target:
        mapped = projected + target_mean
        if strength != 1.0:
            out = x_w * (1.0 - strength) + mapped * strength
        else:
            out = mapped
    else:
        if strength != 1.0:
            out = target_mean + projected * strength
        else:
            out = projected + target_mean

    return out.to(dtype=orig_dtype)


def evaluate_platos_key(
    x_donor: torch.Tensor,
    x_target: torch.Tensor,
    key: PlatosKey,
) -> Dict[str, float]:
    """Compute validation metrics (cosine similarity, R^2, MSE, relative L2 error) on held-out pairs."""
    with torch.no_grad():
        xd = x_donor.detach().to(device="cpu", dtype=torch.float64)
        xt = x_target.detach().to(device="cpu", dtype=torch.float64)
        key_f64 = key.to(device="cpu", dtype=torch.float64)

        pred = apply_platos_key(xd, key_f64, strength=1.0)

        # Per-sample cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(pred, xt, dim=-1)
        mean_cos = float(cos_sim.mean().item())
        min_cos = float(cos_sim.min().item())
        p10_cos = float(torch.quantile(cos_sim, 0.10).item())

        # Coefficient of determination R^2
        residual = xt - pred
        ss_res = float(residual.pow(2).sum().item())
        ss_tot = float((xt - xt.mean(dim=0, keepdim=True)).pow(2).sum().item())
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

        mse = float(residual.pow(2).mean().item())
        rel_l2 = float(residual.norm() / (xt.norm() + 1e-12))

    return {
        "cosine_mean": mean_cos,
        "cosine_p10": p10_cos,
        "cosine_min": min_cos,
        "r2": r2,
        "mse": mse,
        "rel_l2": rel_l2,
    }


def save_platos_key(key: PlatosKey, path: str | Path) -> Path:
    """Save a Plato's Key to a compact `.safetensors` file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        "R": key.R.detach().cpu().contiguous(),
        "scale": key.scale.detach().cpu().reshape(1).contiguous(),
        "donor_mean": key.donor_mean.detach().cpu().contiguous(),
        "target_mean": key.target_mean.detach().cpu().contiguous(),
    }
    save_file(tensors, str(out_path), metadata=key.metadata)
    return out_path


def load_platos_key(path: str | Path, device: str | torch.device = "cpu") -> PlatosKey:
    """Load a Plato's Key from a `.safetensors` file."""
    from safetensors import safe_open

    in_path = Path(path)
    if not in_path.is_file():
        raise FileNotFoundError(f"Plato's Key file not found: {in_path}")

    tensors = load_file(str(in_path), device=str(device))
    required = ("R", "scale", "donor_mean", "target_mean")
    missing = [k for k in required if k not in tensors]
    if missing:
        raise KeyError(f"Invalid Plato's Key file {in_path}; missing tensors: {missing}")

    with safe_open(str(in_path), framework="pt", device=str(device)) as f:
        metadata = dict(f.metadata() or {})

    return PlatosKey(
        R=tensors["R"],
        scale=tensors["scale"].reshape(1),
        donor_mean=tensors["donor_mean"],
        target_mean=tensors["target_mean"],
        metadata=metadata,
    )

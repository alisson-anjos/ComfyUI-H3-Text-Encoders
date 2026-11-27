"""CPU-only synthetic unit tests for ComfyUI-H3TextEncoders."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

# Ensure project root and its parent are importable for both package & script imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from calibrate_platos_key import align_prompt_features, load_prompts, split_train_val
from platos_key_core import apply_platos_key, evaluate_platos_key, fit_platos_key, load_platos_key, save_platos_key

# Import ComfyUI nodes via package loader (handling hyphenated directory name)
import importlib.util

def _load_nodes_module():
    pkg_name = "comfyui_platos_key"
    spec_init = importlib.util.spec_from_file_location(
        pkg_name,
        PROJECT_ROOT / "__init__.py",
        submodule_search_locations=[str(PROJECT_ROOT)],
    )
    pkg = importlib.util.module_from_spec(spec_init)
    sys.modules[pkg_name] = pkg
    spec_init.loader.exec_module(pkg)
    return sys.modules[f"{pkg_name}.nodes"]


nodes_mod = _load_nodes_module()


def _random_orthonormal(d_in: int, d_out: int, generator: torch.Generator) -> torch.Tensor:
    """Create an exact semi-orthogonal matrix R of shape [d_in, d_out]."""
    mat = torch.randn((max(d_in, d_out), max(d_in, d_out)), generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(mat)
    return q[:d_in, :d_out].contiguous()


class TestPlatosKey(unittest.TestCase):
    def test_exact_recovery_same_dim(self) -> None:
        g = torch.Generator().manual_seed(1234)
        n_train, n_val, d = 600, 100, 128
        true_R = _random_orthonormal(d, d, g)
        true_scale = 2.75
        true_donor_mean = torch.randn((d,), generator=g, dtype=torch.float64) * 0.5
        true_target_mean = torch.randn((d,), generator=g, dtype=torch.float64) * 1.2

        x_all = torch.randn((n_train + n_val, d), generator=g, dtype=torch.float64) + true_donor_mean
        y_all = ((x_all - true_donor_mean) @ true_R) * true_scale + true_target_mean

        key = fit_platos_key(x_all[:n_train], y_all[:n_train], save_dtype=torch.float32)
        metrics = evaluate_platos_key(x_all[n_train:], y_all[n_train:], key)

        self.assertGreater(metrics["cosine_mean"], 0.9999)
        self.assertGreater(metrics["r2"], 0.9999)
        self.assertAlmostEqual(float(key.scale.item()), true_scale, places=4)

    def test_exact_recovery_up_projection_subspace(self) -> None:
        """Simulates smaller donor -> larger target (e.g., Qwen 3.5 1.5B -> Qwen 3.5 9B)."""
        g = torch.Generator().manual_seed(5678)
        n_train, n_val = 800, 120
        d_donor, d_target = 64, 256

        true_R = _random_orthonormal(d_donor, d_target, g)  # R @ R.T = I_64
        true_scale = 1.85
        true_donor_mean = torch.randn((d_donor,), generator=g, dtype=torch.float64)
        true_target_mean = torch.randn((d_target,), generator=g, dtype=torch.float64)

        x_all = torch.randn((n_train + n_val, d_donor), generator=g, dtype=torch.float64) + true_donor_mean
        y_all = ((x_all - true_donor_mean) @ true_R) * true_scale + true_target_mean

        key = fit_platos_key(x_all[:n_train], y_all[:n_train], save_dtype=torch.float32)
        metrics = evaluate_platos_key(x_all[n_train:], y_all[n_train:], key)

        self.assertEqual(key.d_donor, d_donor)
        self.assertEqual(key.d_target, d_target)
        self.assertGreater(metrics["cosine_mean"], 0.9999)
        self.assertGreater(metrics["r2"], 0.9999)
        self.assertAlmostEqual(float(key.scale.item()), true_scale, places=4)

    def test_safetensors_roundtrip_and_comfy_node(self) -> None:
        g = torch.Generator().manual_seed(999)
        d_donor, d_target = 48, 96
        true_R = _random_orthonormal(d_donor, d_target, g)
        true_scale = 0.65
        true_donor_mean = torch.randn((d_donor,), generator=g, dtype=torch.float64)
        true_target_mean = torch.randn((d_target,), generator=g, dtype=torch.float64)

        x = torch.randn((400, d_donor), generator=g, dtype=torch.float64) + true_donor_mean
        y = ((x - true_donor_mean) @ true_R) * true_scale + true_target_mean

        key = fit_platos_key(
            x[:350],
            y[:350],
            metadata={"donor_model": "synthetic-donor", "target_model": "synthetic-target"},
            save_dtype=torch.float32,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = Path(tmpdir) / "test_key.safetensors"
            save_platos_key(key, key_path)
            loaded = load_platos_key(key_path)

            self.assertEqual(loaded.metadata["donor_model"], "synthetic-donor")
            self.assertTrue(torch.allclose(loaded.R, key.R))

            # Test ComfyUI PlatosKeyApply node on [B, T, D_donor] conditioning
            cond_in = x[350:370].reshape(2, 10, d_donor).to(torch.float32)
            expected = y[350:370].reshape(2, 10, d_target).to(torch.float32)
            pooled_in = x[350:352].to(torch.float32)

            conditioning = [[cond_in.clone(), {"pooled_output": pooled_in.clone(), "tag": "ok"}]]
            node = nodes_mod.PlatosKeyApply()
            (mapped_cond,) = node.apply(
                conditioning=conditioning,
                key_file="(none)",
                strength=1.0,
                key_path_override=str(key_path),
                apply_to_pooled=True,
            )

            out_tensor, out_dict = mapped_cond[0]
            self.assertEqual(tuple(out_tensor.shape), (2, 10, d_target))
            self.assertEqual(tuple(out_dict["pooled_output"].shape), (2, d_target))
            self.assertEqual(out_dict["tag"], "ok")

            cos = torch.nn.functional.cosine_similarity(out_tensor.reshape(-1, d_target), expected.reshape(-1, d_target), dim=-1)
            self.assertGreater(float(cos.mean().item()), 0.9999)

            # Verify strength=0.0 produces target_mean for cross-dim
            (zero_cond,) = node.apply(
                conditioning=conditioning,
                key_file="(none)",
                strength=0.0,
                key_path_override=str(key_path),
                apply_to_pooled=False,
            )
            self.assertTrue(torch.allclose(zero_cond[0][0], loaded.target_mean.expand_as(zero_cond[0][0]), atol=1e-5))

    def test_prompts_corpus_integrity(self) -> None:
        corpus_path = PROJECT_ROOT / "prompts_corpus.json"
        prompts = load_prompts(corpus_path)
        self.assertEqual(len(prompts), 200)
        self.assertEqual(len(set(prompts)), 200, "All prompts should be unique")

        with corpus_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        langs = {item["lang"] for item in raw["prompts"]}
        self.assertEqual(langs, {"en", "pt-BR"})

    def test_alignment_and_prompt_split(self) -> None:
        g = torch.Generator().manual_seed(42)
        donor_feats = [{"hidden": torch.randn((5, 16), generator=g)} for _ in range(20)]
        target_feats = [{"hidden": torch.randn((5, 32), generator=g)} for _ in range(20)]
        aligned = align_prompt_features(donor_feats, target_feats, align_mode="token")
        x_tr, y_tr, x_va, y_va = split_train_val(aligned, val_fraction=0.10, seed=42)
        self.assertEqual(x_tr.shape, (90, 16))
        self.assertEqual(y_tr.shape, (90, 32))
        self.assertEqual(x_va.shape, (10, 16))
        self.assertEqual(y_va.shape, (10, 32))

    def test_char_overlap_matrix_and_commutativity(self) -> None:
        import types
        pkg = types.ModuleType("h3te_test_pkg")
        pkg.__path__ = [str(PROJECT_ROOT)]
        sys.modules["h3te_test_pkg"] = pkg
        from h3te_test_pkg.nodes import _char_overlap_matrix

        # Target splits "Hello world" as ["Hello", " world"]; donor splits as ["Hel", "lo ", "wor", "ld"]
        q_offs = [(0, 5), (5, 11)]
        d_offs = [(0, 3), (3, 6), (6, 9), (9, 11)]
        A = _char_overlap_matrix(q_offs, d_offs)
        self.assertEqual(tuple(A.shape), (2, 4))
        self.assertTrue(torch.allclose(A.sum(dim=-1), torch.ones(2), atol=1e-6))
        # Exact linear commutativity: A @ ((H - dm) @ W + tm) == ((A @ H) - dm) @ W + tm
        g = torch.Generator().manual_seed(7)
        H = torch.randn((4, 16), generator=g)
        dm = torch.randn((16,), generator=g)
        tm = torch.randn((32,), generator=g)
        W = torch.randn((16, 32), generator=g)
        lhs = A @ ((H - dm) @ W + tm)
        rhs = ((A @ H) - dm) @ W + tm
        self.assertTrue(torch.allclose(lhs, rhs, atol=1e-5))


if __name__ == "__main__":
    unittest.main(verbosity=2)

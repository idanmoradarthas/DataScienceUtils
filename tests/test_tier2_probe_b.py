"""Tests for Probe B tokenizer-variant comparison (.github/tier2/probes/probe_reference_cosines.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_probe_module():
    probe_path = Path(__file__).resolve().parent.parent / ".github" / "tier2" / "probes" / "probe_reference_cosines.py"
    spec = importlib.util.spec_from_file_location("probe_reference_cosines", probe_path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_compare_variants_exact_match():
    """Unit test: Ollama embeddings equal to variant 'bos' yields best_variant 'bos' with min cos 1.0."""
    probe = _load_probe_module()

    # 4-dim unit vectors for 3 texts
    plain_embeds = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    bos_embeds = [
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    eos_embeds = [
        [0.5, 0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5, -0.5],
    ]

    ref_variants = {
        "plain": {"embeddings": plain_embeds, "token_counts": [10, 12, 14]},
        "bos": {"embeddings": bos_embeds, "token_counts": [11, 13, 15]},
        "eos": {"embeddings": eos_embeds, "token_counts": [11, 13, 15]},
    }

    # Target is exact match to "bos"
    ollama_embeds = [list(vec) for vec in bos_embeds]
    ollama_tokens = [11, 13, 15]

    res = probe.compare_variants(ref_variants, ollama_embeds, ollama_tokens)

    assert res["best_variant"] == "bos"
    bos_res = res["variants"]["bos"]
    assert abs(bos_res["min_direct_cos"] - 1.0) < 1e-6
    assert abs(bos_res["mean_direct_cos"] - 1.0) < 1e-6
    assert abs(bos_res["max_abs_diff"]) < 1e-6
    assert bos_res["token_counts_match"] is True

    # Check token_counts_match is False for "plain"
    assert res["variants"]["plain"]["token_counts_match"] is False


def test_compare_variants_perturbed():
    """Unit test: slightly perturbed embeddings pick the nearest variant."""
    probe = _load_probe_module()

    plain_embeds = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    bos_embeds = [
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ]

    ref_variants = {
        "plain": {"embeddings": plain_embeds, "token_counts": [5, 6]},
        "bos": {"embeddings": bos_embeds, "token_counts": [6, 7]},
    }

    # Perturb bos slightly: cos to bos is ~0.999, cos to plain is 0
    perturbed_bos = [
        [0.01, 0.99995, 0.0, 0.0],
        [0.99995, 0.01, 0.0, 0.0],
    ]

    res = probe.compare_variants(ref_variants, perturbed_bos, [6, 7])
    assert res["best_variant"] == "bos"
    assert res["variants"]["bos"]["min_direct_cos"] > 0.99
    assert res["variants"]["plain"]["min_direct_cos"] < 0.1

#!/usr/bin/env python3
"""Probe B: Embedding Correctness vs Reference Cosines.

Evaluates pairwise cosine similarities for a fixed text set:
- The 6 skill descriptions from DataScienceUtils.
- A few short sentences, including one near-duplicate pair.
Checks:
- Max absolute difference of any pairwise cosine <= 0.02.
- Spearman rank correlation of pairs >= 0.98.
- Prints impact of "passage: " prefix for informational purposes.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

FIXED_TEXTS = [
    # Skill 1 (metrics)
    (
        "Machine learning model evaluation, classification diagnostics, threshold tuning, "
        "and learning curve analysis with publication-quality plots."
    ),
    # Skill 2 (preprocess)
    (
        "Exploratory data analysis (EDA), feature distribution visualization, and statistical "
        "correlation analysis for tabular datasets."
    ),
    # Skill 3 (strings)
    (
        "String preprocessing, split-safe tag encoding, and significant term extraction "
        "from text fields for machine learning models."
    ),
    # Skill 4 (transformers)
    (
        "Scikit-learn pipeline transformers for multi-label binarization and sentence embeddings "
        "with feature names and pandas output support."
    ),
    # Skill 5 (unsupervised)
    (
        "Unsupervised clustering analysis, optimal cluster count selection, and 2D scatter "
        "visualization with PCA, t-SNE, and UMAP."
    ),
    # Skill 6 (xai)
    (
        "Explainable AI (XAI) and model interpretation for tree-based models. Use for plotting "
        "feature importances and visualizing decision trees."
    ),
    # Sentence 7
    "Quick brown fox jumps over the lazy dog in the sunny park.",
    # Sentence 8 (near duplicate of 7)
    "A fast brown fox leaped over the sleepy dog under the bright sun.",
    # Sentence 9 (distinct topic)
    "Superconducting quantum interference devices operate at liquid helium temperatures.",
]


def cosine_sim(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two unit vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def compute_pairwise_cosines(embeddings: list[list[float]]) -> list[list[float]]:
    """Compute matrix of pairwise cosines."""
    n = len(embeddings)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            mat[i][j] = cosine_sim(embeddings[i], embeddings[j])
    return mat


def compute_spearman(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation coefficient between two lists."""
    n = len(x)
    if n <= 1:
        return 1.0

    def get_ranks(vals: list[float]) -> list[float]:
        indexed = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and math.isclose(indexed[j + 1][1], indexed[j][1], abs_tol=1e-9):
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx = get_ranks(x)
    ry = get_ranks(y)

    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    num = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mean_rx) ** 2 for a in rx) * sum((b - mean_ry) ** 2 for b in ry))
    if den == 0:
        return 0.0
    return num / den


def generate_reference_matrix_with_transformers() -> dict[str, Any]:
    """Compute reference cosine matrix using BF16 official checkpoint."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    model_id = "nvidia/Nemotron-3-Embed-1B-BF16"
    revision = "0677b2025cbc37daf92d7b9c7a225de8cfbc5b0b"
    print(f"Loading reference model {model_id} (revision: {revision}) on CPU...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float32,
    )
    model.eval()

    def average_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def encode(texts: list[str]) -> list[list[float]]:
        encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            output = model(**encoded)
            pooled = average_pool(output.last_hidden_state, encoded["attention_mask"])
            embeddings = F.normalize(pooled, p=2, dim=-1)
        return embeddings.cpu().tolist()

    raw_embeds = encode(FIXED_TEXTS)
    raw_cosines = compute_pairwise_cosines(raw_embeds)

    passage_embeds = encode(["passage: " + t for t in FIXED_TEXTS])
    passage_cosines = compute_pairwise_cosines(passage_embeds)

    return {
        "metadata": {
            "model_id": model_id,
            "revision": revision,
            "method": "AutoModel + AutoTokenizer, CPU float32, mean pooling, L2-normalized",
            "texts_count": len(FIXED_TEXTS),
        },
        "texts": FIXED_TEXTS,
        "raw_cosines": raw_cosines,
        "passage_cosines": passage_cosines,
    }


def main() -> None:
    """Run Probe B checks comparing Ollama embeddings against reference cosines."""
    base_url = os.environ.get("SKILL_EVAL_EMBEDDING_BASE_URL", "http://127.0.0.1:11434/v1")
    api_key = os.environ.get("SKILL_EVAL_EMBEDDING_API_KEY", "ollama")
    model = os.environ.get("SKILL_EVAL_EMBEDDING_MODEL", "nemotron-3-embed-1b")

    ref_path = Path("tests/resources/tier2_reference_cosines.json")
    recompute = os.environ.get("RECOMPUTE_REFERENCE", "false").lower() in ("true", "1")

    compute_only = os.environ.get("COMPUTE_ONLY", "false").lower() in ("true", "1") or "--compute-only" in sys.argv

    if not ref_path.exists() or recompute or compute_only:
        print(f"Generating reference matrix (exists={ref_path.exists()}, recompute={recompute})...")
        ref_data = generate_reference_matrix_with_transformers()
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ref_path, "w") as f:
            json.dump(ref_data, f, indent=2)
        print(f"Saved reference matrix to {ref_path}")
        if compute_only:
            print("Reference computation complete. Exiting (--compute-only).")
            return
    else:
        with open(ref_path) as f:
            ref_data = json.load(f)

    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)

    # 1. Fetch embeddings for FIXED_TEXTS from Ollama
    print(f"Requesting embeddings for {len(FIXED_TEXTS)} texts from Ollama ({model})...")
    resp_raw = client.embeddings.create(input=FIXED_TEXTS, model=model)
    ollama_embeds = [item.embedding for item in sorted(resp_raw.data, key=lambda x: x.index)]
    ollama_cosines = compute_pairwise_cosines(ollama_embeds)

    # 2. Compare against reference raw_cosines
    ref_cosines = ref_data["raw_cosines"]
    max_abs_diff = 0.0
    ref_flat = []
    ollama_flat = []

    n = len(FIXED_TEXTS)
    for i in range(n):
        for j in range(i + 1, n):
            ref_val = ref_cosines[i][j]
            ollama_val = ollama_cosines[i][j]
            diff = abs(ref_val - ollama_val)
            if diff > max_abs_diff:
                max_abs_diff = diff
            ref_flat.append(ref_val)
            ollama_flat.append(ollama_val)

    spearman = compute_spearman(ref_flat, ollama_flat)

    print("--- Embedding Correctness Probe Results ---")
    print(f"Max absolute pairwise cosine difference: {max_abs_diff:.6f} (Gate: <= 0.02)")
    print(f"Spearman rank correlation:              {spearman:.6f} (Gate: >= 0.98)")

    # 3. Informational check on passage prefix impact
    if "passage_cosines" in ref_data:
        passage_cos = ref_data["passage_cosines"]
        passage_diffs = [abs(ref_cosines[i][j] - passage_cos[i][j]) for i in range(n) for j in range(i + 1, n)]
        avg_prefix_diff = sum(passage_diffs) / len(passage_diffs)
        max_prefix_diff = max(passage_diffs)
        print(
            f"Informational - 'passage: ' prefix impact: avg diff={avg_prefix_diff:.4f}, max diff={max_prefix_diff:.4f}"
        )

    # Apply gates
    gate_failed = False
    if max_abs_diff > 0.02:
        print(f"ERROR: Max pairwise cosine diff {max_abs_diff:.6f} exceeded 0.02!", file=sys.stderr)
        gate_failed = True
    if spearman < 0.98:
        print(f"ERROR: Spearman rank correlation {spearman:.6f} below 0.98!", file=sys.stderr)
        gate_failed = True

    if gate_failed:
        sys.exit(1)

    print("✓ Probe B passed: GGUF embeddings match reference BF16 within gate thresholds.")


if __name__ == "__main__":
    main()

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

    # Diagnostics to stdout
    print(f"model.config.architectures: {getattr(model.config, 'architectures', None)}")
    print(f"getattr(model.config, 'is_causal', None): {getattr(model.config, 'is_causal', None)}")

    # Tokenizer checks on text 0
    t0_encoded = tokenizer(FIXED_TEXTS[0])
    t0_ids = t0_encoded["input_ids"]
    first_is_special = t0_ids[0] in tokenizer.all_special_ids if t0_ids else False
    last_is_special = t0_ids[-1] in tokenizer.all_special_ids if t0_ids else False
    print(f"Text 0 token ids: first_is_special={first_is_special}, last_is_special={last_is_special}")

    # Token counts per text
    raw_token_counts = [len(tokenizer(t)["input_ids"]) for t in FIXED_TEXTS]
    passage_texts = ["passage: " + t for t in FIXED_TEXTS]
    passage_token_counts = [len(tokenizer(t)["input_ids"]) for t in passage_texts]

    def average_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def encode(texts: list[str]) -> torch.Tensor:
        encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            output = model(**encoded)
            pooled = average_pool(output.last_hidden_state, encoded["attention_mask"])
            embeddings = F.normalize(pooled, p=2, dim=-1)
        return embeddings

    # Padding artifacts check: text 0..2 encoded alone vs in padded batch
    first_three = FIXED_TEXTS[:3]
    batch_embeds = encode(first_three)
    individual_embeds = [encode([t])[0] for t in first_three]
    max_pad_diff = 0.0
    for b_vec, i_vec in zip(batch_embeds, individual_embeds):
        diff = (b_vec - i_vec).abs().max().item()
        if diff > max_pad_diff:
            max_pad_diff = diff
    print(f"Padding check (texts 0..2 alone vs padded batch max abs diff): {max_pad_diff:.8f}")

    # Causality check: max abs diff of hidden states of first 5 tokens of text 0 vs text with " extra words"
    t0_tokens = tokenizer(FIXED_TEXTS[0], return_tensors="pt")
    t0_extended = tokenizer(FIXED_TEXTS[0] + " extra words", return_tensors="pt")
    with torch.no_grad():
        out_t0 = model(**t0_tokens)
        out_ext = model(**t0_extended)
    hs_t0_5 = out_t0.last_hidden_state[:, :5, :]
    hs_ext_5 = out_ext.last_hidden_state[:, :5, :]
    causality_diff = (hs_t0_5 - hs_ext_5).abs().max().item()
    print(
        f"Causality check (first 5 tokens hidden states diff with appended words): {causality_diff:.8f} "
        f"({'causal attention' if causality_diff < 1e-4 else 'bidirectional attention'})"
    )

    raw_embeds_t = encode(FIXED_TEXTS)
    raw_embeds = raw_embeds_t.cpu().tolist()
    raw_cosines = compute_pairwise_cosines(raw_embeds)

    passage_embeds_t = encode(passage_texts)
    passage_embeds = passage_embeds_t.cpu().tolist()
    passage_cosines = compute_pairwise_cosines(passage_embeds)

    # Compact vectors: 9 x 2048 rounded to 6 decimals
    raw_compact = [[round(x, 6) for x in vec] for vec in raw_embeds]
    passage_compact = [[round(x, 6) for x in vec] for vec in passage_embeds]

    return {
        "metadata": {
            "model_id": model_id,
            "revision": revision,
            "method": "AutoModel + AutoTokenizer, CPU float32, mean pooling, L2-normalized",
            "texts_count": len(FIXED_TEXTS),
        },
        "texts": FIXED_TEXTS,
        "token_counts": {
            "raw": raw_token_counts,
            "passage": passage_token_counts,
        },
        "raw_embeddings": raw_compact,
        "passage_embeddings": passage_compact,
        "raw_cosines": raw_cosines,
        "passage_cosines": passage_cosines,
    }


def compute_metrics(
    ref_cosines: list[list[float]], target_cosines: list[list[float]]
) -> tuple[float, float, list[dict[str, Any]]]:
    """Compute max absolute difference, Spearman rank correlation, and top 5 worst pairs."""
    n = len(ref_cosines)
    ref_flat = []
    target_flat = []
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            ref_val = ref_cosines[i][j]
            targ_val = target_cosines[i][j]
            diff = abs(ref_val - targ_val)
            ref_flat.append(ref_val)
            target_flat.append(targ_val)
            pairs.append({"i": i, "j": j, "ref": ref_val, "target": targ_val, "diff": diff})

    max_abs_diff = max(p["diff"] for p in pairs) if pairs else 0.0
    spearman = compute_spearman(ref_flat, target_flat)
    pairs.sort(key=lambda x: x["diff"], reverse=True)
    worst_5 = pairs[:5]
    return max_abs_diff, spearman, worst_5


def run_sanity_block(label: str, cosines: list[list[float]]) -> None:
    """Print sanity metrics for a cosine matrix: sentences 7, 8, 9, skills mean, skill-sentence mean."""
    # Sentences are index 6, 7, 8 (texts 7, 8, 9)
    # Skills are index 0..5
    cos_7_8 = cosines[6][7]
    cos_7_9 = cosines[6][8]
    cos_8_9 = cosines[7][8]

    skill_pairs = [cosines[i][j] for i in range(6) for j in range(i + 1, 6)]
    mean_skills = sum(skill_pairs) / len(skill_pairs) if skill_pairs else 0.0

    skill_sent_pairs = [cosines[i][j] for i in range(6) for j in range(6, 9)]
    mean_skill_sent = sum(skill_sent_pairs) / len(skill_sent_pairs) if skill_sent_pairs else 0.0

    print(f"  [{label}] cos(7,8) [near-dup]={cos_7_8:.4f}, cos(7,9)={cos_7_9:.4f}, cos(8,9)={cos_8_9:.4f}")
    print(f"  [{label}] mean skill-skill cos={mean_skills:.4f}, mean skill-sentence cos={mean_skill_sent:.4f}")


def main() -> None:
    """Run Probe B checks comparing Ollama embeddings against reference cosines."""
    base_url = os.environ.get("TIER2_OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
    api_key = os.environ.get("SKILL_EVAL_EMBEDDING_API_KEY", "ollama")
    model = os.environ.get("SKILL_EVAL_EMBEDDING_MODEL", "nemotron-3-embed-1b")
    gate_variant = os.environ.get("TIER2_PROBE_B_GATE", "passage").lower()

    ref_path = Path("tests/resources/tier2_reference_cosines.json")
    recompute = os.environ.get("RECOMPUTE_REFERENCE", "false").lower() in ("true", "1")
    compute_only = os.environ.get("COMPUTE_ONLY", "false").lower() in ("true", "1") or "--compute-only" in sys.argv

    if not ref_path.exists() or recompute or compute_only:
        print(f"Generating reference matrix (exists={ref_path.exists()}, recompute={recompute})...")
        ref_data = generate_reference_matrix_with_transformers()
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        # Write without indentation to keep compact
        with open(ref_path, "w", encoding="utf-8") as f:
            json.dump(ref_data, f, separators=(",", ":"))
        print(f"Saved reference matrix to {ref_path}")
        if compute_only:
            print("Reference computation complete. Exiting (--compute-only).")
            return
    else:
        with open(ref_path, encoding="utf-8") as f:
            ref_data = json.load(f)

    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)

    ref_token_counts = ref_data.get("token_counts", {})
    ref_raw_tokens = ref_token_counts.get("raw", [])
    ref_passage_tokens = ref_token_counts.get("passage", [])

    ref_raw_embeddings = ref_data.get("raw_embeddings")
    ref_passage_embeddings = ref_data.get("passage_embeddings")

    # Embed each text one by one for both variants: raw and "passage: " + text
    print(f"\n--- Requesting Ollama ({base_url}, {model}) embeddings one by one ---")
    ollama_raw_embeds = []
    ollama_raw_tokens = []
    for idx, text in enumerate(FIXED_TEXTS):
        resp = client.embeddings.create(input=text, model=model)
        ollama_raw_embeds.append(resp.data[0].embedding)
        pt = resp.usage.prompt_tokens if hasattr(resp, "usage") and resp.usage else None
        ollama_raw_tokens.append(pt)

    passage_texts = ["passage: " + t for t in FIXED_TEXTS]
    ollama_passage_embeds = []
    ollama_passage_tokens = []
    for idx, text in enumerate(passage_texts):
        resp = client.embeddings.create(input=text, model=model)
        ollama_passage_embeds.append(resp.data[0].embedding)
        pt = resp.usage.prompt_tokens if hasattr(resp, "usage") and resp.usage else None
        ollama_passage_tokens.append(pt)

    # Print token counts per text
    print("\nToken counts (Ollama prompt_tokens vs Reference):")
    for i in range(len(FIXED_TEXTS)):
        ref_r = ref_raw_tokens[i] if i < len(ref_raw_tokens) else "N/A"
        ref_p = ref_passage_tokens[i] if i < len(ref_passage_tokens) else "N/A"
        print(
            f"  Text {i:2d} | Raw: Ollama={ollama_raw_tokens[i]} Ref={ref_r} | "
            f"Passage: Ollama={ollama_passage_tokens[i]} Ref={ref_p}"
        )

    # Direct cosine per text between Ollama vector and reference vector
    print("\nDirect Cosine vs Reference Vector per text:")
    if ref_raw_embeddings and ref_passage_embeddings:
        for i in range(len(FIXED_TEXTS)):
            cos_raw = cosine_sim(ollama_raw_embeds[i], ref_raw_embeddings[i])
            cos_pass = cosine_sim(ollama_passage_embeds[i], ref_passage_embeddings[i])
            print(f"  Text {i:2d} | Raw-vs-Raw cos={cos_raw:.6f} | Passage-vs-Passage cos={cos_pass:.6f}")
    else:
        print("  [NOTE: Reference JSON does not contain embedding vectors; skipping direct cosine comparison]")

    # Pairwise cosine matrices
    ollama_raw_cosines = compute_pairwise_cosines(ollama_raw_embeds)
    ollama_passage_cosines = compute_pairwise_cosines(ollama_passage_embeds)

    ref_raw_cosines = ref_data.get("raw_cosines", [])
    ref_passage_cosines = ref_data.get("passage_cosines", [])

    # Evaluate raw variant
    raw_max_diff, raw_spearman, raw_worst_5 = compute_metrics(ref_raw_cosines, ollama_raw_cosines)
    # Evaluate passage variant
    pass_max_diff, pass_spearman, pass_worst_5 = compute_metrics(ref_passage_cosines, ollama_passage_cosines)

    print("\n================ Pairwise Gate Numbers ================")
    print(f"Raw variant:     Max Abs Diff = {raw_max_diff:.6f}, Spearman = {raw_spearman:.6f}")
    print("  Worst 5 raw pairs (i, j, ref, ollama, diff):")
    for p in raw_worst_5:
        print(f"    ({p['i']}, {p['j']}) ref={p['ref']:.6f} ollama={p['target']:.6f} diff={p['diff']:.6f}")

    print(f"\nPassage variant: Max Abs Diff = {pass_max_diff:.6f}, Spearman = {pass_spearman:.6f}")
    print("  Worst 5 passage pairs (i, j, ref, ollama, diff):")
    for p in pass_worst_5:
        print(f"    ({p['i']}, {p['j']}) ref={p['ref']:.6f} ollama={p['target']:.6f} diff={p['diff']:.6f}")

    print("\n================ Sanity Checks ================")
    print("Reference:")
    run_sanity_block("Ref Raw", ref_raw_cosines)
    run_sanity_block("Ref Passage", ref_passage_cosines)
    print("Ollama:")
    run_sanity_block("Ollama Raw", ollama_raw_cosines)
    run_sanity_block("Ollama Passage", ollama_passage_cosines)

    # Check gates
    headline = {
        "raw": {"max_diff": raw_max_diff, "spearman": raw_spearman},
        "passage": {"max_diff": pass_max_diff, "spearman": pass_spearman},
        "gated_variant": gate_variant,
    }

    gated_diff = pass_max_diff if gate_variant == "passage" else raw_max_diff
    gated_spearman = pass_spearman if gate_variant == "passage" else raw_spearman

    gate_failed = False
    print(f"\n--- Gate Verification (Gated Variant: {gate_variant.upper()}) ---")
    if gated_diff > 0.02:
        print(
            f"ERROR: Gated variant '{gate_variant}' max diff {gated_diff:.6f} exceeds threshold 0.02!",
            file=sys.stderr,
        )
        gate_failed = True
    else:
        print(f"✓ Max diff check passed: {gated_diff:.6f} <= 0.02")

    if gated_spearman < 0.98:
        print(
            f"ERROR: Gated variant '{gate_variant}' Spearman {gated_spearman:.6f} below threshold 0.98!",
            file=sys.stderr,
        )
        gate_failed = True
    else:
        print(f"✓ Spearman check passed: {gated_spearman:.6f} >= 0.98")

    print(f"PROBE_B_JSON: {json.dumps(headline, separators=(',', ':'))}")

    if gate_failed:
        sys.exit(1)

    print(f"✓ Probe B passed: Variant '{gate_variant}' matches reference BF16 within gate thresholds.")


if __name__ == "__main__":
    main()

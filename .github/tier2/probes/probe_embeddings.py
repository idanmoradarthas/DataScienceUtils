#!/usr/bin/env python3
"""Probe A: Embeddings Endpoint Validation.

Verifies:
- POST /v1/embeddings returns 2048-dimensional vectors.
- Vectors are unit-length (L2 norm approx 1.0 within 0.05).
- Batch of 8 texts returns indices 0..7 in order.
"""

from __future__ import annotations

import math
import os
import sys

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package is required for probe_embeddings.py", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    """Run embedding endpoint probe checks."""
    base_url = os.environ.get("SKILL_EVAL_EMBEDDING_BASE_URL", "http://127.0.0.1:11434/v1")
    api_key = os.environ.get("SKILL_EVAL_EMBEDDING_API_KEY", "ollama")
    model = os.environ.get("SKILL_EVAL_EMBEDDING_MODEL", "nemotron-3-embed-1b")

    client = OpenAI(base_url=base_url, api_key=api_key)

    test_batch = [
        "First test text for embeddings validation.",
        "Second text focusing on machine learning and evaluation.",
        "Third document discussing metrics such as precision and recall.",
        "Fourth input checking clustering of high dimensional data.",
        "Fifth sentence exploring explainability in gradient boosting trees.",
        "Sixth description of string preprocessing and feature encoding.",
        "Seventh workflow guide covering exploratory data analysis.",
        "Eighth chunk testing batch ordering and unit normalization.",
    ]

    print(f"Sending batch of {len(test_batch)} texts to {base_url} (model: {model})...")
    resp = client.embeddings.create(input=test_batch, model=model)

    data = resp.data
    if len(data) != len(test_batch):
        print(f"Error: Expected {len(test_batch)} embeddings, got {len(data)}", file=sys.stderr)
        sys.exit(1)

    for i, item in enumerate(data):
        if item.index != i:
            print(f"Error: Batch item out of order. Expected index {i}, got {item.index}", file=sys.stderr)
            sys.exit(1)

        vec = item.embedding
        if len(vec) != 2048:
            print(f"Error: Expected 2048 dimensions, got {len(vec)} for item {i}", file=sys.stderr)
            sys.exit(1)

        norm = math.sqrt(sum(x * x for x in vec))
        if abs(norm - 1.0) > 0.05:
            print(f"Error: Vector {i} norm is {norm:.4f}, expected unit-norm (~1.0)", file=sys.stderr)
            sys.exit(1)

    print(
        f"✓ Probe A passed: Batch of {len(test_batch)} embeddings returned in order, dimension 2048, unit-normalized."
    )


if __name__ == "__main__":
    main()

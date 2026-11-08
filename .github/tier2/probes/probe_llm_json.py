#!/usr/bin/env python3
"""Probe C: LLM Structured Output and Schema Enforcement Validation.

Verifies:
- POST /v1/chat/completions enforces strict json_schema response_format.
- Parses valid JSON response with fields: verdict, confidence, rationale, suggestion.
- Verdict is one of: DUPLICATE, INTENTIONAL_DETAIL, RELATED_BUT_DISTINCT.
- Measures and prints latency and schema adherence.
"""

from __future__ import annotations

import json
import os
import sys
import time

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package is required for probe_llm_json.py", file=sys.stderr)
    sys.exit(1)

VERDICT_JSON_SCHEMA = {
    "name": "dedup_verdict",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["DUPLICATE", "INTENTIONAL_DETAIL", "RELATED_BUT_DISTINCT"],
            },
            "confidence": {"type": "number"},
            "rationale": {"type": "string"},
            "suggestion": {"type": "string"},
        },
        "required": ["verdict", "confidence", "rationale", "suggestion"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = """You are a technical content analyst for AI agent skill packages.
Classify the content overlap as DUPLICATE, INTENTIONAL_DETAIL, or RELATED_BUT_DISTINCT.
Respond with ONLY a JSON object adhering to the schema."""

USER_PROMPT = """---CHUNK 1---
File: SKILL.md
Section: Feature Scaling
Lines: 10-15
Content:
StandardScaler normalizes features to zero mean and unit variance.

---CHUNK 2---
File: SKILL.md
Section: Standardization
Lines: 40-45
Content:
StandardScaler normalizes features to zero mean and unit variance.

Embedding similarity: 0.995

Classify this content overlap."""


def main() -> None:
    """Run Probe C checks on LLM structured JSON output."""
    base_url = os.environ.get("SKILL_EVAL_LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    api_key = os.environ.get("SKILL_EVAL_LLM_API_KEY", "ollama")
    model = os.environ.get("SKILL_EVAL_LLM_MODEL", "gemma4:e2b")

    client = OpenAI(base_url=base_url, api_key=api_key)

    print(f"Sending chat completion request with strict JSON schema to {base_url} (model: {model})...")
    start_time = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
            temperature=0.0,
            response_format={"type": "json_schema", "json_schema": VERDICT_JSON_SCHEMA},
        )
    except Exception as exc:
        print(f"Error: chat completions call failed: {exc}", file=sys.stderr)
        sys.exit(1)

    latency = time.time() - start_time
    choice = resp.choices[0]
    content = choice.message.content or ""

    print(f"Response received in {latency:.2f}s")
    print(f"Raw content: {content}")

    try:
        data = json.loads(content)
    except Exception as err:
        print(f"Error: Content did not parse as valid JSON: {err}", file=sys.stderr)
        sys.exit(1)

    verdict = data.get("verdict")
    valid_verdicts = {"DUPLICATE", "INTENTIONAL_DETAIL", "RELATED_BUT_DISTINCT"}
    if verdict not in valid_verdicts:
        print(f"Error: Verdict '{verdict}' not in {valid_verdicts}", file=sys.stderr)
        sys.exit(1)

    required_keys = {"verdict", "confidence", "rationale", "suggestion"}
    missing = required_keys - set(data.keys())
    if missing:
        print(f"Error: JSON missing required keys: {missing}", file=sys.stderr)
        sys.exit(1)

    conf = data.get("confidence")
    print(
        f"✓ Probe C passed: Strict JSON schema enforced "
        f"(latency: {latency:.2f}s, verdict: {verdict}, confidence: {conf})"
    )


if __name__ == "__main__":
    main()

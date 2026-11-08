#!/usr/bin/env python3
"""Hermetic Mock OpenAI-compatible server for testing SkillEvaluator Tier 2.

Stdlib only.
Serves:
  POST /v1/embeddings
    Deterministic hashed bag-of-words vectors, L2-normalized, 2048 dimensions.
  POST /v1/chat/completions
    Returns chat completion JSON adhering to SkillEvaluator dedup verdict schema:
    {"verdict": VERDICT, "confidence": 0.9, "rationale": "mock", "suggestion": "none"}
    where VERDICT comes from MOCK_VERDICT (default: RELATED_BUT_DISTINCT).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class MockOpenAIHandler(BaseHTTPRequestHandler):
    """HTTP handler mimicking OpenAI embeddings and chat completions endpoints."""

    def log_message(self, format: str, *args: object) -> None:
        """Suppress standard HTTP server logging to stderr unless DEBUG=1."""
        if os.environ.get("DEBUG") == "1":
            super().log_message(format, *args)

    def do_POST(self) -> None:
        """Handle incoming POST requests to /v1/embeddings and /v1/chat/completions."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body.decode("utf-8"))
        except Exception:
            data = {}

        if self.path.endswith("/embeddings"):
            inputs = data.get("input", [])
            if isinstance(inputs, str):
                inputs = [inputs]
            model = data.get("model", "nemotron-3-embed-1b")

            record_file = os.environ.get("MOCK_RECORD_FILE")
            if record_file:
                record = {
                    "input": inputs,
                    "keys": list(data.keys()),
                }
                with open(record_file, "a", encoding="utf-8") as rf:
                    rf.write(json.dumps(record) + "\n")

            dim = 2048
            embeddings_data = []
            for idx, text in enumerate(inputs):
                vec = [0.0] * dim
                words = str(text).split()
                if not words:
                    vec[0] = 1.0
                else:
                    for word in words:
                        h = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
                        vec[h % dim] += 1.0
                    norm = math.sqrt(sum(x * x for x in vec))
                    if norm > 0:
                        vec = [x / norm for x in vec]
                    else:
                        vec[0] = 1.0
                embeddings_data.append(
                    {
                        "object": "embedding",
                        "index": idx,
                        "embedding": vec,
                    }
                )

            resp = {
                "object": "list",
                "data": embeddings_data,
                "model": model,
                "usage": {"prompt_tokens": len(inputs) * 5, "total_tokens": len(inputs) * 5},
            }
            resp_bytes = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_bytes)))
            self.end_headers()
            self.wfile.write(resp_bytes)

        elif self.path.endswith("/chat/completions"):
            # Check for runtime verdict override file if present, else fallback to MOCK_VERDICT env var
            verdict_file = os.environ.get("MOCK_VERDICT_FILE")
            verdict = "RELATED_BUT_DISTINCT"
            if verdict_file and os.path.exists(verdict_file):
                try:
                    with open(verdict_file) as vf:
                        verdict = vf.read().strip()
                except Exception:
                    verdict = os.environ.get("MOCK_VERDICT", "RELATED_BUT_DISTINCT")
            else:
                verdict = os.environ.get("MOCK_VERDICT", "RELATED_BUT_DISTINCT")

            resp_content = json.dumps(
                {
                    "verdict": verdict,
                    "confidence": 0.9,
                    "rationale": "mock rationale",
                    "suggestion": "none",
                }
            )
            resp = {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": 1700000000,
                "model": data.get("model", "mock-model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": resp_content,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }
            resp_bytes = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_bytes)))
            self.end_headers()
            self.wfile.write(resp_bytes)

        else:
            self.send_response(404)
            self.end_headers()


def run_server() -> None:
    """Start the mock server on the requested or dynamic port."""
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    server = HTTPServer(("127.0.0.1", port), MockOpenAIHandler)
    actual_port = server.server_address[1]
    # Print port to stdout and flush so launcher scripts can capture it
    print(actual_port, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()

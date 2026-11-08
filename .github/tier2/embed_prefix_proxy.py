#!/usr/bin/env python3
"""Passage-prefix embedding proxy for Ollama / OpenAI-compatible endpoints.

Stdlib only.
Listens on 127.0.0.1:${TIER2_EMBED_PROXY_PORT:-11435}.
Upstream: ${TIER2_UPSTREAM:-http://127.0.0.1:11434}.

Behavior:
  - POST /v1/embeddings:
      Parse JSON. If "input" is present (str or list[str]), prepend TIER2_EMBED_PREFIX
      (default: "passage: ") to each string.
      Remove "input_type" key if present.
      Forward to upstream, returning upstream status, body, and Content-Type unchanged.
  - All other HTTP methods and paths:
      Transparent pass-through to upstream.
  - Upstream error / unreachable:
      Return HTTP 502 with a short error message.
  - Upstream timeout: 600 seconds.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROXY_PORT = int(os.environ.get("TIER2_EMBED_PROXY_PORT", "11435"))
UPSTREAM = os.environ.get("TIER2_UPSTREAM", "http://127.0.0.1:11434").rstrip("/")
EMBED_PREFIX = os.environ.get("TIER2_EMBED_PREFIX", "passage: ")
TIMEOUT = 600.0


class EmbeddingProxyHandler(BaseHTTPRequestHandler):
    """Proxy HTTP handler forwarding to upstream with passage prefix rewriting on /v1/embeddings."""

    def log_message(self, format: str, *args: object) -> None:
        """Suppress default HTTP logging unless DEBUG=1."""
        if os.environ.get("DEBUG") == "1":
            super().log_message(format, *args)

    def do_GET(self) -> None:
        """Handle GET requests by forwarding to upstream."""
        self._proxy_request("GET")

    def do_POST(self) -> None:
        """Handle POST requests, rewriting /v1/embeddings or forwarding directly."""
        if self.path.rstrip("/") == "/v1/embeddings" or self.path == "/v1/embeddings":
            self._handle_embeddings()
        else:
            self._proxy_request("POST")

    def do_HEAD(self) -> None:
        """Handle HEAD requests by forwarding to upstream."""
        self._proxy_request("HEAD")

    def do_PUT(self) -> None:
        """Handle PUT requests by forwarding to upstream."""
        self._proxy_request("PUT")

    def do_DELETE(self) -> None:
        """Handle DELETE requests by forwarding to upstream."""
        self._proxy_request("DELETE")

    def do_OPTIONS(self) -> None:
        """Handle OPTIONS requests by forwarding to upstream."""
        self._proxy_request("OPTIONS")

    def _handle_embeddings(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body.decode("utf-8"))
        except Exception as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            resp = json.dumps({"error": f"Invalid JSON body: {e}"}).encode("utf-8")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
            return

        if "input" in data:
            inp = data["input"]
            if isinstance(inp, str):
                data["input"] = f"{EMBED_PREFIX}{inp}"
            elif isinstance(inp, list):
                data["input"] = [f"{EMBED_PREFIX}{item}" for item in inp]

        # Remove input_type if present
        data.pop("input_type", None)

        forward_body = json.dumps(data).encode("utf-8")
        headers = {}
        for k, v in self.headers.items():
            k_lower = k.lower()
            if k_lower in ("host", "content-length"):
                continue
            headers[k] = v
        headers["Content-Length"] = str(len(forward_body))
        headers["Content-Type"] = "application/json"

        target_url = f"{UPSTREAM}{self.path}"
        req = urllib.request.Request(target_url, data=forward_body, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                resp_status = resp.status
                resp_headers = resp.headers
                resp_body = resp.read()
        except urllib.error.HTTPError as e:
            resp_status = e.code
            resp_headers = e.headers
            resp_body = e.read()
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            err_msg = f"Bad Gateway: error forwarding to upstream: {e}\n".encode("utf-8")
            self.send_header("Content-Length", str(len(err_msg)))
            self.end_headers()
            self.wfile.write(err_msg)
            return

        self.send_response(resp_status)
        content_type = resp_headers.get("Content-Type", "application/json")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        self.wfile.write(resp_body)

    def _proxy_request(self, method: str) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        headers = {}
        for k, v in self.headers.items():
            k_lower = k.lower()
            if k_lower in ("host", "content-length"):
                continue
            headers[k] = v
        if body is not None:
            headers["Content-Length"] = str(len(body))

        target_url = f"{UPSTREAM}{self.path}"
        req = urllib.request.Request(target_url, data=body, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                resp_status = resp.status
                resp_headers = resp.headers
                resp_body = resp.read()
        except urllib.error.HTTPError as e:
            resp_status = e.code
            resp_headers = e.headers
            resp_body = e.read()
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            err_msg = f"Bad Gateway: error forwarding to upstream: {e}\n".encode("utf-8")
            self.send_header("Content-Length", str(len(err_msg)))
            self.end_headers()
            self.wfile.write(err_msg)
            return

        self.send_response(resp_status)
        if "Content-Type" in resp_headers:
            self.send_header("Content-Type", resp_headers["Content-Type"])
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        if method != "HEAD":
            self.wfile.write(resp_body)


def main() -> None:
    """Run the embedding proxy server."""
    server = ThreadingHTTPServer(("127.0.0.1", PROXY_PORT), EmbeddingProxyHandler)
    actual_port = server.server_address[1]
    print(f"listening on {actual_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

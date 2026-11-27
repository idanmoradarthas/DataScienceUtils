"""A fake Ollama server for tests: the real HTTP API on a local port, with a scriptable "model" behind it.

It checks the client, the tool loop and the generator against the protocol (endpoints, tool calls, schema
``format``, token counts, durations, silent truncation), not against model quality, which only a real model can show.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import threading
from typing import Any, Callable

NANOSECONDS = 1_000_000_000


def reply(
    content: str = "",
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    prompt_tokens: int = 100,
    output_tokens: int = 20,
    done_reason: str = "stop",
) -> dict[str, Any]:
    """Build a ``/api/chat`` response with server-style counts and durations."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [{"function": {"name": name, "arguments": arguments}} for name, arguments in tool_calls]
    return {
        "model": "fake",
        "message": message,
        "done": True,
        "done_reason": done_reason,
        "prompt_eval_count": prompt_tokens,
        "eval_count": output_tokens,
        "load_duration": NANOSECONDS // 2,
        "prompt_eval_duration": max(prompt_tokens, 1) * NANOSECONDS // 1000,
        "eval_duration": max(output_tokens, 1) * NANOSECONDS // 100,
        "total_duration": 2 * NANOSECONDS,
    }


class HttpFailure(Exception):
    """Raised by a responder to make the server answer with an HTTP error."""

    def __init__(self, status: int, message: str) -> None:
        """Keep the status and the error text the server will send."""
        super().__init__(message)
        self.status = status
        self.message = message


Responder = Callable[[dict[str, Any], "FakeOllama"], dict[str, Any]]


class FakeOllama:
    """Run the fake server in a thread; use as a context manager. ``host`` is ``127.0.0.1:<port>``."""

    def __init__(
        self,
        responder: Responder,
        *,
        models: tuple[str, ...] = ("fake:1b",),
        capabilities: tuple[str, ...] = ("completion", "tools"),
        trained_context: int = 131072,
    ) -> None:
        """Configure the server; nothing listens until :meth:`start`."""
        self.responder = responder
        self.models = models
        self.capabilities = list(capabilities)
        self.trained_context = trained_context
        self.requests: list[dict[str, Any]] = []
        self.last_num_ctx = 0
        self.max_window: int | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:  # silence the test output
                return

            def _send(self, status: int, body: dict[str, Any]) -> None:
                data = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:  # noqa: N802 - http.server API
                if self.path == "/api/version":
                    self._send(200, {"version": "0.0.0-fake"})
                elif self.path == "/api/tags":
                    self._send(200, {"models": [{"name": name} for name in outer.models]})
                elif self.path == "/api/ps":
                    loaded = min(outer.last_num_ctx, outer.max_window or outer.last_num_ctx)
                    models = [{"name": outer.models[0], "context_length": loaded}] if loaded else []
                    self._send(200, {"models": models})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802 - http.server API
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/api/show":
                    self._send(
                        200,
                        {
                            "capabilities": outer.capabilities,
                            "details": {"family": "fake", "parameter_size": "1B", "quantization_level": "Q4_K_M"},
                            "model_info": {"fake.context_length": outer.trained_context},
                        },
                    )
                elif self.path == "/api/chat":
                    outer.requests.append(payload)
                    outer.last_num_ctx = int((payload.get("options") or {}).get("num_ctx") or 0)
                    try:
                        self._send(200, outer.responder(payload, outer))
                    except HttpFailure as failure:
                        self._send(failure.status, {"error": failure.message})
                else:
                    self._send(404, {"error": "not found"})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._started = False

    @property
    def host(self) -> str:
        """Return ``127.0.0.1:<port>``."""
        return f"127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> FakeOllama:
        """Start serving."""
        self._thread.start()
        self._started = True
        return self

    def stop(self) -> None:
        """Stop serving and release the port; safe to call when the server was never started, or twice."""
        if self._started:  # ``shutdown`` waits for ``serve_forever``: it blocks forever on a server never started
            self._started = False
            self._server.shutdown()
            self._thread.join(timeout=5)
        self._server.server_close()

    def __enter__(self) -> FakeOllama:
        """Start the server for a ``with`` block."""
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        """Stop the server."""
        self.stop()


class ScriptedModel:
    """A deterministic stand-in for a model that understands the tool loop and the generator prompts.

    * ``chars_per_token`` and ``max_window`` emulate the tokenizer and an Ollama that silently keeps only the last
      ``max_window`` tokens of a prompt (and reports that many in ``prompt_eval_count``);
    * ``tool_script`` is the list of ``(tool name, arguments)`` calls it makes while exploring a repository;
    * ``notes`` is the JSON it returns for the release-notes schema.
    """

    def __init__(
        self,
        *,
        chars_per_token: float = 3.0,
        max_window: int | None = None,
        supports_tools: bool = True,
        tool_script: list[tuple[str, dict[str, Any]]] | None = None,
        notes: dict[str, Any] | None = None,
        prompt_tokens_override: int | None = None,
    ) -> None:
        """Store the behavior; the model itself has no state besides the request log of its server."""
        self.chars_per_token = chars_per_token
        self.max_window = max_window
        self.supports_tools = supports_tools
        self.tool_script = tool_script or []
        self.notes = notes or {}
        self.prompt_tokens_override = prompt_tokens_override

    def __call__(self, payload: dict[str, Any], server: FakeOllama) -> dict[str, Any]:
        """Answer one chat request."""
        options = payload.get("options") or {}
        window = int(options.get("num_ctx") or 4096)
        if self.max_window:
            server.max_window = self.max_window
            window = min(window, self.max_window)
        messages = payload["messages"]
        text = "\n".join(str(message.get("content") or "") for message in messages)
        tokens = int(len(text) / self.chars_per_token) + 20
        prompt_tokens = self.prompt_tokens_override or min(tokens, window)
        tools = payload.get("tools") or []
        schema = payload.get("format")
        if tools and not self.supports_tools:
            raise HttpFailure(400, f"registry.ollama.ai/library/{payload['model']} does not support tools")

        def make(content: str = "", **kwargs: Any) -> dict[str, Any]:
            return reply(content, prompt_tokens=prompt_tokens, **kwargs)

        names = {tool["function"]["name"] for tool in tools}
        called = sum(1 for message in messages if message.get("tool_calls"))
        if tools and schema is None and "changed_files" in names:
            if called < len(self.tool_script):
                return make(tool_calls=[self.tool_script[called]])
            return make("I verified the facts with the tools.")
        if isinstance(schema, dict):
            return make(json.dumps(self._structured(schema, messages)))
        return make("ok")

    def _structured(self, schema: dict[str, Any], messages: list[dict[str, Any]]) -> Any:
        """Return JSON for the schemas the agent tests and the generator use."""
        properties = set(schema.get("properties", {}))
        if properties == {"function", "first_parameter"}:
            tool_text = "\n".join(str(m.get("content")) for m in messages if m.get("role") == "tool")
            match = re.search(r"(plot_zebra_curve)\((\w+)", tool_text)
            return (
                {"function": match.group(1), "first_parameter": match.group(2)}
                if match
                else {"function": "?", "first_parameter": "?"}
            )
        if "release_title" in properties:
            return self.notes
        return {name: "x" for name in properties}

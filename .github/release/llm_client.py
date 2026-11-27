"""Minimal Ollama client for the release tooling (standard library only).

Ollama is a model server: it answers ``/api/chat`` requests and nothing else. It cannot browse a repository by itself.
What it can do is *tool calling*: the request lists tools, the model answers with ``message.tool_calls`` instead of
text, the client runs the tools and sends the results back as ``role: "tool"`` messages. This module is the thin
transport for that protocol, plus the pieces the release pipeline needs around it:

* ``show`` / ``ps`` / ``version`` / ``list_models`` to ask the server what a model supports and what is loaded
  (the ``tools`` capability, the trained context length, the context actually allocated);
* ``chat`` for one request, with optional ``tools`` and an optional JSON ``schema`` (constrained output) and the
  server-reported timings, so speed and truncation can be measured instead of guessed;
* ``chat_json`` for schema-validated JSON with one corrective retry.

The default context window of Ollama is small (4096 tokens on a machine without a large GPU), so ``num_ctx`` is always
sent explicitly with every request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import http.client
import json
import os
import re
import time
from typing import Any
import urllib.error
import urllib.request

DEFAULT_OLLAMA_HOST = "127.0.0.1:11434"
NANOSECONDS = 1e9
CHARS_PER_TOKEN = 3.5
MESSAGE_OVERHEAD_TOKENS = 8  # role markers and separators that a chat template adds around every message


def estimate_tokens(text: str) -> int:
    """Rough token count of a text (the server-reported counts are preferred whenever they exist)."""
    return int(len(text) / CHARS_PER_TOKEN) + 1


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """Rough token count of one chat message, including the tool calls it carries."""
    return estimate_tokens(str(message.get("content") or "") + json.dumps(message.get("tool_calls") or [])) + (
        MESSAGE_OVERHEAD_TOKENS
    )


class LLMError(RuntimeError):
    """The model server failed or never produced valid JSON for the requested schema."""

    def __init__(self, message: str, *, status: int | None = None, body: str = "") -> None:
        """Keep the HTTP status and the server message next to the text, so callers can react to them."""
        super().__init__(message)
        self.status = status
        self.body = body


def validate_schema(schema: dict[str, Any], value: Any, path: str = "$") -> list[str]:
    """Validate ``value`` against a small JSON-schema subset and return human-readable errors.

    Supported keywords: type (object, array, string, integer, number, boolean), enum, required, properties, items,
    minItems, maxItems, minLength, maxLength. Unknown keywords are ignored.
    """
    errors: list[str] = []
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            return [f"{path}: expected an object"]
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}: missing required field '{name}'")
        for name, sub_schema in schema.get("properties", {}).items():
            if name in value:
                errors.extend(validate_schema(sub_schema, value[name], f"{path}.{name}"))
    elif expected == "array":
        if not isinstance(value, list):
            return [f"{path}: expected an array"]
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: needs at least {schema['minItems']} item(s)")
        if len(value) > schema.get("maxItems", len(value)):
            errors.append(f"{path}: allows at most {schema['maxItems']} item(s)")
        for index, entry in enumerate(value):
            errors.extend(validate_schema(schema.get("items", {}), entry, f"{path}[{index}]"))
    elif expected == "string":
        if not isinstance(value, str):
            return [f"{path}: expected a string"]
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path}: shorter than {schema['minLength']} characters")
        if len(value) > schema.get("maxLength", len(value)):
            errors.append(f"{path}: longer than {schema['maxLength']} characters")
    elif expected == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return [f"{path}: expected an integer"]
    elif expected == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return [f"{path}: expected a number"]
    elif expected == "boolean" and not isinstance(value, bool):
        return [f"{path}: expected a boolean"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: must be one of {schema['enum']}")
    return errors


def parse_json(text: str) -> Any:
    """Parse model output as JSON, tolerating a Markdown code fence around it."""
    stripped = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    return json.loads(stripped)


@dataclass
class ToolCall:
    """One tool call requested by the model."""

    name: str
    arguments: dict[str, Any]
    error: str = ""


@dataclass
class ChatResult:
    """One answer of the server, with the numbers it reported about producing it."""

    content: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    message: dict[str, Any] = field(default_factory=dict)
    done_reason: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    load_seconds: float = 0.0
    prompt_seconds: float = 0.0
    eval_seconds: float = 0.0

    @property
    def prompt_tokens_per_second(self) -> float:
        """Speed of reading the prompt (0 when the server did not report it)."""
        return self.prompt_tokens / self.prompt_seconds if self.prompt_seconds else 0.0

    @property
    def output_tokens_per_second(self) -> float:
        """Speed of writing the answer (0 when the server did not report it)."""
        return self.output_tokens / self.eval_seconds if self.eval_seconds else 0.0


@dataclass
class Usage:
    """Accumulated cost of the calls made through one client."""

    calls: int = 0
    retries: int = 0
    failures: int = 0
    tool_calls: int = 0
    seconds: float = 0.0
    load_seconds: float = 0.0
    prompt_seconds: float = 0.0
    eval_seconds: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0


def _tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    """Read ``message.tool_calls``; arguments may arrive as an object or as a JSON string.

    A missing ``tool_calls`` and ``null`` mean "no calls". Anything else that is not a list of objects with a
    ``function`` object raises :class:`LLMError`: that is a broken reply, not a mistake of the model, so there is
    nothing to tell the model about it.
    """
    entries = message.get("tool_calls")
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise LLMError("The server sent tool_calls that are not a list")
    calls: list[ToolCall] = []
    for entry in entries:
        function = entry.get("function") if isinstance(entry, dict) else None
        if not isinstance(function, dict):
            raise LLMError("The server sent a tool call that is not an object with a 'function' object")
        raw = function.get("arguments") or {}
        error = ""
        if isinstance(raw, str):
            try:
                raw = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError as problem:
                error = f"arguments are not valid JSON ({problem.msg})"
                raw = {}
        if not isinstance(raw, dict):
            error = "arguments are not an object"
            raw = {}
        calls.append(ToolCall(name=str(function.get("name") or ""), arguments=raw, error=error))
    return calls


class OllamaClient:
    """Talk to one Ollama model: chat, tool calling, schema-constrained JSON and server introspection."""

    def __init__(
        self,
        model: str,
        *,
        host: str | None = None,
        temperature: float = 0.0,
        seed: int = 7,
        num_ctx: int = 16384,
        num_predict: int = 2048,
        num_thread: int = 0,
        think: bool | None = None,
        keep_alive: str = "30m",
        timeout: float = 1800.0,
    ) -> None:
        """Configure the client; nothing is sent until a method is called."""
        host = host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
        self.base_url = host if "://" in host else f"http://{host}"
        self.model = model
        self.options: dict[str, Any] = {
            "temperature": temperature,
            "seed": seed,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        }
        if num_thread > 0:  # 0 leaves the choice to Ollama
            self.options["num_thread"] = num_thread
        self.think = think
        self.keep_alive = keep_alive
        self.timeout = timeout
        self.usage = Usage()

    @property
    def num_ctx(self) -> int:
        """The context window requested with every chat call."""
        return int(self.options["num_ctx"])

    def _request(self, path: str, payload: dict[str, Any] | None = None, *, timeout: float | None = None) -> Any:
        """GET (no payload) or POST (payload) one endpoint and return the decoded JSON."""
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=None if payload is None else json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            try:
                body = error.read().decode("utf-8", errors="replace")
            except (OSError, http.client.HTTPException):  # the connection broke while the error text was read
                body = ""
            raise LLMError(
                f"Ollama returned HTTP {error.code} for {path} (model '{self.model}'): {body[:300]}",
                status=error.code,
                body=body,
            ) from error
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError, http.client.HTTPException) as error:
            raise LLMError(f"Cannot reach Ollama at {self.base_url}: {error}") from error
        if not isinstance(data, dict):
            raise LLMError(f"Ollama answered {path} with JSON of type {type(data).__name__}, expected an object")
        return data

    def version(self) -> str:
        """Return the server version string."""
        return str(self._request("/api/version", timeout=30).get("version", "unknown"))

    def list_models(self) -> list[str]:
        """Return the names of the models installed on the server."""
        return [str(entry["name"]) for entry in self._request("/api/tags", timeout=30).get("models", [])]

    def show(self) -> dict[str, Any]:
        """Return the model details: capabilities, parameters, template and ``model_info`` (architecture, context)."""
        return self._request("/api/show", {"model": self.model}, timeout=60)

    def capabilities(self) -> list[str]:
        """Return the capability names the server declares for the model (for example ``completion``, ``tools``)."""
        return [str(item) for item in self.show().get("capabilities") or []]

    def trained_context_length(self) -> int | None:
        """Return the context length the model was trained for, taken from ``model_info``, when the server tells."""
        for key, value in (self.show().get("model_info") or {}).items():
            if key.endswith(".context_length") and isinstance(value, int):
                return value
        return None

    def ps(self) -> list[dict[str, Any]]:
        """Return the models that are loaded right now, with their allocated context and memory placement."""
        return list(self._request("/api/ps", timeout=30).get("models", []))

    def loaded_context_length(self) -> int | None:
        """Return the context length allocated for this model while it is loaded, when the server tells."""
        for entry in self.ps():
            if entry.get("name") in {self.model, f"{self.model}:latest"} or entry.get("model") == self.model:
                value = entry.get("context_length")
                return int(value) if isinstance(value, int) else None
        return None

    def ensure_ready(self) -> None:
        """Raise :class:`LLMError` unless the server answers and has the model."""
        names = set(self.list_models())
        if not names & {self.model, f"{self.model}:latest"}:
            raise LLMError(f"Model '{self.model}' is not available on the server (found: {sorted(names) or 'none'}).")

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST one chat request; if the server rejects the ``think`` flag, repeat the request without it."""
        try:
            return self._request("/api/chat", payload)
        except LLMError as error:
            if error.status == 400 and "think" in error.body.lower() and "think" in payload:
                del payload["think"]
                self.think = None
                return self._request("/api/chat", payload)
            raise

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ChatResult:
        """Send ``messages`` once. ``tools`` are tool definitions, ``schema`` constrains the answer to JSON.

        ``options`` override the client defaults for this call only (for example ``num_ctx`` in the probe).
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": messages,
            "options": {**self.options, **(options or {})},
            "keep_alive": self.keep_alive,
        }
        if tools:
            payload["tools"] = tools
        if schema is not None:
            payload["format"] = schema
        if self.think is not None:
            payload["think"] = self.think
        started = time.monotonic()
        response = self._post_chat(payload)
        reply = response.get("message")
        if reply is None:  # an answer without a message is an empty one
            reply = {}
        if not isinstance(reply, dict):
            raise LLMError("The server sent a message that is not an object")
        message = dict(reply)
        message.setdefault("role", "assistant")
        calls = _tool_calls(message)
        if calls:
            # The message goes back to the server in the next request, which wants arguments as an object, not text.
            message["tool_calls"] = [
                {**entry, "function": {**entry["function"], "arguments": call.arguments}}
                for entry, call in zip(message["tool_calls"], calls)
            ]
        result = ChatResult(
            content=str(message.get("content") or ""),
            thinking=str(message.get("thinking") or ""),
            tool_calls=calls,
            message=message,
            done_reason=str(response.get("done_reason") or ""),
            prompt_tokens=int(response.get("prompt_eval_count") or 0),
            output_tokens=int(response.get("eval_count") or 0),
            seconds=time.monotonic() - started,
            load_seconds=float(response.get("load_duration") or 0) / NANOSECONDS,
            prompt_seconds=float(response.get("prompt_eval_duration") or 0) / NANOSECONDS,
            eval_seconds=float(response.get("eval_duration") or 0) / NANOSECONDS,
        )
        self.usage.calls += 1
        self.usage.tool_calls += len(result.tool_calls)
        self.usage.seconds += result.seconds
        self.usage.load_seconds += result.load_seconds
        self.usage.prompt_seconds += result.prompt_seconds
        self.usage.eval_seconds += result.eval_seconds
        self.usage.prompt_tokens += result.prompt_tokens
        self.usage.output_tokens += result.output_tokens
        return result

    def chat_json(self, system: str, user: str, schema: dict[str, Any], *, attempts: int = 2) -> Any:
        """Answer one system and user message with parsed JSON that satisfies ``schema``."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.chat_json_messages(messages, schema, attempts=attempts)[0]

    def chat_json_messages(
        self, messages: list[dict[str, Any]], schema: dict[str, Any], *, attempts: int = 2
    ) -> tuple[Any, ChatResult]:
        """Continue a conversation with a schema-constrained answer; return the parsed JSON and the last result.

        After an invalid answer the conversation continues with the list of problems, up to ``attempts`` calls in
        total. Raises :class:`LLMError` when no attempt produced a valid answer.
        """
        history = list(messages)
        problems: list[str] = []
        no_room = ""
        made = 0
        original_tokens = 0  # size of ``messages``: every corrective request is rebuilt from them, not from ``history``
        for attempt in range(attempts):
            result = self.chat(history, schema=schema)
            made += 1
            self.usage.retries += 1 if attempt else 0
            if not attempt:
                original_tokens = result.prompt_tokens or sum(estimate_message_tokens(message) for message in messages)
            try:
                parsed = parse_json(result.content)
            except json.JSONDecodeError as error:
                problems = [f"the answer is not valid JSON ({error.msg})"]
                if result.done_reason == "length":
                    problems.append("the answer was cut off at the output limit, so make it shorter")
            else:
                problems = validate_schema(schema, parsed)
                if not problems:
                    return parsed, result
            if attempt + 1 == attempts:
                break
            retry = self._retry_history(messages, original_tokens, result, problems)
            if retry is None:
                no_room = "; no room left in the context window for another attempt"
                break
            history = retry
        self.usage.failures += 1
        raise LLMError(
            f"Model '{self.model}' gave no valid JSON after {made} attempt(s): {'; '.join(problems[:3])}{no_room}"
        )

    def _retry_history(
        self, messages: list[dict[str, Any]], original_tokens: int, result: ChatResult, problems: list[str]
    ) -> list[dict[str, Any]] | None:
        """Return the conversation for a corrective attempt, or None when it cannot fit in the context window.

        The request is ``messages`` (``original_tokens`` long) plus, when there is room for it, the rejected answer,
        plus the list of problems. Ollama drops the *start* of a prompt that is too long (the instructions), so a
        prompt that does not fit is never sent.
        """
        correction = {
            "role": "user",
            "content": "Your previous answer was rejected: "
            + "; ".join(problems[:6])
            + ". Reply again with only the corrected JSON.",
        }
        extra = estimate_message_tokens(correction)
        answer = (result.output_tokens or estimate_tokens(result.content)) + MESSAGE_OVERHEAD_TOKENS
        room = self.num_ctx - int(self.options["num_predict"])
        if original_tokens + answer + extra <= room:
            return [*messages, {"role": "assistant", "content": result.content}, correction]
        return [*messages, correction] if original_tokens + extra <= room else None

"""A small tool-calling loop around :class:`llm_client.OllamaClient`.

Two phases, because the Ollama documentation does not say what happens when ``tools`` and a JSON ``format`` are sent in
the same request:

1. **Explore**: the model gets the tools and no ``format``. It calls tools until it has what it needs and answers in
   plain text (or the loop stops it).
2. **Answer**: the same conversation continues without tools and with the JSON schema as ``format``, so the final
   result is always machine-checkable.

The loop protects the context window. Ollama silently drops the oldest part of a prompt that does not fit in
``num_ctx``, which would remove the instructions the model is following. So before every round the loop estimates the
next prompt size (the server-reported size of the last prompt, the model's last answer and the new tool results) and
stops exploring while there is still room for the final JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import sys
import time
from typing import Any, Callable

from llm_client import ChatResult, LLMError, OllamaClient
from repo_tools import RepoTools

CHARS_PER_TOKEN = 3.5
FINAL_INSTRUCTION = (
    "Exploration is over. Using only what the tools and the context showed you, write the final answer now as JSON "
    "that follows the required schema. Do not call tools."
)
REPEAT_NOTICE = "(you already made this exact call; use the earlier result or try something different)"
MAX_REPEATS_IN_A_ROW = 3


@dataclass
class Step:
    """One executed tool call, for the log and the report."""

    index: int
    name: str
    arguments: dict[str, Any]
    output_chars: int
    seconds: float
    repeated: bool = False
    failed: bool = False


@dataclass
class AgentRun:
    """What the loop did and what it produced."""

    final: Any = None
    steps: list[Step] = field(default_factory=list)
    stop_reason: str = ""
    rounds: int = 0
    peak_prompt_tokens: int = 0
    exploration_note: str = ""
    final_result: ChatResult | None = None


def tools_unsupported(error: LLMError) -> bool:
    """Return True when the server refused the request because the model has no tool support."""
    return error.status == 400 and "tool" in error.body.lower()


def estimate_tokens(text: str) -> int:
    """Rough token count of a text (the server-reported counts are preferred whenever they exist)."""
    return int(len(text) / CHARS_PER_TOKEN) + 1


def _stderr(message: str) -> None:
    """Default progress sink: standard error, so standard output stays free for the changelog."""
    print(message, file=sys.stderr, flush=True)


def _summary(arguments: dict[str, Any]) -> str:
    """Short, single-line form of tool arguments for the log."""
    text = ", ".join(f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in arguments.items())
    return text if len(text) <= 120 else text[:117] + "..."


def run_agent(
    client: OllamaClient,
    tools: RepoTools,
    system: str,
    user: str,
    schema: dict[str, Any],
    *,
    max_rounds: int = 12,
    log: Callable[[str], None] = _stderr,
) -> AgentRun:
    """Explore with tools, then return the answer as JSON that satisfies ``schema``.

    ``max_rounds`` limits the number of model calls in the exploration phase. Raises :class:`LLMError` when the server
    fails or the final answer never validates; a model without tool support raises the server's 400 error unchanged
    (see :func:`tools_unsupported`).
    """
    run = AgentRun()
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    reserve = int(client.options["num_predict"]) + estimate_tokens(FINAL_INSTRUCTION) + 200
    tool_budget_tokens = estimate_tokens("x" * tools.max_output_chars)
    seen: set[str] = set()
    repeats = 0
    run.stop_reason = "max_rounds"
    for round_number in range(1, max_rounds + 1):
        result = client.chat(messages, tools=tools.schemas())
        run.rounds = round_number
        run.peak_prompt_tokens = max(run.peak_prompt_tokens, result.prompt_tokens)
        messages.append(result.message)
        if not result.tool_calls:
            run.exploration_note = result.content
            run.stop_reason = "explored"
            break
        tool_chars = 0
        for call in result.tool_calls:
            key = json.dumps([call.name, call.arguments], sort_keys=True)
            repeated = key in seen
            seen.add(key)
            index = len(run.steps) + 1
            started = time.monotonic()
            if call.error:
                output, failed = f"error: {call.error}", True
            elif repeated:
                output, failed = REPEAT_NOTICE, False
            else:
                before = tools.output_chars
                output = tools.call(call.name, call.arguments)
                failed = output.startswith("error:")
                tool_chars += tools.output_chars - before
            repeats = repeats + 1 if repeated else 0
            messages.append({"role": "tool", "tool_name": call.name, "content": output})
            seconds = time.monotonic() - started
            run.steps.append(
                Step(index, call.name, call.arguments, len(output), seconds, repeated=repeated, failed=failed)
            )
            log(
                f"[agent] round {round_number} step {index}: {call.name}({_summary(call.arguments)}) "
                f"-> {len(output)} chars in {seconds:.1f}s{' (repeat)' if repeated else ''}"
                f"{' (error)' if failed else ''}"
            )
        if repeats >= MAX_REPEATS_IN_A_ROW:
            run.stop_reason = "stuck_repeating"
            break
        prompt_so_far = result.prompt_tokens or sum(estimate_tokens(str(m.get("content", ""))) for m in messages)
        next_prompt = prompt_so_far + result.output_tokens + estimate_tokens("x" * tool_chars)
        if next_prompt + tool_budget_tokens + reserve > client.num_ctx:
            run.stop_reason = "context_budget"
            break
    log(f"[agent] exploration stopped: {run.stop_reason} after {run.rounds} round(s), {len(run.steps)} tool call(s)")
    messages.append({"role": "user", "content": FINAL_INSTRUCTION})
    run.final, run.final_result = client.chat_json_messages(messages, schema)
    run.peak_prompt_tokens = max(run.peak_prompt_tokens, run.final_result.prompt_tokens)
    return run

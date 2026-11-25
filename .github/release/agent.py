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

from llm_client import ChatResult, LLMError, OllamaClient, estimate_tokens
from repo_tools import RepoTools

FINAL_INSTRUCTION = (
    "Exploration is over. Using only what the tools and the context showed you, write the final answer now as JSON "
    "that follows the required schema. Do not call tools."
)
REPEAT_NOTICE = "(you already made this exact call; use the earlier result or try something different)"
NO_ROOM_NOTICE = "error: not run, the context window has no room left for more tool answers; write the answer now"
MAX_REPEATS_IN_A_ROW = 3
MESSAGE_OVERHEAD_TOKENS = 8


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


def _message_tokens(message: dict[str, Any]) -> int:
    """Estimate the size of one chat message, including the tool calls it carries."""
    return estimate_tokens(str(message.get("content") or "") + json.dumps(message.get("tool_calls") or [])) + (
        MESSAGE_OVERHEAD_TOKENS
    )


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
    fails, when the prompt and the answer limit cannot fit in the context window at all, or when the final answer never
    validates; a model without tool support raises the server's 400 error unchanged (see :func:`tools_unsupported`).
    """
    run = AgentRun()
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    reserve = int(client.options["num_predict"]) + estimate_tokens(FINAL_INSTRUCTION) + 200
    tool_budget_tokens = estimate_tokens("x" * tools.max_output_chars)
    initial_tokens = sum(_message_tokens(message) for message in messages)
    if initial_tokens + reserve > client.num_ctx:
        raise LLMError(
            f"The prompt (~{initial_tokens} tokens) and the answer limit ({client.options['num_predict']} tokens) do "
            f"not fit in the context window of {client.num_ctx} tokens; Ollama would drop the start of the prompt."
        )
    seen: set[str] = set()
    repeats = 0
    run.stop_reason = "max_rounds"
    rounds = max_rounds
    if initial_tokens + tool_budget_tokens + reserve > client.num_ctx:  # room for the answer, but not for a tool result
        run.stop_reason, rounds = "context_budget", 0
    for round_number in range(1, rounds + 1):
        result = client.chat(messages, tools=tools.schemas())
        run.rounds = round_number
        run.peak_prompt_tokens = max(run.peak_prompt_tokens, result.prompt_tokens)
        messages.append(result.message)
        if not result.tool_calls:
            run.exploration_note = result.content
            run.stop_reason = "explored"
            break
        # Tokens in the conversation so far; the results of this batch are added one by one, and a result that could
        # push the prompt past the window (with the answer still to come) is not fetched at all.
        used = (
            result.prompt_tokens + result.output_tokens
            if result.prompt_tokens
            else sum(_message_tokens(message) for message in messages)
        )
        out_of_room = False
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
            elif used + tool_budget_tokens + reserve > client.num_ctx:
                output, failed, out_of_room = NO_ROOM_NOTICE, True, True
            else:
                output = tools.call(call.name, call.arguments)
                failed = output.startswith("error:")
            repeats = repeats + 1 if repeated else 0
            message = {"role": "tool", "tool_name": call.name, "content": output}
            messages.append(message)
            used += _message_tokens(message)
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
        if out_of_room or used + tool_budget_tokens + reserve > client.num_ctx:
            run.stop_reason = "context_budget"
            break
    log(f"[agent] exploration stopped: {run.stop_reason} after {run.rounds} round(s), {len(run.steps)} tool call(s)")
    messages.append({"role": "user", "content": FINAL_INSTRUCTION})
    run.final, run.final_result = client.chat_json_messages(messages, schema)
    run.peak_prompt_tokens = max(run.peak_prompt_tokens, run.final_result.prompt_tokens)
    return run

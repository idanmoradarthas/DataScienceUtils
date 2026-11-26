"""A small tool-calling loop around :class:`llm_client.OllamaClient`.

Two phases, because the Ollama documentation does not say what happens when ``tools`` and a JSON ``format`` are sent in
the same request:

1. **Explore**: the model gets the tools and no ``format``. It calls tools until it has what it needs and answers in
   plain text (or the loop stops it).
2. **Answer**: the same conversation continues without tools and with the JSON schema as ``format``, so the final
   result is always machine-checkable.

The loop protects the context window. Ollama silently drops the oldest part of a prompt that does not fit in
``num_ctx``, which would remove the instructions the model is following. So the loop keeps count of the conversation
(the server-reported size of the last prompt, the model's last answer and every message added since) and lets nothing
into it that would leave too little room for the final JSON: a tool result is fetched only when its worst case fits,
and calls that do not fit are removed from the model's message, together with the answers that were never fetched.
Exploration stops while there is still room for the final JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import sys
import time
from typing import Any, Callable

from llm_client import (
    MESSAGE_OVERHEAD_TOKENS,
    ChatResult,
    LLMError,
    OllamaClient,
    estimate_message_tokens,
    estimate_tokens,
)
from repo_tools import RepoTools

FINAL_INSTRUCTION = (
    "Exploration is over. Using only what the tools and the context showed you, write the final answer now as JSON "
    "that follows the required schema. Do not call tools."
)
REPEAT_NOTICE = "(you already made this exact call; use the earlier result or try something different)"
MAX_REPEATS_IN_A_ROW = 3
ANSWER_SLACK_TOKENS = 200  # margin for the chat template and for estimates that are a little low


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
    dropped_calls: int = 0  # tool calls of the model that were not run because their answers would not fit
    exploration_note: str = ""
    final_result: ChatResult | None = None


def tools_unsupported(error: LLMError) -> bool:
    """Return True when the server refused the request because the model has no tool support."""
    return error.status == 400 and "tool" in error.body.lower()


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
    # What the final request needs after the conversation: the instruction, the answer itself and a margin.
    reserve = int(client.options["num_predict"]) + estimate_tokens(FINAL_INSTRUCTION) + ANSWER_SLACK_TOKENS
    tool_message_tokens = estimate_tokens("x" * tools.max_output_chars) + MESSAGE_OVERHEAD_TOKENS  # worst case of one
    schema_tokens = estimate_tokens(json.dumps(tools.schemas()))  # sent with every exploration request, not the answer
    initial_tokens = sum(estimate_message_tokens(message) for message in messages)
    if initial_tokens + reserve > client.num_ctx:
        raise LLMError(
            f"The prompt (~{initial_tokens} tokens) and the answer limit ({client.options['num_predict']} tokens) do "
            f"not fit in the context window of {client.num_ctx} tokens; Ollama would drop the start of the prompt."
        )
    seen: set[str] = set()
    repeats = 0
    run.stop_reason = "max_rounds"
    rounds = max_rounds
    if initial_tokens + schema_tokens + tool_message_tokens + reserve > client.num_ctx:
        run.stop_reason, rounds = "context_budget", 0  # room for the answer, but not for the tools and one result
    for round_number in range(1, rounds + 1):
        result = client.chat(messages, tools=tools.schemas())
        run.rounds = round_number
        run.peak_prompt_tokens = max(run.peak_prompt_tokens, result.prompt_tokens)
        messages.append(result.message)
        # Tokens in the conversation now, tool definitions included (so a little more than the final request needs).
        used = (
            result.prompt_tokens + result.output_tokens
            if result.prompt_tokens
            else sum(estimate_message_tokens(message) for message in messages) + schema_tokens
        )
        if not result.tool_calls:
            run.exploration_note = result.content
            run.stop_reason = "explored"
            if used + reserve > client.num_ctx:  # a long free-text answer must not crowd out the final one
                messages.pop()
                log("[agent] the exploration answer is left out of the final request: it does not fit the window")
            break
        answer_at, handled, out_of_room = len(messages) - 1, 0, False
        for call in result.tool_calls:
            key = json.dumps([call.name, call.arguments], sort_keys=True)
            repeated = key in seen
            # The text of an error or of a repeat notice is known; the answer of a real call is assumed to be as long
            # as a tool answer can be.
            known = f"error: {call.error}" if call.error else REPEAT_NOTICE if repeated else None
            cost = estimate_message_tokens({"content": known}) if known else tool_message_tokens
            if used + cost + reserve > client.num_ctx:
                out_of_room = True
                break
            seen.add(key)
            index = len(run.steps) + 1
            started = time.monotonic()
            failed = bool(call.error)
            if known is not None:
                output = known
            else:
                output = tools.call(call.name, call.arguments)
                failed = output.startswith("error:")
            repeats = repeats + 1 if repeated else 0
            message = {"role": "tool", "tool_name": call.name, "content": output}
            messages.append(message)
            used += estimate_message_tokens(message)
            handled += 1
            seconds = time.monotonic() - started
            run.steps.append(
                Step(index, call.name, call.arguments, len(output), seconds, repeated=repeated, failed=failed)
            )
            log(
                f"[agent] round {round_number} step {index}: {call.name}({_summary(call.arguments)}) "
                f"-> {len(output)} chars in {seconds:.1f}s{' (repeat)' if repeated else ''}"
                f"{' (error)' if failed else ''}"
            )
        dropped = len(result.tool_calls) - handled
        if dropped:
            # The calls without an answer are removed from the model's own message, so nothing is left unanswered.
            run.dropped_calls += dropped
            kept = {**result.message, "tool_calls": result.message["tool_calls"][:handled]}
            if handled:
                messages[answer_at] = kept
            elif kept.get("content"):
                messages[answer_at] = {key: value for key, value in kept.items() if key != "tool_calls"}
            else:
                del messages[answer_at]
            log(f"[agent] round {round_number}: {dropped} tool call(s) not run, their answers would not fit the window")
        if repeats >= MAX_REPEATS_IN_A_ROW:
            run.stop_reason = "stuck_repeating"
            break
        if out_of_room or used + tool_message_tokens + reserve > client.num_ctx:
            run.stop_reason = "context_budget"
            break
    log(f"[agent] exploration stopped: {run.stop_reason} after {run.rounds} round(s), {len(run.steps)} tool call(s)")
    messages.append({"role": "user", "content": FINAL_INSTRUCTION})
    run.final, run.final_result = client.chat_json_messages(messages, schema)
    run.peak_prompt_tokens = max(run.peak_prompt_tokens, run.final_result.prompt_tokens)
    return run

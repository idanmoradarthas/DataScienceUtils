"""Tests for the Ollama client, the read-only repository tools and the tool-calling loop of the release tooling.

The server is a fake that speaks the real Ollama HTTP API (see ``fake_ollama.py``), the repositories are real
throwaway git repositories. These tests check the protocol and the safety rules, not the quality of any model.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_RELEASE_DIR = _TESTS_DIR.parent / ".github" / "release"
sys.path.insert(0, str(_RELEASE_DIR))
sys.path.insert(0, str(_TESTS_DIR))
import agent  # noqa: E402
import fake_ollama as fo  # noqa: E402
import llm_client as lc  # noqa: E402
import release_repo as rr  # noqa: E402
import repo_tools as rt  # noqa: E402

GUIDE = "\n".join(f"guide line {number}" for number in range(1, 301)) + "\n"
STATS_V1 = '''"""Stats."""


def mean_abs(values):
    """Mean absolute value."""
    return 0


def old_helper(values):
    """Helper that will move."""
    return values


def legacy_fn(values):
    """Function that will disappear."""
    return values
'''
STATS_V2 = '''"""Stats."""


def mean_abs(values):
    """Mean absolute value."""
    return sum(abs(v) for v in values) / len(values)
'''
HELPERS = '''"""Helpers."""


def old_helper(values):
    """Helper that moved."""
    return values
'''
CURVES = '''"""Curves."""


def plot_zebra_curve(values, *, smooth=True):
    """Plot the zebra curve of values."""
    return values
'''


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=T",
            "-c",
            "user.email=t@example.com",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "tag.gpgsign=false",
            "-c",
            "core.autocrlf=false",
            *args,
        ],
        check=True,
        capture_output=True,
    )


def _commit(repo: Path, message: str, files: dict[str, str | bytes | None], *, rename: tuple[str, str] | None = None):
    if rename:
        (repo / rename[1]).parent.mkdir(parents=True, exist_ok=True)
        _git(repo, "mv", rename[0], rename[1])
    for relative, content in files.items():
        target = repo / relative
        if content is None:
            target.unlink()
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)


@pytest.fixture(scope="module")
def tool_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a repository with tag 1.0.0 and four pull-request commits (new function, fix, renamed file, move)."""
    repo = tmp_path_factory.mktemp("tools")
    _git(repo, "init", "-q", "-b", "main")
    _commit(
        repo,
        "Initial release",
        {
            "pkg/__init__.py": "",
            "pkg/stats.py": STATS_V1,
            "docs/guide.md": GUIDE,
            "assets/logo.bin": b"\x00\x01\x02binary",
        },
    )
    _git(repo, "tag", "1.0.0")
    _commit(
        repo,
        "Add curves (#21)\n\n* Plots the zebra curve.\n\nCo-Authored-By: X <x@example.com>\n",
        {"pkg/curves.py": CURVES},
    )
    _commit(repo, "Fix mean_abs (#22)\n\nIt returned 0 for every input.\n", {"pkg/stats.py": STATS_V2})
    _commit(
        repo,
        "Rename the guide (#23)",
        {"docs/manual.md": GUIDE.replace("line 7\n", "line seven\n")},
        rename=("docs/guide.md", "docs/manual.md"),
    )
    _commit(repo, "Move the helper (#24)", {"pkg/helpers.py": HELPERS})
    return repo


@pytest.fixture
def tools(tool_repo: Path) -> rt.RepoTools:
    """Tools bound to the range 1.0.0..HEAD of the fixture repository."""
    return rt.RepoTools(tool_repo, "1.0.0", "HEAD", "pkg")


def _plain_server(responder: fo.Responder, **kwargs) -> fo.FakeOllama:
    return fo.FakeOllama(responder, **kwargs)


# ---------------------------------------------------------------- client


def test_server_introspection_reports_capabilities_context_and_loaded_context():
    """show/ps give the declared capabilities, the trained context and the allocated context."""
    with _plain_server(lambda payload, server: fo.reply("ok")) as server:
        client = lc.OllamaClient("fake:1b", host=server.host, num_ctx=2048)
        assert client.version() == "0.0.0-fake"
        assert client.list_models() == ["fake:1b"]
        assert client.capabilities() == ["completion", "tools"]
        assert client.trained_context_length() == 131072
        assert client.loaded_context_length() is None
        client.chat([{"role": "user", "content": "hi"}])
        assert client.loaded_context_length() == 2048


def test_ensure_ready_rejects_a_missing_model():
    """A model that is not installed is reported with the names that are."""
    with _plain_server(lambda payload, server: fo.reply("ok")) as server:
        client = lc.OllamaClient("other:7b", host=server.host)
        with pytest.raises(lc.LLMError, match="not available"):
            client.ensure_ready()


def test_unreachable_server_raises_llm_error():
    """A closed port is an LLMError, not a raw socket exception."""
    with _plain_server(lambda payload, server: fo.reply("ok")) as server:
        host = server.host
    client = lc.OllamaClient("fake:1b", host=host, timeout=2)
    with pytest.raises(lc.LLMError, match="Cannot reach Ollama"):
        client.version()


def test_chat_sends_options_and_reads_the_servers_numbers():
    """Context size, tools and format go into the request; token counts and durations come back as numbers."""
    answer = fo.reply("hello", prompt_tokens=1000, output_tokens=50)
    with _plain_server(lambda payload, server: answer) as server:
        client = lc.OllamaClient("fake:1b", host=server.host, num_ctx=4096, num_predict=64)
        tool = {"type": "function", "function": {"name": "t", "description": "d", "parameters": {"type": "object"}}}
        result = client.chat([{"role": "user", "content": "hi"}], tools=[tool], schema={"type": "object"})
        payload = server.requests[0]
        assert payload["options"]["num_ctx"] == 4096
        assert payload["options"]["num_predict"] == 64
        assert payload["stream"] is False
        assert payload["keep_alive"] == "30m"
        assert payload["tools"] == [tool]
        assert payload["format"] == {"type": "object"}
        assert (result.content, result.prompt_tokens, result.output_tokens) == ("hello", 1000, 50)
        assert result.prompt_tokens_per_second == pytest.approx(1000.0)
        assert result.output_tokens_per_second == pytest.approx(100.0)
        assert result.load_seconds == pytest.approx(0.5)
        assert client.usage.calls == 1
        assert client.usage.prompt_tokens == 1000


def test_num_thread_is_sent_only_when_it_is_set():
    """A thread count goes into the options; 0 leaves the choice to Ollama and sends nothing."""
    with _plain_server(lambda payload, server: fo.reply("ok")) as server:
        lc.OllamaClient("fake:1b", host=server.host, num_thread=3).chat([{"role": "user", "content": "hi"}])
        lc.OllamaClient("fake:1b", host=server.host).chat([{"role": "user", "content": "hi"}])
        assert server.requests[0]["options"]["num_thread"] == 3
        assert "num_thread" not in server.requests[1]["options"]


def test_chat_options_override_the_defaults_for_one_call():
    """A single call can override num_ctx without touching the client defaults."""
    with _plain_server(lambda payload, server: fo.reply("ok")) as server:
        client = lc.OllamaClient("fake:1b", host=server.host, num_ctx=4096)
        client.chat([{"role": "user", "content": "hi"}], options={"num_ctx": 999})
        assert server.requests[0]["options"]["num_ctx"] == 999
        assert client.num_ctx == 4096


def test_tool_calls_accept_object_and_json_string_arguments():
    """Arguments may arrive as an object or as a JSON string; broken JSON is kept as an error, not an exception."""
    calls = [("a", {"x": 1}), ("b", '{"y": 2}'), ("c", "{not json"), ("d", "[1, 2]")]
    with _plain_server(lambda payload, server: fo.reply(tool_calls=calls)) as server:
        result = lc.OllamaClient("fake:1b", host=server.host).chat([{"role": "user", "content": "hi"}])
    by_name = {call.name: call for call in result.tool_calls}
    assert by_name["a"].arguments == {"x": 1}
    assert by_name["b"].arguments == {"y": 2}
    assert "not valid JSON" in by_name["c"].error
    assert "not an object" in by_name["d"].error
    assert result.message["role"] == "assistant"
    sent_back = {entry["function"]["name"]: entry["function"]["arguments"] for entry in result.message["tool_calls"]}
    assert sent_back == {"a": {"x": 1}, "b": {"y": 2}, "c": {}, "d": {}}


def test_http_errors_carry_status_and_body():
    """A server error becomes an LLMError that keeps the status and the server message."""

    def failing(payload, server):
        raise fo.HttpFailure(500, "model crashed")

    with _plain_server(failing) as server:
        with pytest.raises(lc.LLMError) as caught:
            lc.OllamaClient("fake:1b", host=server.host).chat([{"role": "user", "content": "hi"}])
    assert caught.value.status == 500
    assert "model crashed" in caught.value.body


def test_a_reply_that_is_not_a_json_object_is_an_llm_error():
    """Callers only expect LLMError from the client, so a list or a number is reported the same way."""
    with _plain_server(lambda payload, server: [1, 2]) as server:
        with pytest.raises(lc.LLMError, match="expected an object"):
            lc.OllamaClient("fake:1b", host=server.host).chat([{"role": "user", "content": "hi"}])


def test_an_error_body_that_cannot_be_read_still_gives_an_llm_error_with_the_status():
    """The connection breaks while the server's error text is read: the status is kept, the body is empty."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(5)

    def read_until(connection: socket.socket, done) -> bytes | None:
        """Read from the client until ``done(data)``; None when the client closed the connection first."""
        data = b""
        while not done(data):
            chunk = connection.recv(4096)
            if not chunk:
                return None
            data += chunk
        return data

    def serve() -> None:
        connection, _ = listener.accept()
        with connection:
            connection.settimeout(5)
            request = read_until(connection, lambda data: b"\r\n\r\n" in data)
            if request is None:
                return
            head, _, body = request.partition(b"\r\n\r\n")
            length = int(re.search(rb"content-length:\s*(\d+)", head, re.IGNORECASE).group(1))
            if read_until(connection, lambda data: len(body + data) >= length) is None:
                return
            connection.sendall(
                b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 100\r\nConnection: close\r\n\r\ncut"
            )

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    try:
        client = lc.OllamaClient("fake:1b", host=f"127.0.0.1:{listener.getsockname()[1]}", timeout=5)
        with pytest.raises(lc.LLMError) as caught:
            client.chat([{"role": "user", "content": "hi"}])
    finally:
        listener.close()
        worker.join(timeout=5)
    assert (caught.value.status, caught.value.body) == (500, "")


def test_a_fake_server_that_was_never_started_can_be_stopped():
    """``shutdown`` waits for a serving loop; stopping a server that never served must not wait forever."""
    server = fo.FakeOllama(lambda payload, server: fo.reply("ok"))
    worker = threading.Thread(target=server.stop, daemon=True)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive()
    server.stop()  # a second stop is harmless


def test_the_think_flag_is_dropped_when_the_server_rejects_it():
    """Models without thinking answer 400 to ``think``; the client repeats the request without the flag."""

    def responder(payload, server):
        if "think" in payload:
            raise fo.HttpFailure(400, '"fake:1b" does not support thinking (think)')
        return fo.reply("ok")

    with _plain_server(responder) as server:
        client = lc.OllamaClient("fake:1b", host=server.host, think=True)
        assert client.chat([{"role": "user", "content": "hi"}]).content == "ok"
        assert client.think is None
        assert ["think" in payload for payload in server.requests] == [True, False]


def test_chat_json_retries_once_with_the_problems_and_counts_it():
    """An invalid answer is sent back with the list of problems; the second answer is accepted."""
    answers = iter([fo.reply("not json"), fo.reply('{"n": 3}')])
    with _plain_server(lambda payload, server: next(answers)) as server:
        client = lc.OllamaClient("fake:1b", host=server.host)
        schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
        assert client.chat_json("system", "user", schema) == {"n": 3}
        retry = server.requests[1]["messages"]
        assert retry[-2] == {"role": "assistant", "content": "not json"}
        assert "rejected" in retry[-1]["content"]
        assert (client.usage.calls, client.usage.retries, client.usage.failures) == (2, 1, 0)


def test_chat_json_fails_after_the_attempts_and_hints_at_cut_off_output():
    """Two invalid answers raise LLMError; an answer cut at the token limit says so in the retry."""
    answers = iter([fo.reply('{"n": ', done_reason="length"), fo.reply("[]")])
    with _plain_server(lambda payload, server: next(answers)) as server:
        client = lc.OllamaClient("fake:1b", host=server.host)
        schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
        with pytest.raises(lc.LLMError, match="no valid JSON"):
            client.chat_json("system", "user", schema)
        assert "cut off" in server.requests[1]["messages"][-1]["content"]
        assert client.usage.failures == 1


ANSWER_SCHEMA = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}


def _retry_after_a_bad_answer(prompt_tokens: int, output_tokens: int):
    """Ask for JSON in a 8192-token window with room for 1024 answer tokens; the first answer is never valid."""
    with _plain_server(
        lambda payload, server: fo.reply("oops", prompt_tokens=prompt_tokens, output_tokens=output_tokens)
    ) as server:
        client = lc.OllamaClient("fake:1b", host=server.host, num_ctx=8192, num_predict=1024)
        with pytest.raises(lc.LLMError) as caught:
            client.chat_json_messages([{"role": "user", "content": "question"}], ANSWER_SCHEMA)
    return server.requests, str(caught.value)


def test_the_retry_repeats_the_rejected_answer_only_when_it_fits_the_window():
    """Ollama drops the start of a prompt that is too long, so the corrective retry is sized before it is sent."""
    requests, _ = _retry_after_a_bad_answer(prompt_tokens=2000, output_tokens=800)
    assert [message["role"] for message in requests[1]["messages"]] == ["user", "assistant", "user"]
    requests, _ = _retry_after_a_bad_answer(prompt_tokens=6500, output_tokens=800)  # the answer would not fit
    assert [message["role"] for message in requests[1]["messages"]] == ["user", "user"]
    assert requests[1]["messages"][-1]["content"].startswith("Your previous answer was rejected")


def test_no_retry_is_sent_when_not_even_the_correction_fits_the_window():
    """A prompt that is already at the limit cannot take another message: stop with a clear reason."""
    requests, message = _retry_after_a_bad_answer(prompt_tokens=7150, output_tokens=800)
    assert len(requests) == 1
    assert "after 1 attempt(s)" in message
    assert "no room left in the context window" in message


def test_a_third_attempt_is_sized_from_the_original_conversation_not_from_the_second_request():
    """Every corrective request is rebuilt from the original messages, so the first correction does not count."""

    def responder(payload, server):
        reported = 6000 if len(payload["messages"]) == 1 else 7150  # the longer request reads 1150 tokens more
        return fo.reply("oops", prompt_tokens=reported, output_tokens=800)

    with _plain_server(responder) as server:
        client = lc.OllamaClient("fake:1b", host=server.host, num_ctx=8192, num_predict=1024)
        with pytest.raises(lc.LLMError) as caught:
            client.chat_json_messages([{"role": "user", "content": "question"}], ANSWER_SCHEMA, attempts=3)
    assert [len(request["messages"]) for request in server.requests] == [1, 3, 3]
    assert "after 3 attempt(s)" in str(caught.value)
    assert "no room left" not in str(caught.value)


@pytest.mark.parametrize(
    "tool_calls",
    [[None], ["search"], [{"function": "search"}], "search", {"function": {}}, 5],
    ids=lambda value: str(value),
)
def test_a_tool_call_that_is_not_an_object_is_an_llm_error(tool_calls):
    """The generator turns an LLMError into its model-error exit; an AttributeError would be a traceback."""
    reply = {"message": {"role": "assistant", "tool_calls": tool_calls}, "done": True}
    with _plain_server(lambda payload, server: reply) as server:
        client = lc.OllamaClient("fake:1b", host=server.host)
        with pytest.raises(lc.LLMError, match="tool"):
            client.chat([{"role": "user", "content": "hi"}])


def test_a_reply_message_that_is_not_an_object_is_an_llm_error():
    """``message`` must be an object like the rest of the reply."""
    with _plain_server(lambda payload, server: {"message": "text", "done": True}) as server:
        client = lc.OllamaClient("fake:1b", host=server.host)
        with pytest.raises(lc.LLMError, match="not an object"):
            client.chat([{"role": "user", "content": "hi"}])


def test_json_inside_a_code_fence_is_accepted():
    """Some models wrap the JSON in a Markdown fence."""
    assert lc.parse_json('```json\n{"a": 1}\n```') == {"a": 1}


@pytest.mark.parametrize(
    ("schema", "value", "fragment"),
    [
        ({"type": "object", "required": ["a"]}, {}, "missing required field 'a'"),
        ({"type": "array", "minItems": 2}, [1], "at least 2"),
        ({"type": "array", "maxItems": 1}, [1, 2], "at most 1"),
        ({"type": "string", "minLength": 3}, "ab", "shorter than 3"),
        ({"type": "string", "enum": ["x"]}, "y", "must be one of"),
        ({"type": "integer"}, True, "expected an integer"),
        ({"type": "object", "properties": {"a": {"type": "array", "items": {"type": "string"}}}}, {"a": [1]}, "$.a[0]"),
    ],
)
def test_validate_schema_reports_the_problem_with_its_path(schema, value, fragment):
    """The schema subset reports readable errors with the location."""
    assert any(fragment in error for error in lc.validate_schema(schema, value))
    assert lc.validate_schema({"type": "string"}, "fine") == []


# ---------------------------------------------------------------- repo tools


def test_tool_definitions_match_the_handlers(tools: rt.RepoTools):
    """Every advertised tool has a handler and every required argument is a declared property."""
    names = [definition["function"]["name"] for definition in tools.schemas()]
    assert len(names) == len(set(names))
    assert set(names) == set(tools._handlers)  # noqa: SLF001 - the handler table is the contract under test
    for definition in tools.schemas():
        parameters = definition["function"]["parameters"]
        assert set(parameters["required"]) <= set(parameters["properties"])
        assert definition["type"] == "function"


def test_changed_files_lists_status_lines_and_renames(tools: rt.RepoTools):
    """Added, modified, deleted and renamed files are listed with their line counts."""
    text = tools.call("changed_files", {})
    assert "A  +" in text
    assert "A  +6 -0  pkg/curves.py" in text
    assert "R  +1 -1  docs/manual.md  (from docs/guide.md)" in text
    assert "M  " in text
    only_pkg = tools.call("changed_files", {"path_prefix": "pkg"})
    assert "docs/" not in only_pkg
    assert "pkg/stats.py" in only_pkg


def test_diff_file_shows_the_change_and_hides_index_noise(tools: rt.RepoTools):
    """The diff of one file has the changed lines and no ``index`` or ``---`` header lines."""
    text = tools.call("diff_file", {"path": "pkg/stats.py"})
    assert "-    return 0" in text
    assert "+    return sum(abs(v) for v in values) / len(values)" in text
    assert "index " not in text
    assert "no differences" in tools.call("diff_file", {"path": "pkg/__init__.py"})


def test_read_file_pages_numbers_lines_and_reads_both_versions(tools: rt.RepoTools):
    """Long files are paged with a footer that says how to continue; base and head can differ."""
    first = tools.call("read_file", {"path": "docs/guide.md", "version": "base"})
    assert first.splitlines()[0].strip() == "1  guide line 1"
    assert "[lines 1-120 of 300; call again with start_line=121 for more]" in first
    second = tools.call("read_file", {"path": "docs/guide.md", "version": "base", "start_line": "121"})
    assert second.splitlines()[0].strip() == "121  guide line 121"
    old = tools.call("read_file", {"path": "pkg/stats.py", "version": "base"})
    new = tools.call("read_file", {"path": "pkg/stats.py", "version": "head"})
    assert "return 0" in old
    assert "return sum" in new
    assert "does not exist at head" in tools.call("read_file", {"path": "docs/guide.md"})
    assert "(no more lines" in tools.call("read_file", {"path": "pkg/stats.py", "start_line": 999})


def test_read_file_refuses_binary_files_and_bad_versions(tools: rt.RepoTools):
    """Binary files are not dumped into the context, and ``version`` must be base or head."""
    assert "is binary" in tools.call("read_file", {"path": "assets/logo.bin", "version": "base"})
    assert tools.call("read_file", {"path": "pkg/stats.py", "version": "main"}).startswith("error: version must be")
    assert tools.call("read_file", {"path": "pkg/stats.py", "start_line": "abc"}).startswith("error: start_line")


def test_search_is_plain_text_case_insensitive_and_scoped(tools: rt.RepoTools):
    """Search treats the text literally (no regular expression), ignores case and can be limited to a folder."""
    found = tools.call("search", {"text": "MEAN_ABS", "path_prefix": "pkg"})
    assert "pkg/stats.py" in found
    assert "matches at head" in found
    assert "(no match" in tools.call("search", {"text": "mean.abs"})
    assert "(no match" in tools.call("search", {"text": "zebra", "version": "base"})
    assert "pkg/curves.py" in tools.call("search", {"text": "zebra"})
    assert "(no match for '-x'" in tools.call("search", {"text": "-x"})
    assert tools.call("search", {"text": "a\nb"}).startswith("error: text must be")


def test_commit_log_and_show_pull_request(tools: rt.RepoTools):
    """Pull requests are listed newest first; one can be shown with a cleaned message and its files."""
    log = tools.call("commit_log", {})
    assert log.splitlines()[0] == "4 pull requests"
    assert [line.split("  ", 1)[1] for line in log.splitlines()[1:]] == [
        "Move the helper (#24)",
        "Rename the guide (#23)",
        "Fix mean_abs (#22)",
        "Add curves (#21)",
    ]
    assert "1 pull requests" in tools.call("commit_log", {"path_prefix": "pkg/curves.py"})
    shown = tools.call("show_pull_request", {"number": "#21"})
    assert shown.startswith("Add curves (#21)")
    assert "Plots the zebra curve." in shown
    assert "Co-Authored-By" not in shown
    assert "A  pkg/curves.py" in shown
    sha = log.splitlines()[-1].split("  ")[0]
    assert tools.call("show_pull_request", {"number": sha}).startswith("Add curves (#21)")
    assert "no pull request '999'" in tools.call("show_pull_request", {"number": "999"})
    assert tools.call("show_pull_request", {"number": "21; rm -rf"}).startswith("error: number must be")


def test_find_symbol_tells_new_moved_existing_and_removed_apart(tools: rt.RepoTools):
    """The status of a public symbol comes from the API diff, not from the wording of a commit."""
    assert "NEW in this release" in tools.call("find_symbol", {"name": "plot_zebra_curve"})
    assert "already existed in the previous release, same module" in tools.call("find_symbol", {"name": "mean_abs"})
    assert "moved from pkg.stats.old_helper" in tools.call("find_symbol", {"name": "old_helper"})
    assert "not public any more" in tools.call("find_symbol", {"name": "legacy_fn"})
    assert "no public function or class named 'nope'" in tools.call("find_symbol", {"name": "nope"})
    assert "plot_zebra_curve" in tools.call("find_symbol", {"name": "zebra"})
    assert "NEW" in tools.call("find_symbol", {"name": "`pkg.curves.plot_zebra_curve()`"})
    assert tools.call("find_symbol", {"name": "a b"}).startswith("error: name must be")


@pytest.mark.parametrize(
    "path", ["../secret", "/etc/passwd", "-rf", ":(glob)*", "C:\\Windows", "a/../b", "", None, 5, "a\nb"]
)
def test_unsafe_paths_are_rejected_by_every_path_tool(tools: rt.RepoTools, path):
    """Absolute paths, parent segments, options, pathspec magic and drive letters never reach git."""
    for name in ("diff_file", "read_file"):
        assert tools.call(name, {"path": path}).startswith("error:")
    if path:
        assert tools.call("changed_files", {"path_prefix": path}).startswith("error:")


def test_backslash_paths_are_normalized_and_glob_characters_stay_literal(tools: rt.RepoTools):
    """Windows-style separators work; a glob is a literal name, so it matches nothing."""
    assert "return sum" in tools.call("read_file", {"path": "pkg\\stats.py"})
    assert "(no differences" in tools.call("diff_file", {"path": "pkg/*.py"})
    assert "(no files differ)" in tools.call("changed_files", {"path_prefix": "pkg/*.py"})


def test_unknown_tools_and_missing_arguments_return_text_not_exceptions(tools: rt.RepoTools):
    """A model that calls a tool wrongly gets an error message it can act on."""
    assert "unknown tool 'rm'" in tools.call("rm", {})
    assert tools.call("read_file", {}).startswith("error: path must be")
    assert tools.call("changed_files", None).startswith(("1", "2", "3", "4", "5", "6", "7", "8", "9"))


def test_output_is_capped_and_observed_text_is_kept(tool_repo: Path):
    """No tool answer is longer than the cap, and the verification step can see what the model saw."""
    small = rt.RepoTools(tool_repo, "1.0.0", "HEAD", "pkg", max_output_chars=120)
    text = small.call("changed_files", {})
    assert "[output cut at 120 characters" in text
    assert len(text) <= 120
    assert small.calls == 1
    assert small.observed == [text]
    assert small.output_chars == len(text)


def test_the_cap_holds_for_text_without_line_breaks_and_for_error_messages(tool_repo: Path):
    """Error texts repeat what the model sent, so a long argument must not make a long answer."""
    tools = rt.RepoTools(tool_repo, "1.0.0", "HEAD", "pkg", max_output_chars=300)
    replies = [
        tools.call("read_file", {"path": "../" + "a" * 5000}),
        tools.call("x" * 5000, {}),
        tools.call("search", {"text": "z" * 5000}),
    ]
    assert [len(reply) <= 300 for reply in replies] == [True, True, True]
    assert replies[0].startswith("error: '../aaa")
    assert replies[1].startswith("error: unknown tool 'xxx")
    assert tools.output_chars == sum(len(reply) for reply in replies)
    assert tools.observed == replies
    assert rt._cap("x" * 1000, 120).endswith("a smaller part]")
    assert len(rt._cap("x" * 1000, 120)) <= 120


def test_a_page_ends_where_the_limit_ends_it_and_says_where_to_continue(tool_repo: Path):
    """The footer names the lines really shown, so paging never skips a line (it used to be cut off with the text)."""
    small = rt.RepoTools(tool_repo, "1.0.0", "HEAD", "pkg", max_output_chars=600)
    first = small.call("read_file", {"path": "docs/manual.md"}).splitlines()
    last = int(first[-2].split()[0])
    assert 1 < last < 120
    assert first[-1] == f"[lines 1-{last} of 300; call again with start_line={last + 1} for more]"
    assert "output cut" not in "\n".join(first)
    second = small.call("read_file", {"path": "docs/manual.md", "start_line": last + 1}).splitlines()
    assert int(second[0].split()[0]) == last + 1
    assert all(len(reply) <= 600 for reply in small.observed)


@pytest.fixture
def edge_repo(tmp_path: Path) -> Path:
    """Create a repository with a removed name inside a longer one, and a file whose first line is huge."""
    repo = tmp_path / "edge"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _commit(repo, "Initial release", {"pkg/__init__.py": "", "pkg/plots.py": "def plot(values):\n    return values\n"})
    _git(repo, "tag", "1.0.0")
    _commit(
        repo,
        "Rename plot (#5)",
        {"pkg/plots.py": "def plot_more(values):\n    return values\n", "docs/long.txt": "x" * 5000 + "\nsecond\n"},
    )
    return repo


def test_one_huge_line_is_clipped_and_the_page_still_continues(edge_repo: Path):
    """Minified or generated files have lines longer than the whole limit."""
    small = rt.RepoTools(edge_repo, "1.0.0", "HEAD", "pkg", max_output_chars=600)
    page = small.call("read_file", {"path": "docs/long.txt"}).splitlines()
    assert len(page) == 2
    assert page[0].endswith("...")
    assert page[1] == "[lines 1-1 of 2; call again with start_line=2 for more]"
    assert small.call("read_file", {"path": "docs/long.txt", "start_line": 2}).splitlines()[0].endswith("second")


def test_a_page_keeps_the_room_its_footer_needs_even_when_the_limit_is_tiny(edge_repo: Path):
    """The footer says where to continue; the body gets what is left of the limit, not a fixed share."""
    tiny = rt.RepoTools(edge_repo, "1.0.0", "HEAD", "pkg", max_output_chars=100)
    page = tiny.call("read_file", {"path": "docs/long.txt"})
    assert len(page) <= 100
    assert page.splitlines()[-1] == "[lines 1-1 of 2; call again with start_line=2 for more]"
    assert "second" in tiny.call("read_file", {"path": "docs/long.txt", "start_line": 2})


def test_a_name_that_was_removed_is_reported_even_when_a_longer_name_contains_it(edge_repo: Path):
    """``plot`` is gone and ``plot_more`` is new: the answer must not present the partial match as the symbol."""
    tools = rt.RepoTools(edge_repo, "1.0.0", "HEAD", "pkg")
    answer = tools.call("find_symbol", {"name": "plot"})
    assert "'plot' is not public any more; the previous release had pkg.plots.plot" in answer
    assert "similar names at head: pkg.plots.plot_more" in answer
    assert "NEW" not in answer
    assert "NEW in this release" in tools.call("find_symbol", {"name": "plot_more"})
    assert "plot_more" in tools.call("find_symbol", {"name": "more"})  # still a partial match when nothing was removed


def test_a_name_whose_old_import_still_works_is_not_reported_as_removed(tmp_path: Path):
    """``plot`` moved to a private module and is re-exported, so ``from pkg.plots import plot`` keeps working."""
    repo = tmp_path / "reexport"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    body = "def {}(values):\n    return values\n"
    _commit(
        repo,
        "Initial release",
        {"pkg/__init__.py": "", "pkg/plots.py": body.format("plot"), "pkg/lazy.py": body.format("draw")},
    )
    _git(repo, "tag", "1.0.0")
    _commit(
        repo,
        "Move the implementations (#2)",
        {
            "pkg/_impl.py": body.format("plot"),
            "pkg/plots.py": "from ._impl import plot\n",
            "pkg/lazy.py": "def __getattr__(name):\n    raise AttributeError(name)\n",
        },
    )
    tools = rt.RepoTools(repo, "1.0.0", "HEAD", "pkg")
    answer = tools.call("find_symbol", {"name": "plot"})
    assert "'from pkg.plots import plot' still works at head, so users are not affected" in answer
    assert "not public any more" not in answer
    dynamic = tools.call("find_symbol", {"name": "draw"})
    assert "cannot tell whether 'from pkg.lazy import draw' still works" in dynamic
    assert "not public any more" not in dynamic


@pytest.mark.skipif(sys.platform == "win32", reason="the file name is written with git plumbing and needs a quote sign")
def test_files_with_names_that_git_quotes_can_be_read(edge_repo: Path):
    """``git ls-tree`` prints a name with a quote sign in quotes unless it is asked for NUL-separated names."""
    blob = subprocess.run(
        ["git", "-C", str(edge_repo), "hash-object", "-w", "--stdin"], input=b"hello\n", capture_output=True, check=True
    ).stdout.decode()
    name = 'docs/say "hi".txt'
    _git(edge_repo, "update-index", "--add", "--cacheinfo", f"100644,{blob.strip()},{name}")
    _git(edge_repo, "commit", "-q", "-m", "A name with quotes (#6)")
    tools = rt.RepoTools(edge_repo, "1.0.0", "HEAD", "pkg")
    assert "hello" in tools.call("read_file", {"path": name})


def test_a_pull_request_is_found_by_number_full_hash_or_a_prefix_of_it(tool_repo: Path, tools: rt.RepoTools):
    """The log shows 8 characters of a hash; the full hash and any longer prefix work too, in any letter case."""
    full = subprocess.run(
        ["git", "-C", str(tool_repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert f"{full[:8]}  Move the helper (#24)" in tools.call("commit_log", {})
    for value in ("#24", "24", full, full[:8], full[:8].upper(), full[:12]):
        assert tools.call("show_pull_request", {"number": value}).startswith("Move the helper (#24)"), value


def test_digits_can_be_a_hash_prefix_a_pull_request_number_or_both(tmp_path: Path):
    """A hash can start with four digits: ``1234`` is then both; ``#1234`` is always the pull request."""
    repo = tmp_path / "digits"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _commit(repo, "Initial release", {"pkg/__init__.py": ""})
    _git(repo, "tag", "1.0.0")
    for number in range(1, 300):  # about one hash in seven starts with four decimal digits
        _commit(repo, f"Change {number} (#{number})", {f"pkg/m{number}.py": f"value = {number}\n"})
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        if sha[:4].isdigit():
            break
    digits = sha[:4]
    assert digits.isdigit()
    tools = rt.RepoTools(repo, "1.0.0", "HEAD", "pkg")
    assert tools.call("show_pull_request", {"number": digits}).startswith(f"Change {number} (#{number})")
    assert "no pull request" in tools.call("show_pull_request", {"number": f"#{digits}"})
    _commit(repo, f"Another change (#{digits})", {"pkg/other.py": "value = 0\n"})
    tools = rt.RepoTools(repo, "1.0.0", "HEAD", "pkg")
    both = tools.call("show_pull_request", {"number": digits})
    assert "fits more than one pull request" in both
    assert f"{sha[:8]}  Change {number} (#{number})" in both
    assert tools.call("show_pull_request", {"number": f"#{digits}"}).startswith("Another change")
    assert tools.call("show_pull_request", {"number": sha[:8]}).startswith(f"Change {number} (#{number})")


def test_refs_are_resolved_once_to_commit_hashes(tool_repo: Path):
    """The model never supplies a ref: the tools hold two commit hashes."""
    tools = rt.RepoTools(tool_repo, "1.0.0", "HEAD", "pkg")
    assert all(len(sha) == 40 for sha in tools.shas.values())
    with pytest.raises(RuntimeError, match="rev-parse"):
        rt.RepoTools(tool_repo, "no-such-tag", "HEAD", "pkg")


# ---------------------------------------------------------------- agent loop

SCHEMA = {
    "type": "object",
    "properties": {"function": {"type": "string"}, "first_parameter": {"type": "string"}},
    "required": ["function", "first_parameter"],
}


@pytest.fixture
def release_repo(tmp_path: Path) -> Path:
    """Create the synthetic repository: tag 1.0.0, then ``plot_zebra_curve`` is added."""
    rr.make_release_repo(tmp_path / "repo")
    return tmp_path / "repo"


def _run(
    model: fo.ScriptedModel, repo: Path, *, max_rounds: int = 6, num_ctx: int = 8192, system: str = "system", **kwargs
):
    logs: list[str] = []
    with fo.FakeOllama(model) as server:
        client = lc.OllamaClient("fake:1b", host=server.host, num_ctx=num_ctx)
        result = agent.run_agent(
            client,
            rt.RepoTools(repo, "1.0.0", "HEAD", "pkg"),
            system,
            "user question",
            SCHEMA,
            max_rounds=max_rounds,
            log=logs.append,
            **kwargs,
        )
    return result, server.requests, logs


def test_agent_explores_with_tools_then_answers_with_a_schema(release_repo: Path):
    """Exploration requests carry tools and no format; the final request carries the format and no tools."""
    model = fo.ScriptedModel(tool_script=[("changed_files", {}), ("find_symbol", {"name": "plot_zebra_curve"})])
    run, requests, logs = _run(model, release_repo)
    assert run.final == {"function": "plot_zebra_curve", "first_parameter": "values"}
    assert [step.name for step in run.steps] == ["changed_files", "find_symbol"]
    assert run.stop_reason == "explored"
    assert run.rounds == 3
    exploration, final = requests[:-1], requests[-1]
    assert all("tools" in request and "format" not in request for request in exploration)
    assert "format" in final
    assert "tools" not in final
    assert final["messages"][-1]["content"] == agent.FINAL_INSTRUCTION
    history = final["messages"]
    assert [message["role"] for message in history[:6]] == ["system", "user", "assistant", "tool", "assistant", "tool"]
    assert history[3]["tool_name"] == "changed_files"
    assert history[2]["tool_calls"]
    assert any("find_symbol(" in line for line in logs)
    assert logs[-1].startswith("[agent] exploration stopped: explored")


def test_agent_stops_when_the_model_repeats_the_same_call(release_repo: Path):
    """The same call over and over is answered with a notice and ends the exploration."""
    model = fo.ScriptedModel(tool_script=[("changed_files", {})] * 10)
    run, requests, _ = _run(model, release_repo, max_rounds=10)
    assert run.stop_reason == "stuck_repeating"
    assert run.rounds == 4
    assert [step.repeated for step in run.steps] == [False, True, True, True]
    tool_messages = [message for message in requests[-1]["messages"] if message["role"] == "tool"]
    assert tool_messages[1]["content"] == agent.REPEAT_NOTICE
    assert run.final["function"] == "?"


def test_agent_respects_the_round_limit(release_repo: Path):
    """Different calls forever end at ``max_rounds`` and still produce the final answer."""
    model = fo.ScriptedModel(tool_script=[("changed_files", {"path_prefix": f"pkg/{index}"}) for index in range(20)])
    run, _, _ = _run(model, release_repo, max_rounds=3)
    assert (run.stop_reason, run.rounds, len(run.steps)) == ("max_rounds", 3, 3)
    assert run.final is not None


def test_agent_stops_exploring_before_the_context_window_is_full(release_repo: Path):
    """When the prompt is already near ``num_ctx``, the loop stops instead of letting Ollama truncate it."""
    model = fo.ScriptedModel(
        tool_script=[("changed_files", {}), ("find_symbol", {"name": "x"})], prompt_tokens_override=7000
    )
    run, requests, _ = _run(model, release_repo, num_ctx=8192)
    assert run.stop_reason == "context_budget"
    assert run.rounds == 1
    assert run.peak_prompt_tokens == 7000
    assert "tools" not in requests[-1]


def test_agent_reports_tool_calls_with_broken_arguments_to_the_model(release_repo: Path):
    """Arguments that are not valid JSON become an error message in the tool result."""

    def responder(payload, server):
        if payload.get("format"):
            return fo.reply(json.dumps({"function": "f", "first_parameter": "p"}))
        if not any(message["role"] == "tool" for message in payload["messages"]):
            return fo.reply(tool_calls=[("changed_files", "{not json")])
        return fo.reply("done")

    with fo.FakeOllama(responder) as server:
        client = lc.OllamaClient("fake:1b", host=server.host)
        run = agent.run_agent(
            client, rt.RepoTools(release_repo, "1.0.0", "HEAD", "pkg"), "s", "u", SCHEMA, log=lambda _m: None
        )
    tool_message = next(message for message in server.requests[-1]["messages"] if message["role"] == "tool")
    assert tool_message["content"].startswith("error: arguments are not valid JSON")
    assert run.steps[0].failed


def test_a_model_without_tool_support_raises_the_servers_error(release_repo: Path):
    """The 400 error of a model without tools is recognizable, so the caller can fall back to the pack."""
    with fo.FakeOllama(fo.ScriptedModel(supports_tools=False)) as server:
        client = lc.OllamaClient("fake:1b", host=server.host)
        with pytest.raises(lc.LLMError) as caught:
            agent.run_agent(
                client, rt.RepoTools(release_repo, "1.0.0", "HEAD", "pkg"), "s", "u", SCHEMA, log=lambda _m: None
            )
    assert agent.tools_unsupported(caught.value)
    assert not agent.tools_unsupported(lc.LLMError("boom", status=500, body="tool failure"))
    assert not agent.tools_unsupported(lc.LLMError("boom", status=400, body="bad json"))


def test_tool_calls_go_back_to_the_server_with_arguments_as_an_object(release_repo: Path):
    """A model may send the arguments as JSON text; the history sent to the server always holds an object."""

    def responder(payload, server):
        if payload.get("format"):
            return fo.reply(json.dumps({"function": "f", "first_parameter": "p"}))
        if not any(message["role"] == "tool" for message in payload["messages"]):
            return fo.reply(tool_calls=[("changed_files", '{"path_prefix": "pkg"}')])
        return fo.reply("done")

    with fo.FakeOllama(responder) as server:
        client = lc.OllamaClient("fake:1b", host=server.host)
        agent.run_agent(
            client, rt.RepoTools(release_repo, "1.0.0", "HEAD", "pkg"), "s", "u", SCHEMA, log=lambda _m: None
        )
    assistant = next(message for message in server.requests[-1]["messages"] if message.get("tool_calls"))
    assert assistant["tool_calls"][0]["function"]["arguments"] == {"path_prefix": "pkg"}


def test_agent_refuses_a_prompt_that_cannot_fit_the_window_with_the_answer(release_repo: Path):
    """Ollama would drop the start of the prompt (the instructions): no request is sent at all."""
    with fo.FakeOllama(fo.ScriptedModel()) as server:
        client = lc.OllamaClient("fake:1b", host=server.host, num_ctx=4096)
        with pytest.raises(lc.LLMError, match="do not fit in the context window of 4096"):
            agent.run_agent(
                client,
                rt.RepoTools(release_repo, "1.0.0", "HEAD", "pkg"),
                "s" * 20000,
                "u",
                SCHEMA,
                log=lambda _m: None,
            )
    assert server.requests == []


def test_agent_skips_the_exploration_when_only_the_answer_fits_the_window(release_repo: Path):
    """The first prompt leaves room for the answer but not for one tool result: go straight to the answer."""
    run, requests, _ = _run(
        fo.ScriptedModel(tool_script=[("changed_files", {})]), release_repo, num_ctx=8192, system="s" * 16000
    )
    assert (run.stop_reason, run.rounds, run.steps) == ("context_budget", 0, [])
    assert len(requests) == 1
    assert "format" in requests[0]
    assert "tools" not in requests[0]


def _reserve(client: lc.OllamaClient) -> int:
    """Return the tokens the final request needs after the conversation, as the loop counts them."""
    return client.options["num_predict"] + lc.estimate_tokens(agent.FINAL_INSTRUCTION) + agent.ANSWER_SLACK_TOKENS


def _scripted(first: dict):
    """Build a responder: ``first`` answers every exploration request, valid JSON answers the final one."""

    def responder(payload, server):
        if payload.get("format"):
            return fo.reply(json.dumps({"function": "f", "first_parameter": "p"}))
        return first

    return responder


def _prompt_leaving(client: lc.OllamaClient, room: int, output_tokens: int = 20) -> int:
    """Return the prompt size to report so that ``room`` tokens are left in the window after the first answer."""
    return client.num_ctx - _reserve(client) - room - output_tokens


def _agent_run(tool_repo: Path, responder, **client_options):
    """Run the loop against ``responder`` with default tools; return the run, the requests and the client."""
    with fo.FakeOllama(responder) as server:
        client = lc.OllamaClient("fake:1b", host=server.host, **client_options)
        tools = rt.RepoTools(tool_repo, "1.0.0", "HEAD", "pkg")
        run = agent.run_agent(client, tools, "s", "u", SCHEMA, log=lambda _m: None)
    return run, server.requests, client, tools


def _messages_after_exploration(requests: list[dict]) -> tuple[dict, list[dict]]:
    """Return the assistant message with the tool calls and the tool messages of the final request."""
    history = requests[-1]["messages"]
    assistant = next((message for message in history if message.get("tool_calls")), {})
    return assistant, [message for message in history if message["role"] == "tool"]


BATCHES = {
    "distinct calls": [("read_file", {"path": "docs/manual.md", "start_line": number}) for number in range(1, 101)],
    "calls with broken arguments": [(f"tool_{number}", "{not json") for number in range(100)],
    "the same call again and again": [("changed_files", {})] * 100,
}


@pytest.mark.parametrize("batch", BATCHES.values(), ids=BATCHES.keys())
def test_a_large_batch_never_pushes_the_final_request_past_the_window(tool_repo: Path, batch: list):
    """Results, errors and repeat notices are all admitted against the window; the rest of the batch is removed."""
    client_probe = lc.OllamaClient("fake:1b", num_ctx=8192)
    tool_tokens = lc.estimate_tokens("x" * 6000) + lc.MESSAGE_OVERHEAD_TOKENS
    prompt_tokens = _prompt_leaving(client_probe, room=tool_tokens + 60)
    first = fo.reply(tool_calls=batch, prompt_tokens=prompt_tokens, output_tokens=20)
    run, requests, client, _ = _agent_run(tool_repo, _scripted(first), num_ctx=8192)
    assistant, tool_messages = _messages_after_exploration(requests)
    assert run.stop_reason in {"context_budget", "stuck_repeating"}  # the repeats of the last batch may end it first
    assert 0 < len(tool_messages) < len(batch)
    assert run.dropped_calls == len(batch) - len(tool_messages) == len(batch) - len(run.steps)
    assert len(assistant["tool_calls"]) == len(tool_messages)  # no call is left without its answer
    tokens = prompt_tokens + 20 + sum(lc.estimate_message_tokens(message) for message in tool_messages)
    assert tokens + _reserve(client) <= client.num_ctx
    assert requests[-1]["messages"][-1]["content"] == agent.FINAL_INSTRUCTION


def test_a_batch_that_has_no_room_for_a_single_answer_leaves_no_trace_in_the_history(tool_repo: Path):
    """The model's message with its calls is removed entirely when none of them can be answered."""
    client_probe = lc.OllamaClient("fake:1b", num_ctx=8192)
    first = fo.reply(
        tool_calls=[("changed_files", {}), ("commit_log", {})],
        prompt_tokens=_prompt_leaving(client_probe, room=10),
        output_tokens=20,
    )
    run, requests, _, _ = _agent_run(tool_repo, _scripted(first), num_ctx=8192)
    assert (run.stop_reason, run.steps, run.dropped_calls) == ("context_budget", [], 2)
    assert [message["role"] for message in requests[-1]["messages"]] == ["system", "user", "user"]


def test_a_long_exploration_reply_without_tool_calls_is_not_sent_back_in_the_final_request(tool_repo: Path):
    """Free text that does not fit next to the final answer stays in the report only; a short one stays in the chat."""
    client_probe = lc.OllamaClient("fake:1b", num_ctx=8192)
    long_text = "the model thinks aloud " * 100
    too_long = fo.reply(long_text, prompt_tokens=_prompt_leaving(client_probe, room=100), output_tokens=500)
    run, requests, _, _ = _agent_run(tool_repo, _scripted(too_long), num_ctx=8192)
    assert run.stop_reason == "explored"
    assert run.exploration_note == long_text
    assert [message["role"] for message in requests[-1]["messages"]] == ["system", "user", "user"]
    short = fo.reply("I have seen enough.", prompt_tokens=1000, output_tokens=10)
    run, requests, _, _ = _agent_run(tool_repo, _scripted(short), num_ctx=8192)
    assert [message["role"] for message in requests[-1]["messages"]] == ["system", "user", "assistant", "user"]


def test_the_tool_definitions_count_against_the_window_before_the_first_exploration_request(tool_repo: Path):
    """A prompt that leaves room for one tool result but not for the tool definitions as well skips the exploration."""
    tools = rt.RepoTools(tool_repo, "1.0.0", "HEAD", "pkg")
    client = lc.OllamaClient("fake:1b", num_ctx=8192)
    schema_tokens = lc.estimate_tokens(json.dumps(tools.schemas()))
    tool_tokens = lc.estimate_tokens("x" * tools.max_output_chars) + lc.MESSAGE_OVERHEAD_TOKENS
    fits_without_definitions = client.num_ctx - _reserve(client) - tool_tokens  # largest prompt, definitions ignored

    def system_for(prompt_tokens: int) -> str:
        """Return a system text that makes the whole first prompt about ``prompt_tokens`` long, not more."""
        size = int(prompt_tokens * lc.CHARS_PER_TOKEN)
        while sum(lc.estimate_message_tokens({"content": text}) for text in ("s" * size, "u")) > prompt_tokens:
            size -= 1
        return "s" * size

    def rounds(prompt_tokens: int) -> int:
        with fo.FakeOllama(_scripted(fo.reply("seen enough", prompt_tokens=100))) as srv:
            run = agent.run_agent(
                lc.OllamaClient("fake:1b", host=srv.host, num_ctx=8192),
                tools,
                system_for(prompt_tokens),
                "u",
                SCHEMA,
                log=lambda _m: None,
            )
        return run.rounds

    assert rounds(fits_without_definitions) == 0  # the old check would have explored here
    assert rounds(fits_without_definitions - schema_tokens) == 1  # and exactly here the definitions fit as well

"""graphtrace against deepagents' own built-in filesystem tools (`ls`,
`read_file`, `write_file`, `edit_file`, `delete`, `grep`) -- a genuinely
different real surface than test_deepagents_compat.py's sub-agent
delegation: these tools run directly on the top-level agent, no `task`
tool/sub-agent indirection at all, so this is pure direct tool-call tracing.

Every scenario here is adapted from deepagents' own real test file,
tests/unit_tests/test_file_system_tools.py -- including the exact arg
shapes and exact error message strings, confirmed by hand against the real
installed package (e.g. `delete`'s real arg is `file_path`, not `path`, and
`ls` returns a stringified list, not a real list) before writing any of
this, not assumed from the test names alone.

The four path-safety tests are deliberately `category="safety"`,
`severity="high"` -- these are the real security boundary deepagents' own
filesystem tools enforce (no path traversal, no absolute OS paths, no tilde
expansion), and a regression here would be a genuine security issue, not a
cosmetic one.
"""

from __future__ import annotations

import pytest

deepagents_graph = pytest.importorskip("deepagents.graph")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from graphtrace import assert_tool_called  # noqa: E402
from tests.test_deepagents_compat import GenericFakeChatModel  # noqa: E402

create_deep_agent = deepagents_graph.create_deep_agent


@pytest.mark.graphtrace(description="Writing a file and then editing a single occurrence of text in it correctly updates its content.", category="filesystem")
def test_write_then_edit_file_single_replacement(run_graph):
    fake_model = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "write_file", "args": {"file_path": "/code.py", "content": "def hello():\n    print('hello world')"}, "id": "call_write_1", "type": "tool_call"}]),
        AIMessage(content="", tool_calls=[{"name": "edit_file", "args": {"file_path": "/code.py", "old_string": "hello world", "new_string": "hello universe"}, "id": "call_edit_1", "type": "tool_call"}]),
        AIMessage(content="I have edited the file."),
    ]))
    app = create_deep_agent(model=fake_model, checkpointer=InMemorySaver())

    traj = run_graph(app, {"messages": [HumanMessage(content="Edit the file")]}, thread_id="edit-single")

    assert_tool_called(traj, "edit_file", args_match={"old_string": "hello world", "new_string": "hello universe"})
    assert "hello universe" in traj.final_state["files"]["/code.py"]["content"]
    assert "hello world" not in traj.final_state["files"]["/code.py"]["content"]


@pytest.mark.graphtrace(description="edit_file with replace_all=True replaces every occurrence of a string in a file, not just the first.", category="filesystem")
def test_write_then_edit_file_replace_all(run_graph):
    fake_model = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "write_file", "args": {"file_path": "/data.txt", "content": "foo bar foo baz foo"}, "id": "call_write_1", "type": "tool_call"}]),
        AIMessage(content="", tool_calls=[{"name": "edit_file", "args": {"file_path": "/data.txt", "old_string": "foo", "new_string": "qux", "replace_all": True}, "id": "call_edit_1", "type": "tool_call"}]),
        AIMessage(content="I have edited all occurrences."),
    ]))
    app = create_deep_agent(model=fake_model, checkpointer=InMemorySaver())

    traj = run_graph(app, {"messages": [HumanMessage(content="Edit all occurrences")]}, thread_id="edit-replace-all")

    assert traj.final_state["files"]["/data.txt"]["content"] == "qux bar qux baz qux"


@pytest.mark.graphtrace(description="Editing a file that doesn't exist fails with a clear error, not a crash, and the file is never created.", category="error_handling")
def test_edit_nonexistent_file_fails_gracefully(run_graph):
    fake_model = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "edit_file", "args": {"file_path": "/nonexistent.txt", "old_string": "hello", "new_string": "goodbye"}, "id": "call_edit_1", "type": "tool_call"}]),
        AIMessage(content="I tried to edit the file."),
    ]))
    app = create_deep_agent(model=fake_model, checkpointer=InMemorySaver())

    traj = run_graph(app, {"messages": [HumanMessage(content="Edit nonexistent file")]}, thread_id="edit-nonexistent")

    edit_call = next(c for c in traj.tool_calls if c.name == "edit_file")
    assert edit_call.result.content == "Error: File '/nonexistent.txt' not found"
    assert "/nonexistent.txt" not in traj.final_state.get("files", {})


@pytest.mark.graphtrace(description="Editing a file with a search string that isn't actually in it fails with a clear error naming the missing string.", category="error_handling")
def test_edit_file_with_missing_string_fails_gracefully(run_graph):
    fake_model = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "write_file", "args": {"file_path": "/test.txt", "content": "hello world"}, "id": "call_write_1", "type": "tool_call"}]),
        AIMessage(content="", tool_calls=[{"name": "edit_file", "args": {"file_path": "/test.txt", "old_string": "goodbye", "new_string": "farewell"}, "id": "call_edit_1", "type": "tool_call"}]),
        AIMessage(content="I tried to edit the file."),
    ]))
    app = create_deep_agent(model=fake_model, checkpointer=InMemorySaver())

    traj = run_graph(app, {"messages": [HumanMessage(content="Edit with non-existent string")]}, thread_id="edit-string-not-found")

    edit_call = next(c for c in traj.tool_calls if c.name == "edit_file")
    assert edit_call.result.content == "Error: String not found in file: 'goodbye'"


@pytest.mark.graphtrace(description="Writing two files in parallel, then grepping across them, finds the pattern in both and reports which file matched.", category="filesystem")
def test_grep_finds_content_across_parallel_written_files(run_graph):
    fake_model = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[
            {"name": "write_file", "args": {"file_path": "/project/main.py", "content": "import os\nimport sys\n\ndef main():\n    print('Hello World')"}, "id": "call_write_1", "type": "tool_call"},
            {"name": "write_file", "args": {"file_path": "/project/utils.py", "content": "def helper():\n    return 42"}, "id": "call_write_2", "type": "tool_call"},
        ]),
        AIMessage(content="", tool_calls=[{"name": "grep", "args": {"pattern": "import", "output_mode": "content"}, "id": "call_grep_1", "type": "tool_call"}]),
        AIMessage(content="Found the imports."),
    ]))
    app = create_deep_agent(model=fake_model, checkpointer=InMemorySaver())

    traj = run_graph(app, {"messages": [HumanMessage(content="Write files and search")]}, thread_id="grep-parallel")

    assert "/project/main.py" in traj.final_state["files"]
    assert "/project/utils.py" in traj.final_state["files"]
    grep_call = next(c for c in traj.tool_calls if c.name == "grep")
    assert "import" in grep_call.result.content.lower()
    assert "/project/main.py" in grep_call.result.content


@pytest.mark.graphtrace(description="Attempting to edit a file via a path-traversal sequence is blocked with a clear error, never executed.", severity="high", category="safety")
def test_path_traversal_in_edit_file_is_blocked(run_graph):
    fake_model = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "edit_file", "args": {"file_path": "./question/..", "old_string": "test", "new_string": "replaced"}, "id": "call_path_traversal", "type": "tool_call"}]),
        AIMessage(content="I see there was an error with the path."),
    ]))
    app = create_deep_agent(model=fake_model, checkpointer=InMemorySaver())

    traj = run_graph(app, {"messages": [HumanMessage(content="Edit a file with bad path")]}, thread_id="path-traversal-edit")

    edit_call = next(c for c in traj.tool_calls if c.name == "edit_file")
    assert edit_call.result.content == "Error: Path traversal not allowed: ./question/.."


@pytest.mark.graphtrace(description="Reading a file by a real Windows absolute path (not a virtual path) is rejected with a clear error.", severity="high", category="safety")
def test_windows_absolute_path_in_read_file_is_blocked(run_graph):
    fake_model = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"file_path": "C:\\Users\\test\\file.txt"}, "id": "call_windows_path", "type": "tool_call"}]),
        AIMessage(content="I see there was an error with the path."),
    ]))
    app = create_deep_agent(model=fake_model, checkpointer=InMemorySaver())

    traj = run_graph(app, {"messages": [HumanMessage(content="Read a file with Windows path")]}, thread_id="windows-path-read")

    read_call = next(c for c in traj.tool_calls if c.name == "read_file")
    assert read_call.result.content == (
        "Error: Windows absolute paths are not supported: C:\\Users\\test\\file.txt. "
        "Please use virtual paths starting with / (e.g., /workspace/file.txt)"
    )


@pytest.mark.graphtrace(description="Writing a file to a tilde-expanded home-directory path is rejected rather than silently escaping the virtual filesystem.", severity="high", category="safety")
def test_tilde_path_in_write_file_is_blocked(run_graph):
    fake_model = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "write_file", "args": {"file_path": "~/secret.txt", "content": "secret data"}, "id": "call_tilde_path", "type": "tool_call"}]),
        AIMessage(content="I see there was an error with the path."),
    ]))
    app = create_deep_agent(model=fake_model, checkpointer=InMemorySaver())

    traj = run_graph(app, {"messages": [HumanMessage(content="Write a file with tilde path")]}, thread_id="tilde-path-write")

    write_call = next(c for c in traj.tool_calls if c.name == "write_file")
    assert write_call.result.content == "Error: Path traversal not allowed: ~/secret.txt"


@pytest.mark.graphtrace(description="Listing a directory via a path-traversal sequence is blocked with a clear error instead of crashing or leaking the real filesystem.", severity="high", category="safety")
def test_ls_path_traversal_is_blocked(run_graph):
    fake_model = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "ls", "args": {"path": "../../../etc"}, "id": "call_ls_invalid", "type": "tool_call"}]),
        AIMessage(content="I see there was an error with the path."),
    ]))
    app = create_deep_agent(model=fake_model, checkpointer=InMemorySaver())

    traj = run_graph(app, {"messages": [HumanMessage(content="List directory with invalid path")]}, thread_id="ls-path-traversal")

    ls_call = next(c for c in traj.tool_calls if c.name == "ls")
    assert ls_call.result.content == "Error: Path traversal not allowed: ../../../etc"


@pytest.mark.graphtrace(description="Listing a directory after writing a file to it shows that file, and deleting it afterward correctly removes it from the agent's filesystem state.", category="filesystem")
def test_ls_lists_written_file_then_delete_removes_it(run_graph):
    fake_model = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "write_file", "args": {"file_path": "/notes.txt", "content": "hi"}, "id": "call_write_1", "type": "tool_call"}]),
        AIMessage(content="", tool_calls=[{"name": "ls", "args": {"path": "/"}, "id": "call_ls_1", "type": "tool_call"}]),
        AIMessage(content="", tool_calls=[{"name": "delete", "args": {"file_path": "/notes.txt"}, "id": "call_delete_1", "type": "tool_call"}]),
        AIMessage(content="Done."),
    ]))
    app = create_deep_agent(model=fake_model, checkpointer=InMemorySaver())

    traj = run_graph(app, {"messages": [HumanMessage(content="write, list, then delete")]}, thread_id="ls-then-delete")

    ls_call = next(c for c in traj.tool_calls if c.name == "ls")
    assert "/notes.txt" in ls_call.result.content
    delete_call = next(c for c in traj.tool_calls if c.name == "delete")
    assert delete_call.result.content == "Deleted /notes.txt"
    assert "/notes.txt" not in traj.final_state.get("files", {})

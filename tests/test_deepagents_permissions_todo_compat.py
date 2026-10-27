"""graphtrace, pointed at two more of deepagents' own real, built-in
middleware features -- FilesystemPermission-based write/delete enforcement
(deepagents/middleware/filesystem.py) and TodoListMiddleware's planning tool
(langchain.agents.middleware.TodoListMiddleware) -- going through a real,
fully-compiled create_deep_agent() graph end to end, not the unit-level
internal-function tests deepagents' own test_permissions.py and
test_todo_middleware.py use.

Gated with importorskip, not a hard dependency -- see
test_deepagents_compat.py's own module docstring for why `deepagents` is
kept as a dev-extra compatibility test rather than a real dependency.

Running these real scenarios against this real external app found a second
structural gap, distinct from the nested-subgraph-visibility one documented
in test_deepagents_compat.py: TodoListMiddleware's parallel-write_todos
rejection (and, by the same mechanism, FilesystemMiddleware's *first-match*
style checks that short-circuit before the underlying tool is ever invoked)
can reject a tool call entirely within the middleware's own `wrap_tool_call`
without graphtrace's instrumentation -- which patches `BaseTool.invoke`/
`ainvoke` -- ever seeing a call happen at all. The real ToolMessage errors
land in the conversation either way, so the agent's own behavior is
unaffected; only the `Trajectory.tool_calls` *list* undercounts in this one
case. See `test_todo_middleware_rejects_parallel_write_todos_via_graphtrace`
below for the real, verified repro.
"""

from __future__ import annotations

import pytest

deepagents_graph = pytest.importorskip("deepagents.graph")
deepagents_filesystem = pytest.importorskip("deepagents.middleware.filesystem")
langchain_middleware = pytest.importorskip("langchain.agents.middleware")

from langchain_core.messages import AIMessage  # noqa: E402

from graphtrace import assert_state, assert_tool_called  # noqa: E402
from tests.test_deepagents_compat import GenericFakeChatModel  # noqa: E402

create_deep_agent = deepagents_graph.create_deep_agent
FilesystemPermission = deepagents_filesystem.FilesystemPermission
TodoListMiddleware = langchain_middleware.TodoListMiddleware


@pytest.mark.graphtrace(
    description="A FilesystemPermission deny rule blocks a write_file call to a restricted path, with the real permission-denied error visible as the tool's own result.",
    severity="high",
    category="safety",
)
def test_write_denied_by_permission_rule_via_graphtrace(run_graph):
    """Real scenario from deepagents' own test_permissions.py
    (TestFilesystemMiddlewarePermissions.test_write_denied_on_restricted_path),
    exercised here through a full create_deep_agent() graph and a real model
    tool call instead of invoking FilesystemMiddleware's internals directly."""
    model = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "write_file", "args": {"file_path": "/secrets/key.txt", "content": "top secret"}, "id": "call_1"}]),
        AIMessage(content="I'm not able to write to that location."),
    ]))
    app = create_deep_agent(model=model, permissions=[FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="deny")])

    traj = run_graph(app, {"messages": [{"role": "user", "content": "write 'top secret' to /secrets/key.txt"}]})

    write_call = next(c for c in traj.tool_calls if c.name == "write_file")
    assert "permission denied for write" in write_call.result.content
    assert traj.final_message == "I'm not able to write to that location."


@pytest.mark.graphtrace(
    description="The same permission rule set has no effect on a write outside the denied path.",
    category="safety",
)
def test_write_allowed_outside_denied_path_via_graphtrace(run_graph):
    """Contrast case for the test above: the identical rule set, a path it
    doesn't cover, confirming graphtrace sees a normal successful write
    recorded the same way it would with no permissions configured at all."""
    model = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "write_file", "args": {"file_path": "/workspace/notes.txt", "content": "hello"}, "id": "call_1"}]),
        AIMessage(content="Written."),
    ]))
    app = create_deep_agent(model=model, permissions=[FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="deny")])

    traj = run_graph(app, {"messages": [{"role": "user", "content": "write 'hello' to /workspace/notes.txt"}]})

    write_call = next(c for c in traj.tool_calls if c.name == "write_file")
    assert "permission denied" not in write_call.result.content
    assert traj.final_message == "Written."


@pytest.mark.graphtrace(
    description="A recursive delete is refused, all-or-nothing, when any file under the target subtree is write-denied.",
    severity="high",
    category="safety",
)
def test_recursive_delete_denied_when_descendant_restricted_via_graphtrace(run_graph):
    """Real scenario from deepagents' own test_permissions.py
    (TestRecursiveDeletePermissions.test_recursive_delete_refused_when_descendant_denied),
    exercised through a full agent conversation: the agent writes two files,
    one inside a denied subtree, then tries to delete their shared parent
    directory in one call."""
    model = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "write_file", "args": {"file_path": "/work/a.txt", "content": "a"}, "id": "call_1"}]),
        AIMessage(content="", tool_calls=[{"name": "write_file", "args": {"file_path": "/work/secrets/key.txt", "content": "k"}, "id": "call_2"}]),
        AIMessage(content="", tool_calls=[{"name": "delete", "args": {"file_path": "/work"}, "id": "call_3"}]),
        AIMessage(content="I couldn't delete /work -- it contains a restricted file."),
    ]))
    app = create_deep_agent(model=model, permissions=[FilesystemPermission(operations=["write"], paths=["/work/secrets/**"], mode="deny")])

    traj = run_graph(app, {"messages": [{"role": "user", "content": "create /work/a.txt and /work/secrets/key.txt, then delete /work"}]})

    delete_call = next(c for c in traj.tool_calls if c.name == "delete")
    assert "permission denied for write" in delete_call.result.content
    assert "/work/secrets/**" in delete_call.result.content


@pytest.mark.graphtrace(
    description="A single write_todos call updates the agent's todos state, visible on the trajectory's final state.",
    category="planning",
)
def test_todo_list_tracks_single_write_todos_call_via_graphtrace(run_graph):
    """TodoListMiddleware is not in create_deep_agent's default stack --
    added here via its own documented `middleware=[...]` merge point, the
    same real public API deepagents' own test_todo_middleware.py exercises
    directly against create_agent()."""
    model = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "write_todos", "args": {"todos": [{"content": "Task 1", "status": "pending", "activeForm": "Doing task 1"}]}, "id": "call_1"}]),
        AIMessage(content="Planned."),
    ]))
    app = create_deep_agent(model=model, middleware=[TodoListMiddleware()])

    traj = run_graph(app, {"messages": [{"role": "user", "content": "plan the work"}]})

    assert_tool_called(traj, "write_todos")
    assert traj.final_state["todos"] == [{"content": "Task 1", "status": "pending"}]
    assert traj.final_message == "Planned."


@pytest.mark.graphtrace(
    description="Two write_todos calls in the same AIMessage are both rejected, and this rejection happens below graphtrace's own tool-call interception layer.",
    severity="high",
    category="planning",
)
def test_todo_middleware_rejects_parallel_write_todos_via_graphtrace(run_graph):
    """Real scenario copied from deepagents' own test_todo_middleware.py
    (test_todo_middleware_rejects_multiple_write_todos_in_same_message),
    exercised through run_graph() for the first time.

    This is the real repro for the structural gap this module's docstring
    describes: TodoListMiddleware's `wrap_tool_call` recognizes the second
    parallel write_todos call and returns a synthesized error ToolMessage
    for *both* without graphtrace's patched `BaseTool.invoke` ever firing
    for either one -- confirmed by hand that `traj.tool_calls` is empty for
    `write_todos` here, even though the real conversation (`final_state`)
    carries both real error ToolMessages and the todos list stays empty,
    exactly as deepagents' own unit test expects."""
    model = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[
            {"name": "write_todos", "args": {"todos": [{"content": "First task", "status": "in_progress", "activeForm": "Working on first task"}]}, "id": "call_1"},
            {"name": "write_todos", "args": {"todos": [{"content": "First task", "status": "completed", "activeForm": "Working on first task"}, {"content": "Second task", "status": "pending", "activeForm": "Working on second task"}]}, "id": "call_2"},
        ]),
        AIMessage(content="Both tasks have been planned successfully."),
    ]))
    app = create_deep_agent(model=model, middleware=[TodoListMiddleware()])

    traj = run_graph(app, {"messages": [{"role": "user", "content": "plan the work"}]})

    assert not any(c.name == "write_todos" for c in traj.tool_calls)

    tool_messages = [m for m in traj.final_state["messages"] if m.type == "tool"]
    assert len(tool_messages) == 2
    expected_error = (
        "Error: The `write_todos` tool should never be called multiple times in parallel."
        " Please call it only once per model invocation to update the todo list."
    )
    for tool_msg in tool_messages:
        assert tool_msg.content == expected_error
        assert tool_msg.status == "error"

    assert_state(traj, "todos", None)
    assert traj.final_message == "Both tasks have been planned successfully."

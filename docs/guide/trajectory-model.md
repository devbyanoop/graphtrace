# Core concepts: Trajectory & Step

Implemented and tested (`src/graphtrace/core/trajectory.py`, `tests/test_trajectory.py`). Nothing on this page touches LangGraph — that's the point, see [Why core stays framework-agnostic](#why-core-stays-framework-agnostic) below.

## The shape

```python
@dataclass
class LLMCall:
    node: str
    model: str
    input_hash: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    output: Any
    cost_usd: float = 0.0

@dataclass
class ToolCall:
    node: str
    name: str
    args: dict[str, Any]
    result: Any
    error: str | None
    latency_s: float

@dataclass
class Step:
    index: int
    node: str
    kind: Literal["node", "interrupt", "subgraph_enter", "subgraph_exit"]
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    state_diff: dict[str, Any]
    llm_calls: list[LLMCall] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    namespace: tuple[str, ...] = ()
    latency_s: float = 0.0

@dataclass
class Trajectory:
    steps: list[Step]
    final_state: dict[str, Any]
    interrupted: bool
    mode: Literal["record", "replay", "live"]
```

A `Trajectory` is one recorded run: every node visited (`steps`), in order, each carrying its own state transition and whatever it called.

## Derived properties

| Property / method | Returns |
| --- | --- |
| `.path` | `list[str]` of node names, in order — what [`assert_path`](assertions.md#assert_path) checks |
| `.visits(node)` | How many times `node` appears in `.path` |
| `.tool_calls` / `.llm_calls` | Every call of that kind across every step, flattened, in order |
| `.final_message` | The last message's text content, if the state has a `messages` key (supports both LangChain message objects and plain dicts) |
| `.cost_usd` | Sum of every `LLMCall.cost_usd` in the run |
| `.latency_s` | Sum of every `Step.latency_s` |
| `.tokens_total` | Sum of every `LLMCall.prompt_tokens + completion_tokens` |

## Why core stays framework-agnostic

Nothing in `graphtrace.core` may import LangGraph, an LLM SDK, or anything from `graphtrace.runtime` (the layer, not yet built, that will actually run a LangGraph and produce one of these). A `Trajectory` is plain dataclasses and plain Python values — lists, dicts, strings, floats — on purpose:

- **Everything downstream only ever needs a `Trajectory`.** Assertions, budgets, and the eventual HTML report don't know or care whether it came from a real LangGraph run or was built by hand in a test (see the [Quickstart](../getting-started/quickstart.md), which does exactly that).
- **A future non-LangGraph adapter costs nothing extra here.** If graphtrace ever grows a CrewAI or AutoGen adapter, it only has to produce a `Trajectory` too — nothing in this file, or in `assertions.py`, would need to change.

## Where cost comes from

`LLMCall.cost_usd` is a plain field, not something `Trajectory` computes on the fly. This is a deliberate fix relative to the project's original architecture plan, which had `Trajectory.cost_usd` reach into a `runtime/pricing.py` token→USD table directly — that would mean `core` importing from `runtime`, which breaks the rule above. Instead, the (not-yet-built) recorder will resolve a cost in dollars at record time and set `LLMCall.cost_usd` once; `Trajectory.cost_usd` just sums it:

```python
@property
def cost_usd(self) -> float:
    return sum(call.cost_usd for call in self.llm_calls)
```

A hand-built `LLMCall` in a test can simply leave `cost_usd` at its default of `0.0` if cost isn't what that test is checking.

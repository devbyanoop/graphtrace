# Security test pack (v0.2)

!!! warning "Planned for v0.2 — not built yet, and not scheduled for v0.1"
    This is scoped in the project's internal plan but explicitly out of scope for v0.1. See [Roadmap & status](../project/roadmap.md).

## Scope

Red-teaming an agent using the same trajectory/cassette/assertions core v0.1 builds — an attack is just a cassette entry with adversarial content planted in it, no new runtime required.

- **Attack classes:** direct prompt injection (user message), indirect/tool-result injection (a poisoned tool response), tool poisoning (a malicious tool description), data exfiltration, and a "canary" disallowed action that must never fire.
- **Payload library:** seeded from public taxonomies (OWASP's LLM injection patterns, a curated subset of `garak`/`PyRIT` probes) rather than hand-invented strings, versioned the same way rubrics are.
- **Assertions:** `assert_resisted_injection`, `assert_no_canary_fired`, `assert_no_data_leak(patterns=[...])` — softer judgments (half-compliance, "apologizes then does it anyway") go through the same judge contract with a dedicated rubric class.
- Runs as its own `graphtrace redteam` command, or as a marked pytest suite (`@pytest.mark.graphtrace_security`) for teams that want it on every PR rather than gated to pre-release.

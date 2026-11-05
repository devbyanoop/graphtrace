# Security Policy

## Supported versions

graphtrace is pre-1.0 (currently v0.1.x). Only the latest release on PyPI is supported — there's no parallel maintenance of older minor versions yet.

## Reporting a vulnerability

Please report security issues privately rather than opening a public GitHub issue — use GitHub's [private vulnerability reporting](https://github.com/devbyanoop/graphtrace/security/advisories/new) for this repository ("Security" tab → "Report a vulnerability").

Include what you'd include in any bug report: the version, a minimal reproduction, and what you expected vs. what happened. You should get a response within a few days.

## Scope notes specific to this project

- **API keys.** graphtrace's judge feature (`graphtrace calibrate`, live-mode judge calls) calls a real LLM provider via `litellm`, authenticated through whatever provider key is set in your environment (e.g. `.env`, never committed — see `.env.example`). graphtrace never writes a provider key into a cassette, report, or any file it generates.
- **Cassette and config files.** Recorded cassettes (`replay/cassette.py`) and config (`config.py`) are parsed with `ruamel.yaml`'s round-trip and safe loaders respectively — not an unsafe loader that executes arbitrary Python tags — so loading a cassette or config file you didn't write yourself doesn't execute arbitrary code by construction. That said, a cassette is still data from wherever you got it; don't point `mode="replay"` at a cassette file you don't trust.
- **The HTML report.** `pytest --gt-report=report.html` renders with Jinja2's `autoescape=True`, so test names, descriptions, and trajectory content are escaped — not a known XSS vector — but the report is a static file meant to be shared locally or in CI artifacts, not served from an untrusted multi-tenant web app.

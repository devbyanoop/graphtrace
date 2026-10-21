# CLI reference

!!! warning "Planned — no CLI exists yet"
    graphtrace is currently a library (`pip install -e .`) plus a pytest test suite — there's no `graphtrace` console command yet. See [Roadmap & status](../project/roadmap.md).

## Planned commands

| Command | Will do | Status |
| --- | --- | --- |
| `graphtrace record` | One-shot cassette recording outside pytest (e.g. from a notebook or script) | Planned |
| `graphtrace from-trace` | Convert an exported production trace into a cassette + generated test | Planned |
| `graphtrace calibrate` | Score the judge against a human-labelled set, report agreement/kappa | Planned |
| `graphtrace report` | Render the static HTML report from existing JSON results | Planned |
| `graphtrace redteam` | Run the security test pack (v0.2) | Planned, v0.2 |

## Planned pytest options

| Flag / env var | Controls |
| --- | --- |
| `--gt-mode` / `GRAPHTRACE_MODE` | `replay` \| `record` \| `record_missing` \| `live` |
| `--gt-report` | Where to write the HTML report |
| `--gt-export` | Observability export target (v0.2) |

See [Recording & replay](../guide/recording-and-replay.md) for what each mode means.

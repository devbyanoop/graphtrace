"""Trace importers: turn a production trace export (Langfuse, OTel/OpenInference,
LangSmith) into the same NormalizedTrace shape, regardless of source format.

Adding a new source is one file implementing `TraceImporter` -- nothing else in
the codebase needs to change, the same plugin-shaped design kbguard uses for its
own vector-store adapters.
"""

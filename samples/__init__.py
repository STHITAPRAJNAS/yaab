"""YAAB samples — end-to-end agent apps and patterns.

Each sample is a self-contained package exposing ``build(model=None)`` that
returns a ready-to-run agent (or app). With no ``model`` it uses a deterministic
``TestModel`` so the sample — and its test — runs fully offline. Pass a real
model string (e.g. ``"ollama/llama3"``, ``"gemini/gemini-2.0-flash"``,
``"openai/gpt-4o"``) to run it for real.

See ``samples/README.md`` for the catalog and how to use a free model.
"""

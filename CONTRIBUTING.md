# Contributing to YAAB

Thanks for your interest! YAAB is a Python-first SDK with an optional Rust
performance core. Contributions are welcome — bug fixes, integrations (new
models, vector stores, rerankers, sinks, metrics, compliance mappers), docs, and
tests.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev,all]'
# Optional: build the Rust core (otherwise the pure-Python fallback is used)
maturin develop -m yaab-core/Cargo.toml --release
```

## Before you open a PR

Run the same gates CI runs:

```bash
ruff check yaab/ tests/ examples/        # lint
ruff format --check yaab/ tests/         # format
mypy yaab/                               # types
pytest -q                                # tests (Rust backend if built)
YAAB_NO_RUST=1 pytest -q                 # tests on the pure-Python fallback
cargo fmt --manifest-path yaab-core/Cargo.toml --check
cargo clippy --manifest-path yaab-core/Cargo.toml -- -D warnings
```

**Both backends must stay green** — anything in `yaab._core` needs a Rust
implementation *and* a matching pure-Python fallback.

## Design principles

- **Extensible by default.** New capabilities are `typing.Protocol`s and/or
  registered components (`yaab.extensions`) discoverable via entry points — not
  hard-coded into the core.
- **Optional dependencies stay optional.** Import heavy/third-party libs lazily
  (inside the function/factory), and raise a clear "install X" error if missing.
- **Governance is a first-class runtime concern**, toggled by mode.
- Match the surrounding code's style, type hints, and docstring density.

## Adding an integration

Implement the relevant protocol (e.g. `VectorStore`, `Reranker`, `AuditSink`,
an eval metric), register it under the matching component kind, and add it to
`pyproject.toml` as an optional extra + an entry point where appropriate. Add
tests that run offline (use `TestModel` and dependency-free fakes).

## Commits & PRs

Keep PRs focused; include tests and a CHANGELOG entry under `[Unreleased]`.

# Docs & examples maturity — design

**Date:** 2026-06-01
**Goal:** Close the documentation and example-quality gaps surfaced by the
2026-06-01 multi-agent audits (yaab vs Google ADK / LangGraph), so that:
1. every confirmed defect in docs/examples is fixed,
2. `examples/` has the same CI-enforced test coverage `samples/` already has, and
3. the docs ship as a hosted, searchable site with an auto-generated API
   reference — the two gaps the audit rated *critical*.

**Branch / PR:** all three tiers land on one branch
(`feature/docs-examples-maturity`, developed in an isolated worktree) as a
single PR into `develop`.

**Out of scope (this wave):** notebooks, K8s/deployment manifests,
failure-mode/anti-pattern doc sections, community templates
(CODE_OF_CONDUCT, issue/PR templates), docstring backfill, `mike` versioning,
and any public-API code changes (e.g. exporting `Role` from top-level `yaab`,
a public alias for `_get_runner().run_stream`). API changes belong to the
parallel feature-gap workstream (see Coordination). Also deferred to
follow-up waves: examples for the Wave 1 features, the SRE incident/runbook
docs page (kill switch / replay / audit-trail query), perf-harness docs,
golden-signal metrics docs — these depend on the feature workstream's
roadmap items.

---

## Coordination with the feature-gap workstream

A separate agent is concurrently closing feature gaps vs ADK. Its
**ADK-parity Wave 1 (PR #26) is already merged into `develop`** and into this
branch: model router, OpenAPI toolset, EvalSet, memory intelligence
(extraction), graph retries, context caching, per-run cost tracking. Two
consequences:

1. **Conflict surface is kept minimal.** This wave touches `docs/`,
   `examples/`, `samples/coding_helper/`, `tests/`, `mkdocs.yml`,
   `.github/workflows/`, and `pyproject.toml` (docs extra only). The only
   files both waves may touch are `ci.yml` and `pyproject.toml` — both are
   small, append-only edits here, so merge conflicts are trivially resolvable.
2. **New features will need new examples.** Each Wave 1 capability (model
   router, OpenAPI tools, EvalSet, memory extraction, graph retries) should
   get an example/sample *plus an entry in `tests/test_examples.py`*. The
   test harness built in Tier 2 is designed so adding a new example is a
   two-line parametrization change. That follow-up examples wave is
   explicitly **not** part of this PR. The new modules DO get API-reference
   pages in Tier 3 (§3.2), since those are auto-generated.

---

## Tier 1 — Defect fixes

Every defect below was confirmed by adversarial verification against the
codebase on 2026-06-01.

### 1.1 Windows cp1252 crash in the flagship example
**Problem:** `examples/01_quickstart.py:40` prints a literal `→` (U+2192). On
a default Windows console (cp1252) `print()` raises `UnicodeEncodeError`; a
Windows newcomer's first run of the quickstart ends in a traceback.

**Design:** Replace with ASCII `->`. Sweep all 10 examples + 7 samples for
non-ASCII characters in `print()` literals and replace likewise. The Tier 2
subprocess smoke test (run on the new Windows CI job) makes this class of bug
permanently unshippable.

### 1.2 `state.md` Role-import NameError
**Problem:** `docs/state.md` (Sessions snippet, ~line 24) calls
`sessions.append_text(s.id, Role.USER, "Hello")` without importing `Role`.
`Role` lives in `yaab.types` and is not exported from top-level `yaab`, so
the snippet raises `NameError` when pasted.

**Design:** Add `from yaab.types import Role` to the snippet. (Exporting
`Role` from top-level `yaab` would be nicer UX but is an API change —
deferred to the feature workstream.)

### 1.3 `samples/coding_helper` fabricated success
**Problem:** The scripted `TestModel(custom_output="The sum is 45.",
call_tools=["python_exec"])` emits an empty-args tool call; `python_exec`
requires `code: str`, so the tool fails Pydantic validation at runtime. The
printed "45" is the canned output — identical whether the tool runs, errors,
or is rejected. The covering test (`'45' in out`) passes on the fabrication.
The headline sandbox + HITL pattern is never actually exercised.

**Design:**
- Seed the scripted model so the tool call carries real arguments
  (e.g. `{"code": "print(sum(range(10)))"}`) and the sandbox actually executes.
- The final answer must reflect the real tool result (e.g. via a
  `FunctionModel` that echoes the tool result), so success is observable and
  rejection is observably different.
- Strengthen `tests/test_samples.py`: the approved path asserts the *real*
  computed value flowed through; the rejected path asserts the output
  differs from the approved run (not just `'45' in out`).
- Add the "SubprocessSandbox is not a security boundary" caveat to the
  sample's docstring (mirrors SECURITY.md).

### 1.4 `streaming-events.md` documents events that don't exist
**Problem (verified against `yaab/runner.py`):** the EventType table lists a
`GUARDRAIL` event the Runner never emits, labels per-token deltas as
`MODEL_DELTA` (the real per-token event is `TEXT_DELTA`), and omits
`MODEL_REQUEST`/`TEXT_DELTA`.

**Design:** Regenerate the table from the actual `EventType` enum and Runner
emit sites; every row gets a "when it fires" description verified against
source. The `agent._get_runner().run_stream(...)` surface stays documented
as-is — verification confirmed it is the intentional semantic-stream API used
by `yaab/serve.py` and `yaab/agui.py`; renaming it is an API change for the
feature workstream.

### 1.5 Stale meta-docs
**Problem:** Three documents contradict the code:
- `COMPARISON.md` §3: "No `yaab web` dev UI yet" — but `yaab/web.py` and the
  CLI subcommand exist.
- `ROADMAP.md` demand table: lists "no Postgres/Redis yet" and "no global
  switch or trace redaction" as pending; both are implemented.
- `CHANGELOG.md`: names six samples ("research assistant", "document Q&A", …)
  that don't exist; the repo ships seven differently-named samples.

**Design:** Update each to match the code as of this branch. Add a one-line
"verified against code on 2026-06-01" footer to COMPARISON.md and ROADMAP.md
so staleness is at least dated.

### 1.6 Onboarding inconsistencies
**Problem:** `index.md` lists install extras `[litellm]/[otel]/[all]` while
`get-started.md` lists `[rust]/[litellm]`; a get-started comment says the
builtin tool is `time` but it is `current_time`; `quickstart.md` claims
"Every example in these docs runs without a network using TestModel" while
most snippets show `openai/gpt-4o`.

**Design:** Single canonical install matrix (in `get-started.md`, linked from
`index.md`); fix the tool name; reword the offline claim to "every example
*can* run offline by swapping in `TestModel`" and link to models.md's
"Testing without a network" section.

---

## Tier 2 — Example test coverage

### 2.1 Refactor: every example exposes `main()`
**Problem:** All 10 `examples/*.py` run code at import time (no
`if __name__ == "__main__":` guard), so they cannot be imported for testing
without side effects.

**Design:** Identical mechanical pattern for all 10:

```python
async def main() -> <result data>:   # returns asserted-on data; prints stay
    ...

if __name__ == "__main__":
    asyncio.run(main())
```

`main()` returns the data a test needs (e.g. quickstart returns the three
result strings; `02_graph_hitl` returns the final `GraphResult`). Printed
output is unchanged so the user-facing behavior is identical.

### 2.2 `tests/test_examples.py` — two layers
1. **Logic tests (in-process):** import each example's `main()`, run it via
   `asyncio.run`/pytest-asyncio, assert on returned data. Examples stay
   offline (TestModel) so tests are deterministic and fast.
2. **Smoke tests (subprocess):** one parametrized test runs each
   `examples/*.py` file with `sys.executable`, asserts exit code 0. This
   catches entry-point breakage and (on the Windows CI job) console-encoding
   crashes — exactly the class of bug found in 1.1.

Adding a future example = adding one parametrize entry per layer.

### 2.3 `tests/test_serve_app.py` — cover the deployment artifact
**Problem:** `examples/serve_app.py` is what `yaab serve` and the Dockerfile
load, yet nothing tests it; existing serve tests build their own agents.

**Design:** A test that loads `examples.serve_app:agent` through the same
import path `yaab serve` uses (`yaab.cli`'s module:attr loader), mounts it
with `fastapi_server_app`, and drives `/run` via the ASGI test client,
asserting the canned TestModel response. Also fix the docstring's misleading
claim that `YAAB_AGENT` selects "a real model" (it selects the module).

### 2.4 CI changes (`.github/workflows/ci.yml`)
- Add `examples/` to `ruff format --check` (currently lint-only).
- New `windows` job: `windows-latest`, Python 3.13, `YAAB_NO_RUST=1`
  (pure-Python — avoids Rust toolchain setup cost on Windows), runs the full
  pytest suite. This is the job that makes cp1252-class bugs unshippable.
- The existing ubuntu matrix picks up `tests/test_examples.py` and
  `tests/test_serve_app.py` automatically (both backends).

### TDD discipline
Each Tier 1 defect gets its failing test written first where a test can
express it (1.1 → subprocess smoke under forced cp1252 io encoding;
1.3 → strengthened sample tests). Doc-only fixes (1.4–1.6) are verified by
the Tier 3 doc-snippet tests where applicable, otherwise by review.

---

## Tier 3 — Docs site & API reference

### 3.1 MkDocs Material site
- `mkdocs.yml` at repo root: Material theme, `docs/` as source, nav mirroring
  `docs/index.md`'s TOC (Get started → Concepts → per-topic guides →
  Operations → Meta), built-in search.
- `docs/index.md` stays the nav source of truth; mkdocs nav lists the same
  files in the same grouping.
- Internal links are already relative `.md` links → they work unchanged.

### 3.2 Auto-generated API reference (mkdocstrings)
- `mkdocstrings[python]` plugin; new `docs/api/` section with one page per
  public module: `yaab` (top-level exports), `yaab.tools` (incl. the new
  `yaab.tools.openapi`), `yaab.models` (incl. the new `yaab.models.router`),
  `yaab.graph`, `yaab.multiagent`, `yaab.rag`, `yaab.governance` (incl. the
  new `yaab.governance.evalset`), `yaab.sessions`/`memory`/`artifacts`
  (incl. the new `yaab.memory.extraction`), `yaab.testing`, `yaab.serve`.
- Renders existing docstrings + signatures (source is `py.typed` with PEP 224
  attribute docstrings). Coverage holes are accepted this wave; backfill is a
  follow-up.

### 3.3 Deploy: GitHub Pages
- `.github/workflows/docs.yml`:
  - On PR: `mkdocs build --strict` (broken links/anchors/nav fail the check).
  - On push to `develop`: build + deploy via `actions/deploy-pages` to
    `https://sthitaprajnas.github.io/yaab/`.
- One-time repo setting (Settings → Pages → Source: GitHub Actions) — done at
  merge time via `gh api` or by the owner.
- README gets a docs badge + link.

### 3.4 Doc-snippet testing (scoped)
- `pytest-examples` runs the fenced Python blocks of the three
  onboarding-critical files: `quickstart.md`, `get-started.md`, `state.md`
  (this is the test that would have caught the `Role` NameError).
- Snippets that intentionally require a live key (`openai/gpt-4o`) are marked
  to skip execution but still lint/format.
- Advanced topic pages keep illustrative placeholders and are *not* executed
  (running them would mean inventing fake symbols).

### 3.5 llms.txt
- `mkdocs-llmstxt` plugin generates `llms.txt` (and `llms-full.txt`) at build
  time from the nav — parity with LangGraph's docs feature.

### 3.6 Packaging
- `pyproject.toml`: new `docs` extra — `mkdocs-material`,
  `mkdocstrings[python]`, `mkdocs-llmstxt`, `pytest-examples`.
- These never install with `[all]`; docs tooling stays out of the runtime
  footprint. The doc-snippet tests are skipped automatically when the `docs`
  extra isn't installed (CI installs it; contributors may not).

---

## Cross-cutting principles
- **TDD:** failing test → fix → green, for every change a test can express.
- **No behavior changes to the library:** this wave touches zero `yaab/`
  runtime code. (The only source-adjacent change is `samples/coding_helper`,
  which is a sample, not the library.)
- **Examples stay offline-first:** every example/test runs with no API key
  and no network, preserving the project's verified differentiator.
- **Keep the conflict surface small:** `ci.yml` and `pyproject.toml` edits
  are append-only; everything else is in files the feature workstream
  doesn't touch.

## Acceptance criteria
1. All Tier 1 defects fixed; each has a test or doc-test that fails on the
   old code/doc where expressible.
2. `pytest` green on ubuntu (both backends) *and* the new Windows job.
3. `tests/test_examples.py` covers all 10 examples (logic + smoke);
   `tests/test_serve_app.py` covers the serve artifact.
4. `mkdocs build --strict` passes; the site deploys to GitHub Pages on merge;
   the API reference renders for all listed modules.
5. `llms.txt` is generated and served.
6. Existing tests (399 passed / 3 skipped baseline, post-Wave-1 merge) stay
   green throughout.

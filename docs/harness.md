# Coding harness

The **coding harness** is a packaged, automatable coding agent — the kind of
autonomous "do work in a repo" agent a CLI coding tool ships, but as a plain
`Agent` you construct in one call and run headlessly, in CI, or embedded in your
own app. It is provider-neutral (any LiteLLM model) and safe by construction:
every action that writes, spawns a process, reaches the network, schedules, or
reads the environment is gated; only reads run unattended.

It is **not** a deep-research agent. It is a focused loop + a sandboxed
filesystem/shell/web toolset + a capability approval gate + budgets + resume.

```python
from yaab import coding_agent

agent = coding_agent(root="./workspace", model="anthropic/claude-sonnet-4-6")
result = await agent.run("Add a docstring to utils.py and show the diff.")
print(result.output)
```

## What you get

`coding_agent(root=...)` returns an ordinary [`Agent`](api/agent.md) wired with:

- **Sandboxed file tools** — `file_read`, `file_write`, `file_list`, and
  `file_edit` (surgical, single-match edits), all confined to `root`. Traversal,
  symlinks, and protected paths (`.git`, lockfiles) are refused.
- **`file_edit` safety** — an edit must follow a `file_read` of the same file
  (read-before-edit) and is rejected if the file changed on disk since (a
  content-hash staleness guard). Ambiguous matches require `replace_all`. Oversized
  files (>1 MB) are refused rather than truncated. Success returns a unified diff.
- **Optional web access** — `web_search` and `fetch_url` (SSRF-guarded), on by
  default; turn off with `allow_net=False`.
- **Optional shell** — `shell_exec`, off by default (see below).
- **An `AGENTS.md` brief** — if `root/AGENTS.md` exists, it is appended to the
  agent's instructions automatically.
- **A bounded loop** — `max_steps` caps iterations; the `yaab run` CLI also applies
  a token/request/wall-clock budget.
- **An approval gate** — a [`ToolApprovalPlugin`](governance.md) that gates by
  *capability* (effect), not tool name.

## The security model

The harness gates by **capability**, so a destructive action is caught no matter
which tool performs it. Capabilities: `fs_read`, `fs_write_in_root`,
`fs_write_out`, `net_egress`, `process_spawn`, `env_read`, `schedule`.

- **Gated by default** (`DEFAULT_GATE`): writes, process spawns, network egress,
  scheduling, and env reads. Only `fs_read` runs unattended.
- **Fail-closed.** If any tool carries a gate-worthy capability the gate does not
  cover, construction raises — you cannot accidentally ship an ungated
  destructive tool. Narrow the gate too far and `coding_agent()` refuses to build.
- **No silent auto-approve.** With no approver wired, a gated tool *pauses* the run
  for out-of-band sign-off; it never runs unreviewed.
- **Shell is opt-in and isolation-checked.** `enable_shell=True` requires a
  `sandbox=` and, unless you *explicitly* downgrade, an **isolating** one.
- **MCP tools are gated by name** regardless of the capabilities they declare —
  opaque external effects don't get a free pass.

### Allowlisted shell

`shell_exec` never sees a shell. The model passes an **argv list**
(`["git", "status"]`), so `;`, `|`, and `$(...)` are inert. Each binary must be in
an allowlist, and each carries a `ShellRule` constraining its arguments:

```python
from yaab import coding_agent
from yaab.tools.builtin.shell import ShellRule
from yaab.tools.exec import DockerCommandSandbox

agent = coding_agent(
    root="./workspace",
    enable_shell=True,
    sandbox=DockerCommandSandbox(workdir="./workspace"),  # is_isolated=True
    shell_rules={
        "git": ShellRule(subcommands=frozenset({"status", "diff", "log"})),
        "pytest": ShellRule(max_args=4, deny_substrings=("..",)),
    },
)
```

An empty ruleset allows *nothing*. To run on a non-isolating sandbox (e.g. a dev
box without Docker) you must pass `allow_unsandboxed_shell=True` — an informed
downgrade, never the default.

## Running headlessly: `yaab run`

```bash
# the bundled coder on the current directory
yaab run --coding --root . "Fix the failing test in test_math.py"

# any of your own agents (module:attribute, like `yaab serve`)
yaab run app.main:agent "summarize today's changes" --json
```

`yaab run` is built for automation:

- **The answer goes to stdout; everything else to stderr.** `--json` makes stdout
  a single envelope (`status`, `output`, `paused`, `resume_id`, `usage`).
- **Deterministic exit codes** so CI can branch on *why* a run stopped:

  | code | meaning | code | meaning |
  |---|---|---|---|
  | 0 | ok | 6 | cancelled (SIGINT/SIGTERM) |
  | 1 | error | 7 | max steps exceeded |
  | 2 | CLI usage error | 8 | output validation failed |
  | 3 | paused for approval | 9 | model/provider error |
  | 4 | budget exceeded | 10 | policy violation |
  | 5 | timeout | | |

- **Always bounded.** Every run gets a `UsageLimits`; raise the ceilings with
  `--budget-tokens`, `--max-tool-calls`, `--max-steps`, `--timeout`.
- **Approval policy is explicit** — pick one, or none to pause:
  - `--auto` — auto-approve gated tools (trusted automation).
  - `--read-only` — deny every gated tool (with the default gate, read-only).
  - `--approve-caps fs_write_in_root,net_egress` — auto-approve only those effects.
  - *(none, interactive TTY)* — prompt y/N per gated tool.
  - *(none, non-interactive)* — pause and print a resume id (exit 3).

  There is deliberately no blanket `--yes`.
- **Interruptible & resumable.** `SIGINT`/`SIGTERM` cancels cooperatively; with
  `--state-db runs.db` a pause survives the process and `--resume-id <id>`
  continues it.

```bash
# pause on first write (non-interactive), then resume after reviewing
yaab run --coding --root . --state-db runs.db "refactor utils.py"   # exit 3, prints resume id
yaab run --coding --root . --state-db runs.db --approve-caps fs_write_in_root --resume-id <id>
```

## Composing it

The return value is a normal `Agent`, so the harness composes with the rest of
YAAB: give it `extra_tools=[...]`, attach it as a sub-agent, run it under your own
[`Runner`](api/runner.md) with governance plugins, or point an
[evaluation](evaluation.md) set at it. Everything in [Tools](tools.md),
[Human-in-the-loop](hitl.md), [Usage limits](limits.md), and
[Durable runs](durable-runs.md) applies.

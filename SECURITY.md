# Security Policy

## Reporting a vulnerability

Please report security issues **privately**, not via public issues. Use GitHub's
[private vulnerability reporting](https://github.com/sthitaprajnas/yaab/security/advisories/new)
or email the maintainers. We aim to acknowledge within 3 business days and to
ship a fix or mitigation as fast as the severity warrants.

Please include: affected version, a reproduction, and the impact you observed.

## Supported versions

YAAB is alpha (0.x). Security fixes target the latest released minor version.

## Security model & sharp edges

YAAB is positioned for regulated/enterprise use; please understand these
boundaries before deploying:

- **Code execution (`python_exec`) is not a security sandbox by default.** The
  default `SubprocessSandbox` isolates crashes/hangs (separate process +
  timeout) but does **not** contain a malicious payload. For untrusted input,
  use `DockerSandbox` (no network, read-only root, dropped capabilities,
  cpu/mem/time caps) **and** gate the tool behind `ToolApprovalPlugin` /
  `ToolAuthorizationPlugin`. Run the agent process itself in a container/VM for
  real isolation.
- **Tools run arbitrary code you give them.** Authorize side-effecting tools
  (`ToolAuthorizationPlugin`), require human approval for sensitive ones
  (`ToolApprovalPlugin`), and use `IdempotencyPlugin` to avoid double execution.
- **Retrieved context is untrusted input.** Use retrieval guardrails
  (`min_score`, `context_guard`) to defend against context/memory poisoning.
- **Prompt injection is a live risk.** The `PolicyEngine` ships input/output
  scanners (prompt-injection, PII, secrets, system-prompt-leak); enable
  `GovernanceMode.ENFORCING` in production.
- **Secrets:** never hard-code API keys; the SDK reads provider keys from the
  environment via LiteLLM. The audit log supports redaction — configure it
  before logging sensitive payloads.
- **The compliance mappers produce *evidence*, not legal sign-off.** Effective
  challenge and conformity assessment require qualified human reviewers.

## Dependencies

The core depends only on `pydantic` + `typing-extensions`. Every integration
(LiteLLM, FastAPI, vector stores, eval suites, etc.) is an **optional extra**,
so your attack surface is only what you install.

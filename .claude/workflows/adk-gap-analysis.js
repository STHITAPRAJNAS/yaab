export const meta = {
  name: 'adk-gap-analysis',
  description: 'Exhaustive YAAB vs Google ADK gap analysis, verified against the codebase',
  phases: [
    { title: 'Analyze', detail: 'one agent per capability cluster: read YAAB code vs ADK 2.0 features' },
    { title: 'Verify', detail: 'adversarially verify every claimed gap against the YAAB codebase' },
  ],
}

const FINDINGS_SCHEMA = {
  type: 'object',
  properties: {
    cluster: { type: 'string' },
    yaab_strengths: { type: 'array', items: { type: 'string' }, description: 'things YAAB has that match or beat ADK in this cluster' },
    gaps: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          feature: { type: 'string', description: 'short name of the missing/weaker capability' },
          adk_capability: { type: 'string', description: 'what ADK provides exactly' },
          yaab_status: { type: 'string', enum: ['missing', 'partial', 'weaker'] },
          detail: { type: 'string', description: 'precise description of the delta, citing YAAB files inspected' },
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          effort: { type: 'string', enum: ['small', 'medium', 'large'] },
        },
        required: ['feature', 'adk_capability', 'yaab_status', 'detail', 'severity', 'effort'],
      },
    },
  },
  required: ['cluster', 'yaab_strengths', 'gaps'],
}

const VERDICTS_SCHEMA = {
  type: 'object',
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          feature: { type: 'string' },
          confirmed: { type: 'boolean', description: 'true if the gap is REAL (YAAB genuinely lacks it)' },
          correction: { type: 'string', description: 'if refuted, cite the YAAB file/class that provides it; if confirmed but severity wrong, say why' },
          revised_severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
        },
        required: ['feature', 'confirmed'],
      },
    },
  },
  required: ['verdicts'],
}

const repoNote = `The YAAB SDK repo is the current working directory. Use Read/Grep/Glob to inspect code. Do NOT guess about YAAB — every claim about YAAB must come from reading actual files. Key layout: yaab/agent.py, yaab/runner.py, yaab/types.py, yaab/models/, yaab/tools/, yaab/sessions/, yaab/memory/, yaab/artifacts/, yaab/rag/, yaab/graph/, yaab/multiagent.py, yaab/optimize/, yaab/governance/ (registry, lifecycle, policy, audit, eval, compliance, monitor, authorization, approval, guardrails/), yaab/plugins/, yaab/skills.py, yaab/prompts.py, yaab/config.py, yaab/context.py, yaab/limits.py, yaab/streaming.py, yaab/agui.py, yaab/a2a/, yaab/serve.py, yaab/auth.py, yaab/web.py, yaab/cli.py, yaab/batch.py, yaab/extensions.py, yaab/observability/, yaab/eval/, docs/, tests/.`

const analyzePrompt = (c) => `You are auditing the YAAB agent SDK against Google ADK 2.0 for the capability cluster: **${c.name}**.

${repoNote}

## What Google ADK 2.0 provides in this cluster (ground truth from current ADK docs):
${c.adk}

## Your job
1. Read the relevant YAAB code thoroughly: ${c.yaabAreas}. Also check docs/ and tests/ for evidence of features.
2. For EVERY ADK capability listed above, determine whether YAAB has it (equivalent or better), has it partially, has it but weaker, or lacks it entirely.
3. Also note YAAB strengths in this cluster that ADK lacks (be specific).
4. The bar: the user wants YAAB to be "super smart, feature rich, fast and fool-proof" — judge depth, not just presence. A stub or simplified version of an ADK feature counts as 'partial' or 'weaker', not a match.

Return findings in the structured schema. Severity guide: critical = adopters would reject YAAB over this; high = frequently needed in production; medium = important for parity marketing; low = niche. Be exhaustive — list every real gap, but do NOT invent gaps for things YAAB actually has (you will be adversarially fact-checked against the codebase).`

const verifyPrompt = (c, findings) => `You are an adversarial fact-checker for an audit of the YAAB SDK. Another agent claimed YAAB has these gaps vs Google ADK in the cluster "${c.name}". Your job is to try to REFUTE each claim by finding evidence in the YAAB codebase that the capability actually exists.

${repoNote}

## Claimed gaps to verify:
${JSON.stringify(findings.gaps, null, 2)}

For each claimed gap:
1. Search the codebase (Grep/Glob/Read) for the capability — check yaab/, docs/, tests/, scripts/, samples/, examples/. Features are sometimes in unexpected modules (e.g. context compaction in yaab/context.py, RBAC in yaab/governance/authorization.py, callbacks as 'plugins' in yaab/plugins/).
2. confirmed=true ONLY if YAAB genuinely lacks it (or genuinely has only a weaker version as claimed).
3. confirmed=false if you find YAAB actually provides it — cite the file/class in 'correction'.
4. If the gap is real but the severity seems wrong (e.g. claimed critical but it's a niche GCP-only feature), set revised_severity and explain.

Be rigorous: the cost of a FALSE gap (claiming YAAB lacks something it has) is high — it goes into a public report.`

const ARGS = typeof args === 'string' ? JSON.parse(args) : args
const CLUSTERS = Array.isArray(ARGS) ? ARGS : ARGS.clusters

const results = await pipeline(
  CLUSTERS,
  (c) => agent(analyzePrompt(c), { label: 'analyze:' + c.key, phase: 'Analyze', schema: FINDINGS_SCHEMA }),
  (findings, c) => {
    if (!findings || !findings.gaps || findings.gaps.length === 0) {
      return { cluster: c.key, findings, verdicts: [] }
    }
    return agent(verifyPrompt(c, findings), { label: 'verify:' + c.key, phase: 'Verify', schema: VERDICTS_SCHEMA })
      .then(v => ({ cluster: c.key, findings, verdicts: v ? v.verdicts : [] }))
  }
)

const out = results.filter(Boolean)
log('Clusters analyzed: ' + out.length + ' / ' + CLUSTERS.length)
return out
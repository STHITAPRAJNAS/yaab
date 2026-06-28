# Research assistant

A durable, multi-step research flow: retrieve grounding from a **hybrid
(BM25 + dense)** knowledge base, then draft a brief from it — with the draft
**streaming** token-by-token.

```bash
python -m cookbook.apps.research_assistant
```

Output (offline):

```python
{'brief': 'The Antikythera mechanism is an ancient analog computer.',
 'streamed_chunks': 4}
```

## What it shows

| Capability | Where |
|---|---|
| **Flow** | `retrieve → draft` steps lowered onto the durable engine |
| **Hybrid retrieval** | `KnowledgeBase(hybrid=True)` fuses BM25 + dense recall so exact terms surface |
| **Grounded drafting** | the draft agent is instructed from the retrieved `{notes}` |
| **Streaming** | the draft streams token-by-token for a live UI |

## Going live

Set `YAAB_SAMPLE_MODEL=gemini/gemini-2.5-flash` to draft with a real model. For
sources without an API, add the [browser tools](../../recipes/browser.py)
(`browser_toolset(allow_domains=[...])`) as a `retrieve` step. To make a live run
reproducible in docs/CI, wrap the model in a [`CassetteModel`](../../recipes/record_replay.py):
record once with `YAAB_RECORD=1`, then replay deterministically.

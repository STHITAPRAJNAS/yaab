# Prompts & skills

## Prompt management & versioning

Prompts are first-class, versioned artifacts — not strings buried in code. A
`PromptRegistry` stores immutable versions, tracks the active one, and stamps a
content hash so a prompt change is auditable and can be pinned.

```python
from yaab import PromptRegistry

prompts = PromptRegistry()
prompts.register("greeting", "Hello {name}, how can I help?")
prompts.register("greeting", "Hi {name} — what can I do for you?")   # creates v2

prompts.render("greeting", name="Alice")               # uses the active (latest) version
prompts.get("greeting").render(version=1, name="Alice")  # pin a specific version
prompts.get("greeting").get(2).hash                      # content hash for audit
```

Render a prompt into an agent's instructions, or pin a version for deterministic
production behavior alongside a [compiled optimizer artifact](optimization.md).

## Skills (reusable bundles)

A `Skill` packages an instruction fragment, tools, an optional prompt, and
declared permissions so a capability can be attached to any agent and shared
across a team. Skill permissions feed the registry's action scope, and skill
tools appear in the agent card.

```python
from yaab import Agent, tool
from yaab.skills import Skill

@tool
def web_search(query: str) -> str:
    """Search the web."""
    return "..."

research = Skill(
    name="research",
    instructions="Always search before answering factual questions.",
    tools=[web_search],
    permissions=["net:read"],
    version="1.0.0",
)

agent = Agent("analyst", model="openai/gpt-4o", skills=[research])
assert "net:read" in agent.permissions          # surfaced for governance
assert any(t.name == "web_search" for t in agent.tools)
```

### Sharing skills as packages

Third parties can publish skills discoverable via the `yaab.skills` entry point:

```toml
# pyproject.toml of a plugin package
[project.entry-points."yaab.skills"]
research = "my_pkg.skills:research_skill"
```

```python
from yaab.skills import load_skills
available = load_skills()      # {"research": Skill(...), ...}
```

See [Extending YAAB](extending.md) for the full extensibility model.

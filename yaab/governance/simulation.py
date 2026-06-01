"""User-simulation driver for multi-turn agent evaluation.

Conversational agents are best evaluated by standing up a *simulated user* — an
LLM playing a persona with a goal — and letting it drive a multi-turn dialogue
against the agent under test. This is strictly more powerful than a fixed,
pre-scripted conversation: the simulated user *reacts* to what the agent says,
so the eval exercises the agent's real turn-taking, clarification, and recovery
behavior rather than a happy-path transcript the author imagined up front.

This module provides three pieces:

- :class:`UserSimulator` — a model-driven persona. Given the conversation so
  far (the agent's last reply), it produces the next user message *in
  character*, or the literal sentinel ``[DONE]`` when its goal is achieved or
  abandoned. It is itself an LLM call, which is why the simulator model is
  injectable: tests pass a ``FunctionModel``/``TestModel`` so the whole loop
  runs offline and deterministically.
- :func:`simulate` — the eval driver. It alternates simulator → agent → simulator
  until the simulator emits ``[DONE]``, a ``stop_when`` predicate fires on the
  agent's reply, or ``max_turns`` is hit. The agent is run with a stable
  ``session_id`` so it accumulates history across turns (the whole point of a
  *multi-turn* eval), and per-turn :class:`~yaab.types.Usage` is aggregated.
- :class:`SimulationEvaluator` / :func:`simulate_evalset` — wrap the driver as an
  evaluation (score = ``goal_achieved`` as 1/0, or a custom metric), and seed a
  whole suite of personas from an :class:`~yaab.governance.evalset.EvalSet` where
  each case carries ``metadata['persona']`` and ``metadata['goal']``.

WHY a separate self-assessment call: the simulator decides *whether* the goal was
met, but it signals turn-termination with ``[DONE]`` for the loop's benefit. We
ask it one final ``GOAL_ACHIEVED: yes/no`` question so the success signal is an
explicit, parseable judgement rather than something we have to infer from the
presence/absence of the sentinel (a simulator that gives up also emits ``[DONE]``).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..types import Message, Role, Usage

if TYPE_CHECKING:
    from ..models.base import ModelProvider

#: Sentinel the simulator emits to end the conversation (goal met or abandoned).
DONE_TOKEN = "[DONE]"

#: Marker embedded in the final self-assessment prompt; the simulator replies
#: ``GOAL_ACHIEVED: yes`` / ``GOAL_ACHIEVED: no``.
GOAL_MARKER = "GOAL_ACHIEVED"


class SimulationResult(BaseModel):
    """The outcome of a simulated multi-turn conversation.

    ``transcript`` is the full dialogue as ``{"role", "content"}`` dicts in
    order (user/assistant alternating); ``turns`` counts completed user→agent
    exchanges; ``goal_achieved`` is the simulator's own final self-assessment;
    ``agent_usage`` aggregates the agent's token/cost accounting across turns.
    """

    transcript: list[dict[str, str]] = Field(default_factory=list)
    turns: int = 0
    goal_achieved: bool = False
    agent_usage: Usage = Field(default_factory=Usage)


class UserSimulator:
    """An LLM playing a persona with a goal, driving a multi-turn conversation.

    The simulator is *itself* model-driven: :meth:`next_message` renders a system
    prompt (persona + goal + instructions) plus the conversation so far — but
    from the *user's* point of view, so the agent's turns are presented as the
    counterpart's messages — and asks the model for the next user utterance.
    A reply of ``[DONE]`` (or one containing it) signals the simulator is
    finished. :meth:`assess_goal` issues the final ``GOAL_ACHIEVED: yes/no``
    self-assessment.
    """

    def __init__(
        self,
        model: ModelProvider | str,
        persona: str,
        goal: str,
        max_turns: int = 8,
        stop_when: Callable[[str], bool] | None = None,
    ) -> None:
        from ..models import resolve_model

        self.model = resolve_model(model)
        self.persona = persona
        self.goal = goal
        self.max_turns = max_turns
        self.stop_when = stop_when

    # --- prompt construction ------------------------------------------------
    def _system_prompt(self) -> str:
        """Build the simulator's system prompt from persona + goal.

        WHY the explicit ``[DONE]`` instruction: the loop relies on the sentinel
        to terminate, so the contract has to be stated to the model in-band; we
        keep it terse and unambiguous so even small models comply.
        """
        return (
            "You are role-playing as a human user talking to an AI assistant. "
            f"Your persona: {self.persona}\n"
            f"Your goal in this conversation: {self.goal}\n\n"
            "Stay in character and pursue your goal across the conversation. "
            "Reply with ONLY your next message to the assistant, as the user would "
            "phrase it — do not narrate or explain. "
            f"When your goal is achieved or you have given up, reply with exactly {DONE_TOKEN} "
            "and nothing else."
        )

    def _to_simulator_view(self, transcript: list[dict[str, str]]) -> list[Message]:
        """Render the transcript from the *simulator's* perspective.

        The transcript stores roles from the agent's point of view (the human is
        ``user``, the agent is ``assistant``). For the simulator the roles flip:
        the simulator *is* the user, so its own past lines are ``assistant`` and
        the agent's replies are ``user`` input it must respond to. This framing
        is what lets a plain chat model produce the next user turn naturally.
        """
        messages: list[Message] = [Message(role=Role.SYSTEM, content=self._system_prompt())]
        for item in transcript:
            if item["role"] == "user":
                messages.append(Message(role=Role.ASSISTANT, content=item["content"]))
            else:
                messages.append(Message(role=Role.USER, content=item["content"]))
        return messages

    # --- model-driven turns -------------------------------------------------
    async def next_message(self, transcript: list[dict[str, str]]) -> tuple[str, bool]:
        """Produce the next user message, or signal completion.

        Returns ``(message, done)``. ``done`` is True when the model emits the
        ``[DONE]`` sentinel; in that case ``message`` is the empty string (the
        sentinel itself is never appended to the transcript).
        """
        messages = self._to_simulator_view(transcript)
        # On the very first turn there is no agent message to react to, so nudge
        # the model to open the conversation.
        if not transcript:
            messages.append(
                Message(role=Role.USER, content="Begin the conversation with your first message.")
            )
        resp = await self.model.complete(messages)
        text = (resp.content or "").strip()
        if DONE_TOKEN in text:
            return "", True
        return text, False

    async def assess_goal(self, transcript: list[dict[str, str]]) -> bool:
        """Ask the simulator whether its goal was achieved (final self-assessment).

        Parsed leniently: any ``yes`` in the reply (case-insensitive) counts as
        achieved, so a chatty model that says "GOAL_ACHIEVED: yes, because…"
        still scores correctly.
        """
        messages = self._to_simulator_view(transcript)
        messages.append(
            Message(
                role=Role.USER,
                content=(
                    f"The conversation is over. Did you achieve your goal ({self.goal})? "
                    f"Answer with exactly '{GOAL_MARKER}: yes' or '{GOAL_MARKER}: no'."
                ),
            )
        )
        resp = await self.model.complete(messages)
        return "yes" in (resp.content or "").lower()


async def simulate(
    agent: Any,
    simulator: UserSimulator,
    *,
    session_id: str | None = None,
) -> SimulationResult:
    """Drive a multi-turn conversation between ``simulator`` and ``agent``.

    The loop: the simulator produces a user turn → the agent answers (carrying a
    stable ``session_id`` so it sees the running history) → repeat. It ends when
    the simulator emits ``[DONE]``, the simulator's ``stop_when`` predicate fires
    on the agent's reply, or ``simulator.max_turns`` is reached. Finally the
    simulator self-assesses ``goal_achieved``.

    A ``session_id`` is *always* used (generated if not supplied) because
    multi-turn evaluation only means something if the agent accumulates history;
    without it each ``agent.run`` would be amnesiac and the eval would be a
    series of unrelated single-turn calls.
    """
    if session_id is None:
        session_id = f"sim_{uuid.uuid4().hex[:12]}"

    transcript: list[dict[str, str]] = []
    usage = Usage()
    turns = 0

    while turns < simulator.max_turns:
        user_message, done = await simulator.next_message(transcript)
        if done:
            break
        transcript.append({"role": "user", "content": user_message})

        result = await agent.run(user_message, session_id=session_id)
        reply = _result_text(result)
        transcript.append({"role": "assistant", "content": reply})
        if result.usage is not None:
            usage.add(result.usage)
        turns += 1

        if simulator.stop_when is not None and simulator.stop_when(reply):
            break

    goal_achieved = await simulator.assess_goal(transcript) if transcript else False
    return SimulationResult(
        transcript=transcript,
        turns=turns,
        goal_achieved=goal_achieved,
        agent_usage=usage,
    )


def _result_text(result: Any) -> str:
    """Coerce an agent run result's output to text for the transcript.

    The agent's ``output`` is usually a string but may be a structured object
    (typed output); we stringify so the transcript is always serializable.
    """
    output = getattr(result, "output", result)
    return output if isinstance(output, str) else str(output)


class SimulationEvaluator:
    """Wrap :func:`simulate` as an evaluation that scores a run in [0, 1].

    With no ``metric`` the score is ``goal_achieved`` mapped to 1.0/0.0 — the
    simplest useful signal: did the simulated user accomplish what it set out to
    do. Pass a ``metric`` callable (:class:`SimulationResult` → float) for richer
    scoring (turn efficiency, transcript length, an LLM judge over the
    transcript, …).
    """

    name = "user_simulation"

    def __init__(self, metric: Callable[[SimulationResult], float] | None = None) -> None:
        self.metric = metric

    async def ascore(
        self,
        agent: Any,
        simulator: UserSimulator,
        *,
        session_id: str | None = None,
    ) -> float:
        result = await simulate(agent, simulator, session_id=session_id)
        return self.score(result)

    def score(self, result: SimulationResult) -> float:
        """Score an already-computed :class:`SimulationResult`."""
        if self.metric is not None:
            return float(self.metric(result))
        return 1.0 if result.goal_achieved else 0.0


async def simulate_evalset(
    agent: Any,
    evalset: Any,
    simulator_model: ModelProvider | str,
    *,
    max_turns: int = 8,
    stop_when: Callable[[str], bool] | None = None,
) -> list[SimulationResult]:
    """Run a persona-driven simulation per case in an :class:`EvalSet`.

    Each :class:`~yaab.governance.evalset.EvalCase` seeds one simulation: the
    persona and goal come from ``case.metadata['persona']`` and
    ``case.metadata['goal']``. When a case omits them, the case's conversation is
    used as a fallback (the first user turn becomes the goal and the persona
    defaults to a generic user) so legacy single-turn cases still drive a
    sensible simulation instead of erroring.

    Every case gets its own ``session_id`` so simulations don't bleed history
    into each other. Returns one :class:`SimulationResult` per case, in order.
    """
    results: list[SimulationResult] = []
    for case in evalset.cases:
        metadata = getattr(case, "metadata", {}) or {}
        persona = metadata.get("persona") or "A typical user"
        goal = metadata.get("goal")
        if not goal:
            conversation = getattr(case, "conversation", None) or []
            goal = conversation[0] if conversation else "Have a helpful conversation"
        simulator = UserSimulator(
            model=simulator_model,
            persona=persona,
            goal=goal,
            max_turns=max_turns,
            stop_when=stop_when,
        )
        session_id = f"sim_{getattr(case, 'id', uuid.uuid4().hex[:8])}_{uuid.uuid4().hex[:8]}"
        results.append(await simulate(agent, simulator, session_id=session_id))
    return results

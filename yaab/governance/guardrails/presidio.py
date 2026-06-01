"""Microsoft Presidio adapter — NER-based PII detection + redaction.

Presidio uses spaCy/transformer NER plus recognizers to find PII far more
robustly than regex, then anonymizes it. This adapter exposes that as a YAAB
:class:`~yaab.governance.policy.GuardrailScanner` that ``REDACT``s on detection.

The ``presidio-analyzer`` / ``presidio-anonymizer`` packages are an optional
extra (``pip install 'yaab-sdk[presidio]'``), imported lazily; inject ``analyzer`` /
``anonymizer`` to test or customize.
"""

from __future__ import annotations

from typing import Any

from ..policy import Action, GuardrailResult, Stage


class PresidioPIIScanner:
    """Detect & redact PII via Microsoft Presidio."""

    name = "presidio"
    stages: tuple[Stage, ...] = (Stage.INPUT, Stage.OUTPUT)

    def __init__(
        self,
        *,
        analyzer: Any | None = None,
        anonymizer: Any | None = None,
        language: str = "en",
        entities: list[str] | None = None,
        action: Action = Action.REDACT,
    ) -> None:
        self._analyzer: Any = analyzer
        self._anonymizer: Any = anonymizer
        self.language = language
        self.entities = entities
        self.action = action

    def _ensure_engines(self) -> None:
        if self._analyzer is not None and self._anonymizer is not None:
            return
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "PresidioPIIScanner needs Microsoft Presidio. "
                "Install it with `pip install 'yaab-sdk[presidio]'` (and a spaCy model, "
                "e.g. `python -m spacy download en_core_web_lg`)."
            ) from exc
        if self._analyzer is None:
            self._analyzer = AnalyzerEngine()
        if self._anonymizer is None:
            self._anonymizer = AnonymizerEngine()

    def scan(self, text: str, stage: Stage) -> GuardrailResult:
        self._ensure_engines()
        results = self._analyzer.analyze(text=text, language=self.language, entities=self.entities)
        if not results:
            return GuardrailResult(action=Action.ALLOW, scanner=self.name, text=text)
        anonymized = self._anonymizer.anonymize(text=text, analyzer_results=results)
        found = sorted({getattr(r, "entity_type", "PII") for r in results})
        return GuardrailResult(
            action=self.action,
            scanner=self.name,
            reason=f"PII detected: {', '.join(found)}",
            text=getattr(anonymized, "text", text),
        )


__all__ = ["PresidioPIIScanner"]

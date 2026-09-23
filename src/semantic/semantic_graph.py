"""
semantic/semantic_graph.py
--------------------------

Convert the output of SemanticExtractor into a cleaner
semantic graph.

This layer does NOT select the final target.

Architecture:

    Sentence
        ↓
    SemanticExtractor
        ↓
    SemanticRepresentation
        ↓
    SemanticGraph
        ↓
    TargetSelector


The graph separates:

    - entities
    - events
    - subjects
    - objects
    - contextual noun phrases

The goal is to give target_selector.py a structured
representation that is easier to reason about than raw
spaCy dependency information.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from semantic.semantic_extractor import (
    Entity,
    Event,
    NounPhrase,
    Relation,
    SemanticRepresentation,
    TokenInfo,
)


# ============================================================
# Graph structures
# ============================================================


@dataclass
class GraphEntity:
    """Entity inside the semantic graph."""

    text: str
    label: str
    start: int
    end: int


@dataclass
class GraphEvent:
    """Event with its semantic participants."""

    trigger: str
    lemma: str
    start: int
    end: int

    subjects: list[str]
    objects: list[str]


@dataclass
class SemanticGraph:
    """Structured semantic graph for one sentence."""

    sentence: str

    entities: list[GraphEntity]

    events: list[GraphEvent]

    noun_phrases: list[str]

    subjects: list[str]

    objects: list[str]

    tokens: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert graph into a JSON-friendly dictionary."""

        return asdict(self)


# ============================================================
# Semantic graph builder
# ============================================================


class SemanticGraphBuilder:
    """
    Build a SemanticGraph from SemanticRepresentation.

    This class deliberately does not decide which target
    is important.

    It only organizes semantic information.
    """

    # ========================================================
    # Public API
    # ========================================================

    def build(
        self,
        representation: SemanticRepresentation,
        
    ) -> SemanticGraph:
        """
        Convert a SemanticRepresentation into a SemanticGraph.
        """

        if not isinstance(
            representation,
            SemanticRepresentation,
        ):
            raise TypeError(
                "representation must be a "
                "SemanticRepresentation"
            )

        entities = self._build_entities(
            representation.entities
        )

        events = self._build_events(
            representation
        )

        tokens = representation.tokens
        noun_phrases = [
            phrase.text
            for phrase in representation.noun_phrases
        ]

        subjects = self._collect_subjects(
            events
        )

        objects = self._collect_objects(
            events
        )

        return SemanticGraph(
            sentence=representation.sentence,
            entities=entities,
            events=events,
            noun_phrases=noun_phrases,
            subjects=subjects,
            objects=objects,
            tokens=tokens,
        )

    # ========================================================
    # Entity conversion
    # ========================================================

    @staticmethod
    def _build_entities(
        entities: list[Entity],
    ) -> list[GraphEntity]:

        return [
            GraphEntity(
                text=entity.text,
                label=entity.label,
                start=entity.start,
                end=entity.end,
            )
            for entity in entities
        ]

    # ========================================================
    # Event conversion
    # ========================================================

    def _build_events(
        self,
        representation: SemanticRepresentation,
    ) -> list[GraphEvent]:

        graph_events = []

        for event in representation.events:

            subjects = []
            objects = []

            for relation in representation.relations:

                if not self._relation_matches_event(
                    relation,
                    event,
                ):
                    continue

                if relation.subject:
                    subjects.append(
                        relation.subject
                    )

                if relation.object:
                    objects.append(
                        relation.object
                    )

            graph_events.append(
                GraphEvent(
                    trigger=event.trigger,
                    lemma=event.lemma,
                    start=event.start,
                    end=event.end,
                    subjects=self._unique(
                        subjects
                    ),
                    objects=self._unique(
                        objects
                    ),
                )
            )

        return graph_events

    # ========================================================
    # Relation matching
    # ========================================================

    @staticmethod
    def _relation_matches_event(
        relation: Relation,
        event: Event,
    ) -> bool:
        """
        A relation belongs to an event only if it came from the
        SAME verb token in the sentence (matched by character
        offset), not just a matching lemma.

        This matters for sentences with repeated verbs, e.g.:

            "Google announced layoffs and Meta announced hiring."

        Both verbs share the lemma "announce", so lemma-based
        matching would incorrectly merge Google/layoffs and
        Meta/hiring into both events. Positional matching keeps
        them separate.
        """

        return relation.event_start == event.start

    # ========================================================
    # Subject collection
    # ========================================================

    @staticmethod
    def _collect_subjects(
        events: list[GraphEvent],
    ) -> list[str]:

        subjects = []

        for event in events:
            subjects.extend(
                event.subjects
            )

        return SemanticGraphBuilder._unique(
            subjects
        )

    # ========================================================
    # Object collection
    # ========================================================

    @staticmethod
    def _collect_objects(
        events: list[GraphEvent],
    ) -> list[str]:

        objects = []

        for event in events:
            objects.extend(
                event.objects
            )

        return SemanticGraphBuilder._unique(
            objects
        )

    # ========================================================
    # Utility
    # ========================================================

    @staticmethod
    def _unique(
        values: list[str],
    ) -> list[str]:

        result = []

        seen = set()

        for value in values:

            if not value:
                continue

            normalized = value.strip()

            if not normalized:
                continue

            key = normalized.lower()

            if key in seen:
                continue

            seen.add(key)

            result.append(
                normalized
            )

        return result


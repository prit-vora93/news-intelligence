"""
semantic/entity_resolver.py
---------------------------

Rule-based entity resolver V2 for the semantic pipeline.

Responsibilities:
    1. Resolve within-sentence pronouns.
    2. Resolve within-sentence definite descriptions.
    3. Resolve pronouns/definite descriptions that refer to an
       entity in the IMMEDIATELY PRECEDING sentence (one sentence
       back only -- a narrow, deliberately conservative first step
       toward cross-sentence coreference, not full document-level
       resolution).
    4. Use character/token position when available.
    5. Use grammatical role and entity type when scoring candidates.
    6. Preserve resolution evidence for debugging.
    7. Keep unresolved mentions instead of silently dropping them.

V2 is still heuristic.

Cross-sentence resolution is intentionally narrow: only looks
ONE sentence back, and (since the previous sentence's semantic
graph isn't available at the point of scoring) does NOT use
grammatical role, same-event, or character-distance bonuses for
cross-sentence candidates -- only a flat base score plus entity-type
compatibility, the same check used for same-sentence candidates.
This means cross-sentence resolution only succeeds when the
mention's type hint clearly matches the candidate's entity type
(e.g. "he" -> a PERSON), a deliberately high bar. Full
document-level (multi-sentence-back) coreference remains a
separate, larger undertaking -- see entity_resolver_scope_v1.md
Section 9.

Architecture:

    SemanticGraph
        +
    TargetSelection
        ↓
    EntityResolver
        ↓
    ResolutionResult

Important:
    EntityResolver does NOT decide which targets are important.
    That remains TargetSelector's responsibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from semantic.semantic_graph import SemanticGraph
from semantic.target_selector import TargetSelection, TargetSelector


# ============================================================
# Type compatibility
# ============================================================

MALE_PRONOUNS: set[str] = {"he", "him", "his"}
FEMALE_PRONOUNS: set[str] = {"she", "her", "hers"}


PRONOUN_TYPE_HINTS: dict[str, set[str]] = {
    "it": {
        "ORG",
        "GPE",
        "PRODUCT",
        "LAW",
        "NORP",
        "FAC",
        "MONEY",
    },
    "its": {
        "ORG",
        "GPE",
        "PRODUCT",
        "LAW",
        "NORP",
        "FAC",
        "MONEY",
    },
    "he": {"PERSON"},
    "him": {"PERSON"},
    "his": {"PERSON"},
    "she": {"PERSON"},
    "her": {"PERSON"},
}


GENERIC_HEAD_NOUN_TYPE_HINTS: dict[str, set[str]] = {
    "company": {"ORG"},
    "companies": {"ORG"},
    "firm": {"ORG"},
    "firms": {"ORG"},
    "organization": {"ORG"},
    "organizations": {"ORG"},
    "government": {"ORG", "GPE"},
    "country": {"GPE"},
    "countries": {"GPE"},
    "nation": {"GPE"},
    "nations": {"GPE"},
}


# ============================================================
# Mention
# ============================================================


@dataclass
class Mention:
    """
    One occurrence of a possible real-world entity.

    start/end are character offsets in the original sentence.

    A value of -1 means the position is unavailable.
    """

    text: str
    sentence_index: int
    mention_type: str
    start: int = -1
    end: int = -1
    target_reference: TargetSelection | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "sentence_index": self.sentence_index,
            "mention_type": self.mention_type,
            "start": self.start,
            "end": self.end,
            "target_reference": (
                self.target_reference.target
                if self.target_reference
                else None
            ),
        }


# ============================================================
# Resolved entity
# ============================================================


@dataclass
class ResolvedEntity:
    """
    A group of mentions believed to refer to the same
    real-world entity.
    """

    canonical_id: str
    canonical_name: str
    entity_type: str | None
    mentions: list[Mention] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_id": self.canonical_id,
            "canonical_name": self.canonical_name,
            "entity_type": self.entity_type,
            "mentions": [
                mention.to_dict()
                for mention in self.mentions
            ],
        }


# ============================================================
# Resolution result
# ============================================================


@dataclass
class ResolutionResult:
    """
    Final output of EntityResolver.
    """

    resolved_entities: list[ResolvedEntity] = field(
        default_factory=list
    )

    unresolved_mentions: list[Mention] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolved_entities": [
                entity.to_dict()
                for entity in self.resolved_entities
            ],
            "unresolved_mentions": [
                mention.to_dict()
                for mention in self.unresolved_mentions
            ],
        }


# ============================================================
# Internal candidate
# ============================================================


@dataclass
class _Candidate:
    """
    Internal candidate antecedent.

    Candidates are generated first and scored separately.
    """

    text: str
    label: str | None
    role: str
    event_index: int
    start: int = -1
    end: int = -1
    cross_sentence: bool = False
    established_gender: str | None = None


# ============================================================
# Entity Resolver
# ============================================================


class EntityResolver:
    """
    Rule-based entity resolver V2.

    V2 improvements over the initial implementation:

        - character-position tracking
        - backward-only candidate filtering
        - grammatical-role scoring
        - entity-type compatibility
        - same-event subject support
        - embedded pronoun detection
        - embedded definite-description detection
        - conservative unresolved behavior
    """

    def __init__(self) -> None:
        self._selector = TargetSelector()

    # ========================================================
    # Public API
    # ========================================================

    def resolve(
        self,
        sentence_graphs: list[SemanticGraph],
        target_lists: list[list[TargetSelection]],
    ) -> ResolutionResult:
        """
        Resolve references in the supplied sentence graphs.

        V2 remains within-sentence only.

        The list-based API is retained so cross-sentence
        resolution can be added later without changing the
        public data shape.
        """

        if not isinstance(sentence_graphs, list):
            raise TypeError(
                "sentence_graphs must be a list"
            )

        if not isinstance(target_lists, list):
            raise TypeError(
                "target_lists must be a list"
            )

        if len(sentence_graphs) != len(target_lists):
            raise ValueError(
                "sentence_graphs and target_lists must be "
                "the same length"
            )

        result = ResolutionResult()

        # canonical_id -> entity cluster
        clusters: dict[str, ResolvedEntity] = {}

        for sentence_index, (
            graph,
            targets,
        ) in enumerate(
            zip(sentence_graphs, target_lists)
        ):
            if not isinstance(
                graph,
                SemanticGraph,
            ):
                raise TypeError(
                    "Every item in sentence_graphs must be "
                    "a SemanticGraph"
                )

            self._resolve_one_sentence(
                graph=graph,
                targets=targets,
                sentence_index=sentence_index,
                clusters=clusters,
                result=result,
            )

        result.resolved_entities = list(
            clusters.values()
        )

        return result

    @staticmethod
    def _find_established_type(
        entity: ResolvedEntity,
        clusters: dict[str, ResolvedEntity],
    ) -> str | None:
        """
        Find an already-established, more reliable type for this
        entity from earlier in the article.

        Tries an EXACT canonical_id match first. If that misses,
        falls back to a SURNAME match: does this entity's text match
        the LAST WORD of an already-established, longer entity name?

        The surname fallback is necessary because journalism
        style commonly introduces a person with their full name
        first ("JD Vance") and refers to them by surname alone
        afterward ("Vance") -- spaCy recognizes these as two
        DIFFERENT entity spans with different canonical_ids, so an
        exact-match-only lookup would miss the established type
        entirely. Confirmed via real data: "JD Vance" was correctly
        tagged PERSON on first mention, but subsequent bare "Vance"
        mentions were inconsistently mistagged ORG by spaCy, and an
        exact-match-only reconciliation failed to catch this because
        "jd-vance" and "vance" are different canonical_ids.

        Only matches a SINGLE-WORD entity against a longer
        established name (the common "surname later" pattern), not
        the reverse -- conservative, matching the real observed
        pattern rather than guessing more broadly.
        """

        established = clusters.get(entity.canonical_id)
        if established is not None and established.entity_type is not None:
            return established.entity_type

        entity_words = entity.canonical_name.split()

        if len(entity_words) != 1:
            return None

        surname = entity_words[0].lower()

        for other_entity in clusters.values():

            other_words = other_entity.canonical_name.split()

            if (
                len(other_words) > 1
                and other_words[-1].lower() == surname
                and other_entity.entity_type is not None
            ):
                return other_entity.entity_type

        return None

    # ========================================================
    # Sentence processing
    # ========================================================

    def _resolve_one_sentence(
        self,
        graph: SemanticGraph,
        targets: list[TargetSelection],
        sentence_index: int,
        clusters: dict[str, ResolvedEntity],
        result: ResolutionResult,
    ) -> None:
        """
        Process one sentence.

        Named entities are created first.

        Pronouns and definite descriptions are then discovered
        and resolved against those named entities.
        """

        sentence_entities = (
            self._build_named_entities(
                graph,
                sentence_index,
            )
        )

        # Reconcile against any type ALREADY ESTABLISHED for this
        # same entity earlier in the article, before these entities
        # are used as candidates. Confirmed necessary via real data:
        # spaCy can inconsistently tag the SAME real entity across
        # different sentences of one article (e.g. "Vance" tagged
        # PERSON in one sentence, ORG in another, purely due to NER
        # unreliability, not a real type change). Without this, a
        # flaky local mistag could pass a type-compatibility check
        # that the entity's more reliable, already-established type
        # would have correctly rejected -- confirmed via "the
        # government" (needs ORG/GPE) incorrectly matching "Vance"
        # only because THAT sentence's copy of Vance happened to be
        # mistagged ORG, even though PERSON had already been
        # established from two earlier, correctly-tagged mentions.
        for entity in sentence_entities:

            established_type = self._find_established_type(
                entity, clusters
            )

            if established_type is not None:
                entity.entity_type = established_type

        # Map local entity ID -> entity object.
        local_entities: dict[
            str,
            ResolvedEntity,
        ] = {}

        for entity in sentence_entities:
            local_entities[
                entity.canonical_id
            ] = entity

        # ----------------------------------------------------
        # Resolve references.
        #
        # Candidate pool = this sentence's entities PLUS any
        # entity from the IMMEDIATELY PRECEDING sentence (one
        # sentence back only, deliberately narrow for a first
        # cross-sentence pass -- see module docstring). `clusters`
        # at this point only contains entities from sentences
        # BEFORE this one (merging into it happens later, below),
        # so this is safe -- no risk of a sentence's own entities
        # duplicating themselves here.
        # ----------------------------------------------------

        candidate_pool = list(sentence_entities)

        previous_sentence_index = sentence_index - 1

        if previous_sentence_index >= 0:

            for entity in clusters.values():

                if any(
                    m.sentence_index == previous_sentence_index
                    for m in entity.mentions
                ):
                    candidate_pool.append(entity)

        references = self._find_references(
            graph=graph,
            targets=targets,
            sentence_index=sentence_index,
        )

        for mention in references:

            antecedent = self._resolve_one_mention(
                mention=mention,
                graph=graph,
                targets=targets,
                resolved_entities=candidate_pool,
            )

            if antecedent is None:
                result.unresolved_mentions.append(
                    mention
                )
                continue

            antecedent.mentions.append(
                mention
            )

        # ----------------------------------------------------
        # Merge sentence entities into global clusters.
        # ----------------------------------------------------

        for entity in sentence_entities:

            key = entity.canonical_id

            if key not in clusters:
                clusters[key] = entity
                continue

            existing = clusters[key]

            existing.mentions.extend(
                entity.mentions
            )

            # Prefer a known entity type.
            if (
                existing.entity_type is None
                and entity.entity_type is not None
            ):
                existing.entity_type = (
                    entity.entity_type
                )

    # ========================================================
    # Named entities
    # ========================================================

    def _build_named_entities(
        self,
        graph: SemanticGraph,
        sentence_index: int,
    ) -> list[ResolvedEntity]:
        """
        Convert graph entities into initial entity clusters.

        Character offsets are preserved from GraphEntity.
        """

        results: list[ResolvedEntity] = []

        seen: set[str] = set()

        for entity in graph.entities:

            text = self._normalize_text(
                entity.text
            )

            if not text:
                continue

            if (
                entity.label
                in self._selector.WEAK_ENTITY_LABELS
            ):
                continue

            key = text.lower()

            if key in seen:
                continue

            seen.add(key)

            mention = Mention(
                text=text,
                sentence_index=sentence_index,
                mention_type="named_entity",
                start=entity.start,
                end=entity.end,
            )

            results.append(
                ResolvedEntity(
                    canonical_id=self._canonical_id(
                        text
                    ),
                    canonical_name=text,
                    entity_type=entity.label,
                    mentions=[mention],
                )
            )

        return results

    # ========================================================
    # Reference discovery
    # ========================================================

    def _find_references(
        self,
        graph: SemanticGraph,
        targets: list[TargetSelection],
        sentence_index: int,
    ) -> list[Mention]:
        """
        Find pronouns and definite descriptions.

        We inspect:

            1. event subjects
            2. event objects
            3. embedded pronouns inside targets
            4. embedded definite descriptions inside targets
            5. standalone target references
        """

        references: list[Mention] = []

        seen: set[tuple[int, str, str]] = set()

        # ----------------------------------------------------
        # Event participants.
        # ----------------------------------------------------

        for event in graph.events:

            values = list(event.subjects)
            values.extend(event.objects)

            for value in values:

                text = self._normalize_text(
                    value
                )

                if not text:
                    continue

                mention_type = (
                    self._mention_type(text)
                )

                if mention_type is not None:

                    start, end = (
                        self._locate_text(
                            graph.sentence,
                            text,
                        )
                    )

                    key = (
                        sentence_index,
                        mention_type,
                        text.lower(),
                    )

                    if key not in seen:
                        seen.add(key)

                        references.append(
                            Mention(
                                text=text,
                                sentence_index=sentence_index,
                                mention_type=mention_type,
                                start=start,
                                end=end,
                            )
                        )

                    continue

                # ------------------------------------------------
                # Embedded references.
                # ------------------------------------------------

                for surface in (
                    self._find_embedded_pronouns(
                        text
                    )
                ):

                    start, end = (
                        self._locate_text(
                            graph.sentence,
                            surface,
                        )
                    )

                    key = (
                        sentence_index,
                        "pronoun",
                        surface.lower(),
                    )

                    if key in seen:
                        continue

                    seen.add(key)

                    references.append(
                        Mention(
                            text=surface,
                            sentence_index=sentence_index,
                            mention_type="pronoun",
                            start=start,
                            end=end,
                        )
                    )

                for surface, normalized in (
                    self._find_embedded_generic_descriptions(
                        text
                    )
                ):

                    start, end = (
                        self._locate_text(
                            graph.sentence,
                            surface,
                        )
                    )

                    key = (
                        sentence_index,
                        "definite_description",
                        normalized.lower(),
                    )

                    if key in seen:
                        continue

                    seen.add(key)

                    references.append(
                        Mention(
                            text=surface,
                            sentence_index=sentence_index,
                            mention_type=(
                                "definite_description"
                            ),
                            start=start,
                            end=end,
                        )
                    )

        # ----------------------------------------------------
        # Targets can contain references not present directly
        # in event participants.
        # ----------------------------------------------------

        for target in targets:

            if target.target is None:
                continue

            text = self._normalize_text(
                target.target
            )

            if not text:
                continue

            mention_type = (
                self._mention_type(text)
            )

            if mention_type is not None:

                start, end = (
                    self._locate_text(
                        graph.sentence,
                        text,
                    )
                )

                key = (
                    sentence_index,
                    mention_type,
                    text.lower(),
                )

                if key not in seen:
                    seen.add(key)

                    references.append(
                        Mention(
                            text=text,
                            sentence_index=sentence_index,
                            mention_type=mention_type,
                            start=start,
                            end=end,
                            target_reference=target,
                        )
                    )

                continue

            # ------------------------------------------------
            # Embedded references inside the target.
            # ------------------------------------------------

            for surface in (
                self._find_embedded_pronouns(text)
            ):

                start, end = (
                    self._locate_text(
                        graph.sentence,
                        surface,
                    )
                )

                key = (
                    sentence_index,
                    "pronoun",
                    surface.lower(),
                )

                if key in seen:
                    continue

                seen.add(key)

                references.append(
                    Mention(
                        text=surface,
                        sentence_index=sentence_index,
                        mention_type="pronoun",
                        start=start,
                        end=end,
                        target_reference=target,
                    )
                )

            for surface, normalized in (
                self._find_embedded_generic_descriptions(
                    text
                )
            ):

                start, end = (
                    self._locate_text(
                        graph.sentence,
                        surface,
                    )
                )

                key = (
                    sentence_index,
                    "definite_description",
                    normalized.lower(),
                )

                if key in seen:
                    continue

                seen.add(key)

                references.append(
                    Mention(
                        text=surface,
                        sentence_index=sentence_index,
                        mention_type=(
                            "definite_description"
                        ),
                        start=start,
                        end=end,
                        target_reference=target,
                    )
                )

        # Sort by actual sentence position.
        references.sort(
            key=lambda mention: (
                mention.start
                if mention.start >= 0
                else float("inf")
            )
        )

        return references

    # ========================================================
    # Resolve one mention
    # ========================================================

    def _resolve_one_mention(
        self,
        mention: Mention,
        graph: SemanticGraph,
        targets: list[TargetSelection],
        resolved_entities: list[ResolvedEntity],
    ) -> ResolvedEntity | None:
        """
        Resolve one mention.

        Steps:

            1. gather candidates
            2. score candidates
            3. reject weak/ambiguous matches
            4. return winner
        """

        candidates = self._gather_candidates(
            mention=mention,
            resolved_entities=resolved_entities,
            graph=graph,
            targets=targets,
        )

        if not candidates:
            return None

        scored: list[
            tuple[int, _Candidate, ResolvedEntity]
        ] = []

        for candidate in candidates:

            entity = self._find_entity(
                candidate.text,
                resolved_entities,
            )

            if entity is None:
                continue

            mention_event_index = (
                self._find_mention_event(
                    mention,
                    graph,
                )
            )

            score = self._score_candidate(
                mention=mention,
                candidate=candidate,
                mention_event_index=mention_event_index,
            )

            if score is None:
                continue

            scored.append(
                (
                    score,
                    candidate,
                    entity,
                )
            )

        if not scored:
            return None

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        best_score, _, best_entity = scored[0]

        # ----------------------------------------------------
        # Ambiguity protection.
        #
        # If two candidates are effectively tied, don't guess.
        # ----------------------------------------------------

        if len(scored) > 1:

            second_score = scored[1][0]

            if best_score - second_score < 10:
                return None

        # Minimum confidence threshold.
        if best_score < 40:
            return None

        return best_entity

    @staticmethod
    def _established_gender(
        entity: ResolvedEntity,
    ) -> str | None:
        """
        Determine an entity's established gender from its own
        mention history, by scanning for any prior male or female
        pronoun mention (he/him/his vs she/her/hers).

        Returns "male", "female", or None (no gendered pronoun seen
        yet, or conflicting evidence -- stays conservative rather
        than picking one).

        Confirmed necessary via real data: without this, a pronoun
        like "his" could resolve to a PERSON entity already
        established as female by an earlier "she" mention, purely
        because no OTHER type-compatible candidate existed -- the
        resolver was checking broad NER type (PERSON) but never
        checking gender agreement at all.
        """

        saw_male = False
        saw_female = False

        for mention in entity.mentions:

            text = mention.text.strip().lower()

            if text in MALE_PRONOUNS:
                saw_male = True
            elif text in FEMALE_PRONOUNS:
                saw_female = True

        if saw_male and not saw_female:
            return "male"

        if saw_female and not saw_male:
            return "female"

        return None

    # ========================================================
    # Candidate gathering
    # ========================================================

    def _gather_candidates(
        self,
        mention: Mention,
        resolved_entities: list[ResolvedEntity],
        graph: SemanticGraph,
        targets: list[TargetSelection],
    ) -> list[_Candidate]:
        """
        Gather plausible antecedent candidates.

        V2 position rule:

            candidate.start < mention.start

        when both positions are known.

        This prevents a mention from resolving forward to an
        entity that appears later in the sentence.

        Same-event subject resolution is allowed for possessives
        such as:

            Elon Musk sold his stake.

        where "his" refers to the subject of the same event.
        """

        candidates: list[_Candidate] = []

        mention_start = mention.start

        # ----------------------------------------------------
        # Entity mentions.
        # ----------------------------------------------------

        for entity in resolved_entities:

            if not entity.mentions:
                continue

            if (
                entity.canonical_name.strip().lower()
                == mention.text.strip().lower()
            ):
                continue

            # Prefer a mention from THIS sentence (existing,
            # unchanged within-sentence path). Only fall back to a
            # mention from the IMMEDIATELY PRECEDING sentence if no
            # same-sentence mention exists for this entity.
            same_sentence_mention = next(
                (
                    m for m in entity.mentions
                    if m.sentence_index == mention.sentence_index
                ),
                None,
            )

            if same_sentence_mention is not None:

                entity_mention = same_sentence_mention

                entity_start = entity_mention.start
                entity_end = entity_mention.end

                # ------------------------------------------------
                # Position filtering (same-sentence only -- character
                # offsets are only comparable within one sentence).
                # ------------------------------------------------

                if (
                    mention_start >= 0
                    and entity_start >= 0
                    and entity_start >= mention_start
                ):
                    continue

                role = self._infer_entity_role(
                    entity.canonical_name,
                    graph,
                )

                candidates.append(
                    _Candidate(
                        text=entity.canonical_name,
                        label=entity.entity_type,
                        role=role,
                        event_index=self._find_entity_event(
                            entity.canonical_name,
                            graph,
                        ),
                        start=entity_start,
                        end=entity_end,
                        cross_sentence=False,
                        established_gender=self._established_gender(entity),
                    )
                )

                continue

            # ------------------------------------------------
            # No same-sentence mention -- try the immediately
            # preceding sentence (one sentence back only). We
            # don't have that sentence's graph here, so role/event
            # inference and character-distance comparisons aren't
            # attempted -- _score_candidate applies a flat, more
            # conservative score for these instead (see there).
            # ------------------------------------------------

            previous_sentence_mention = next(
                (
                    m for m in entity.mentions
                    if m.sentence_index == mention.sentence_index - 1
                ),
                None,
            )

            if previous_sentence_mention is None:
                continue

            candidates.append(
                _Candidate(
                    text=entity.canonical_name,
                    label=entity.entity_type,
                    role="unknown",
                    event_index=-1,
                    start=-1,
                    end=-1,
                    cross_sentence=True,
                    established_gender=self._established_gender(entity),
                )
            )

        # ----------------------------------------------------
        # Deduplicate.
        # ----------------------------------------------------

        unique: dict[
            tuple[str, str, int],
            _Candidate,
        ] = {}

        for candidate in candidates:

            key = (
                candidate.text.lower(),
                candidate.role,
                candidate.event_index,
            )

            unique[key] = candidate

        return list(unique.values())

    # ========================================================
    # Candidate scoring
    # ========================================================

    def _score_candidate(
        self,
        mention: Mention,
        candidate: _Candidate,
        mention_event_index: int = -1,
    ) -> int | None:
        """
        Score a candidate antecedent.

        Higher is better.

        Scoring:

            subject role        +30
            object role         +10

            same event           +25

            closer mention       +0..20

            type compatible      +30

            unknown type         +5

        Hard rejection:

            incompatible known entity type
        """

        score = 0

        # ----------------------------------------------------
        # Gender agreement (applies to BOTH same-sentence and
        # cross-sentence candidates -- checked first, before any
        # other scoring path, since a gender conflict is a hard
        # grammatical impossibility, not a matter of degree).
        # ----------------------------------------------------

        mention_text_lower = mention.text.strip().lower()

        mention_gender: str | None = None
        if mention_text_lower in MALE_PRONOUNS:
            mention_gender = "male"
        elif mention_text_lower in FEMALE_PRONOUNS:
            mention_gender = "female"

        if (
            mention_gender is not None
            and candidate.established_gender is not None
            and mention_gender != candidate.established_gender
        ):
            return None

        # ----------------------------------------------------
        # Cross-sentence candidates: deliberately conservative.
        # No role bonus (would need the PREVIOUS sentence's graph,
        # not available here), no same-event bonus (doesn't apply
        # across sentences), no character-distance bonus (offsets
        # aren't comparable across different sentences). Just a
        # flat base score plus the SAME type-compatibility check
        # used for same-sentence candidates -- meaning cross-
        # sentence resolution only succeeds when the mention's type
        # hint clearly matches the candidate's entity type (e.g.
        # "he" -> a PERSON), a deliberately high bar consistent with
        # "leave unresolved rather than guess wrong".
        # ----------------------------------------------------

        if candidate.cross_sentence:

            score = 15

            compatible_types = self._compatible_entity_types(mention)

            if compatible_types:

                if candidate.label is None:
                    score += 5
                elif candidate.label in compatible_types:
                    score += 30
                else:
                    return None
            else:
                score += 5

            return score

        # ----------------------------------------------------
        # Grammatical role
        # ----------------------------------------------------

        if candidate.role == "subject":
            score += 30

        elif candidate.role == "object":
            score += 10


        # ----------------------------------------------------
        # Entity type compatibility
        # ----------------------------------------------------

        compatible_types = (
            self._compatible_entity_types(
                mention
            )
        )

        if compatible_types:

            if candidate.label is None:
                score += 5

            elif candidate.label in compatible_types:
                score += 30

            else:
                # Known incompatible entity type.
                # Never guess.
                return None

        else:
            score += 5


        # ----------------------------------------------------
        # Same-event bonus
        # ----------------------------------------------------

        if (
            mention_event_index >= 0
            and candidate.event_index >= 0
        ):

            if (
                mention_event_index
                == candidate.event_index
            ):
                score += 25


        # ----------------------------------------------------
        # Character distance
        # ----------------------------------------------------

        if (
            mention.start >= 0
            and candidate.start >= 0
        ):

            distance = (
                mention.start
                - candidate.start
            )

            if distance < 0:
                return None

            distance_bonus = max(
                0,
                20 - min(
                    distance // 10,
                    20,
                ),
            )

            score += distance_bonus


        return score

    # ========================================================
    # Entity lookup
    # ========================================================

    @staticmethod
    def _find_entity(
        text: str,
        entities: list[ResolvedEntity],
    ) -> ResolvedEntity | None:

        normalized = text.strip().lower()

        for entity in entities:

            if (
                entity.canonical_name
                .strip()
                .lower()
                == normalized
            ):
                return entity

        return None

    # ========================================================
    # Entity role inference
    # ========================================================

    @staticmethod
    def _infer_entity_role(
        entity_text: str,
        graph: SemanticGraph,
    ) -> str:
        """
        Infer whether an entity behaves primarily as a subject
        or object in the graph.

        Subject wins if the entity appears in both roles.
        """

        normalized = (
            entity_text.strip().lower()
        )

        subject = False
        object_ = False

        for event in graph.events:

            for value in event.subjects:

                if (
                    value.strip().lower()
                    == normalized
                ):
                    subject = True

            for value in event.objects:

                if (
                    value.strip().lower()
                    == normalized
                ):
                    object_ = True

        if subject:
            return "subject"

        if object_:
            return "object"

        return "object"

    # ========================================================
    # Event lookup
    # ========================================================

    @staticmethod
    def _find_entity_event(
        entity_text: str,
        graph: SemanticGraph,
    ) -> int:
        """
        Return the first event containing the entity.

        -1 means no direct event participant was found.
        """

        normalized = (
            entity_text.strip().lower()
        )

        for index, event in enumerate(
            graph.events
        ):

            values = (
                list(event.subjects)
                + list(event.objects)
            )

            for value in values:

                if (
                    value.strip().lower()
                    == normalized
                ):
                    return index

        return -1
    
    def _find_mention_event(
        self,
        mention: Mention,
        graph: SemanticGraph,
    ) -> int:
        """
        Find the event closest to the mention.

        Uses character offsets.

        Returns -1 when no event can be associated.
        """

        if mention.start < 0:
            return -1

        best_index = -1
        best_distance = float("inf")

        for index, event in enumerate(
            graph.events
        ):

            event_start = event.start
            event_end = event.end

            # Mention is inside the event span.
            if (
                event_start >= 0
                and event_end >= 0
                and event_start
                <= mention.start
                <= event_end
            ):
                return index

            # Otherwise find the closest event.
            if event_start >= 0:

                distance = abs(
                    mention.start
                    - event_start
                )

                if distance < best_distance:
                    best_distance = distance
                    best_index = index

        return best_index

    # ========================================================
    # Type compatibility
    # ========================================================

    def _compatible_entity_types(
        self,
        mention: Mention,
    ) -> set[str]:
        """
        Return allowed entity types for a mention.

        Empty set means no reliable type constraint.
        """

        text = (
            mention.text
            .strip()
            .lower()
        )

        if mention.mention_type == "pronoun":

            return PRONOUN_TYPE_HINTS.get(
                text,
                set(),
            )

        if (
            mention.mention_type
            == "definite_description"
        ):

            head = (
                self._generic_head_noun(
                    text
                )
            )

            return GENERIC_HEAD_NOUN_TYPE_HINTS.get(
                head,
                set(),
            )

        return set()

    # ========================================================
    # Mention classification
    # ========================================================

    def _mention_type(
        self,
        text: str,
    ) -> str | None:
        """
        Determine whether text is a resolvable reference.
        """

        normalized = (
            self._normalize_text(text)
            .lower()
        )

        if normalized in self._selector.PRONOUNS:
            return "pronoun"

        if (
            # normalized
            # in self._selector.GENERIC_TERMS
            normalized in self._selector.GENERIC_TERMS
            and normalized.startswith("the ")
        ):
            return "definite_description"

        return None

    # ========================================================
    # Embedded pronouns
    # ========================================================

    def _find_embedded_pronouns(
        self,
        text: str,
    ) -> list[str]:
        """
        Find pronouns embedded inside a larger noun phrase.

        Examples:

            "his stake in the company"
                -> ["his"]

            "its subscription prices"
                -> ["its"]
        """

        words = text.split()

        found: list[str] = []

        for word in words:

            cleaned = (
                word.strip(
                    ".,!?;:\"'()[]{}"
                )
            )

            normalized = cleaned.lower()

            if (
                normalized
                in self._selector.PRONOUNS
            ):
                found.append(cleaned)

        return found

    # ========================================================
    # Embedded definite descriptions
    # ========================================================

    def _find_embedded_generic_descriptions(
        self,
        text: str,
    ) -> list[tuple[str, str]]:
        """
        Find generic definite descriptions inside a noun phrase.

        Handles both normal and possessive forms.

        Examples:
            "the company"
                -> ("the company", "the company")

            "the country's exports"
                -> ("the country's", "the country")

            "the company's shares"
                -> ("the company's", "the company")

            "his stake in the company"
                -> ("the company", "the company")
        """

        words = text.split()

        found: list[tuple[str, str]] = []

        generic_heads = set(
            GENERIC_HEAD_NOUN_TYPE_HINTS
        )

        for i in range(len(words) - 1):

            first = words[i].strip(
                ".,!?;:\"'()[]{}"
            )

            second_raw = words[i + 1].strip(
                ".,!?;:\"'()[]{}"
            )

            if first.lower() != "the":
                continue

            # -----------------------------------------------
            # Normal form:
            #
            # "the company"
            # "the government"
            # -----------------------------------------------

            second = second_raw.lower()

            if second in generic_heads:

                surface = f"{first} {second_raw}"

                found.append(
                    (
                        surface,
                        f"the {second}",
                    )
                )

                continue

            # -----------------------------------------------
            # Possessive form:
            #
            # "the country's"
            # "the company's"
            # "the firm's"
            # -----------------------------------------------

            if second.endswith("'s"):

                head = second[:-2]

                if head in generic_heads:

                    surface = (
                        f"{first} {second_raw}"
                    )

                    found.append(
                        (
                            surface,
                            f"the {head}",
                        )
                    )

                    continue

            # -----------------------------------------------
            # Curly/normalized possessive:
            #
            # "the country’s"
            # -----------------------------------------------

            if second.endswith("’s"):

                head = second[:-2]

                if head in generic_heads:

                    surface = (
                        f"{first} {second_raw}"
                    )

                    found.append(
                        (
                            surface,
                            f"the {head}",
                        )
                    )

        return found

    # ========================================================
    # Generic head noun
    # ========================================================

    @staticmethod
    def _generic_head_noun(
        text: str,
    ) -> str:
        """
        Extract the generic head noun.

        Examples:
            "the company"   -> "company"
            "the company's" -> "company"
            "the country's" -> "country"
            "the government" -> "government"
        """

        normalized = (
            text.strip()
            .lower()
            .replace("’", "'")
        )

        if not normalized.startswith("the "):
            return normalized

        words = normalized.split()

        if len(words) < 2:
            return normalized

        head = words[1]

        if head.endswith("'s"):
            head = head[:-2]

        return head

    # ========================================================
    # Text normalization
    # ========================================================

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        """
        Normalize spaCy surface text without changing meaning.
        """

        text = str(text).strip()

        if not text:
            return ""

        # spaCy possessive artifact:
        #
        # "China 's"
        #     ->
        # "China's"
        text = re.sub(
            r"\s+'\s*s\b",
            "'s",
            text,
            flags=re.IGNORECASE,
        )

        # Collapse whitespace.
        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # ========================================================
    # Character position lookup
    # ========================================================

    @staticmethod
    def _locate_text(
        sentence: str,
        text: str,
    ) -> tuple[int, int]:
        """
        Locate text inside the original sentence.

        Returns:

            (start, end)

        or:

            (-1, -1)

        if the surface form cannot be located.

        Case-insensitive matching is used because the graph may
        contain normalized text.
        """

        if not sentence or not text:
            return -1, -1

        match = re.search(
            re.escape(text),
            sentence,
            flags=re.IGNORECASE,
        )

        if match is None:
            return -1, -1

        return (
            match.start(),
            match.end(),
        )

    # ========================================================
    # Canonical ID
    # ========================================================

    @staticmethod
    def _canonical_id(
        text: str,
    ) -> str:
        """
        Create a deterministic canonical ID.

        V2 remains conservative:
        "Microsoft" and "Microsoft Corp" are NOT automatically
        merged yet.
        """

        normalized = (
            text.lower().strip()
        )

        normalized = re.sub(
            r"[^a-z0-9]+",
            "-",
            normalized,
        )

        return normalized.strip("-")


# ============================================================
# Convenience function
# ============================================================


def resolve_entities(
    sentence_graphs: list[SemanticGraph],
    target_lists: list[list[TargetSelection]],
) -> ResolutionResult:
    """
    Convenience wrapper around EntityResolver.resolve().
    """

    resolver = EntityResolver()

    return resolver.resolve(
        sentence_graphs,
        target_lists,
    )


# ============================================================
# Manual module test
# ============================================================


if __name__ == "__main__":

    print(
        "EntityResolver V2 module loaded successfully."
    )

    print(
        "Run test_entity_resolver.py to test "
        "real sentence resolution."
    )
"""
semantic/target_selector.py
---------------------------

Rule-based semantic target selector.

Architecture:

    Sentence
        ↓
    SemanticExtractor
        ↓
    SemanticGraph
        ↓
    TargetSelector
        ↓
    Target

Important:
    This module does NOT use gold targets.
    This module does NOT use the old candidate ranker.

The selector reasons over semantic roles:

    event
      ├── subject
      └── object

The goal of this first version is interpretability.
Every selected target should have an explicit reason.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from semantic.semantic_graph import SemanticGraph


# ============================================================
# Target result
# ============================================================


@dataclass
class TargetSelection:
    """Result returned by the semantic target selector."""

    target: str | None

    reason: str

    event: str | None

    role: str | None

    evidence: list[str]

    confidence: str

    embedded_entities: list[str] = field(default_factory=list)
    """
    Named entities recognized INSIDE this target's text, if any.

    Example: target="a new plan for Tesla" -> embedded_entities=["Tesla"]

    Per target_annotation_guidelines_v1.md Section 2: the target
    TEXT stays as the full extracted phrase (no information lost),
    but any named entity found within it is tagged here separately,
    so dataset-building / downstream steps can cheaply recover the
    short-entity view without re-parsing the phrase.

    Empty list means either no entity was found inside the phrase,
    or the target text itself already IS a bare entity (e.g.
    target="Tesla" with nothing else in the phrase) -- in that case
    checking embedded_entities isn't needed since the target IS the
    entity.
    """

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a JSON-friendly dictionary."""

        return asdict(self)


# ============================================================
# Target selector
# ============================================================


class TargetSelector:
    """
    Select the most semantically important target.

    This is intentionally rule-based.

    We want to understand the behavior of the semantic
    architecture before introducing another ML model.
    """

    # --------------------------------------------------------
    # Events where the OBJECT is usually the main target.
    # --------------------------------------------------------

    OBJECT_FOCUSED_EVENTS = {
        "acquire",
        "acquired",
        "buy",
        "bought",
        "purchase",
        "purchased",
        "launch",
        "launched",
        "introduce",
        "introduced",
        "announce",
        "announced",
        "impose",
        "imposed",
        "cut",
        "cuts",
        "raise",
        "raised",
        "reduce",
        "reduced",
        "increase",
        "increased",
        "approve",
        "approved",
        "ban",
        "banned",
        "sanction",
        "sanctioned",
        "invest",
        "invested",
    }

    # --------------------------------------------------------
    # Events where the SUBJECT is usually the main target.
    # --------------------------------------------------------

    SUBJECT_FOCUSED_EVENTS = {
        "be",
        "is",
        "was",
        "were",
        "are",
        "report",
        "reported",
        "speak",
        "spoke",
        "say",
        "said",
        "meet",
        "met",
        "visit",
        "visited",
        "rise",
        "rose",
        "fall",
        "fell",
        "climb",
        "climbed",
        "drop",
        "dropped",
        "compete",
        "competing",
        "discuss",
        "discussed",
    }

    # --------------------------------------------------------
    # Generic / weak noun phrases.
    #
    # These should not normally become the target when a
    # meaningful entity is available.
    # --------------------------------------------------------

    GENERIC_TERMS = {
        # Bare party-affiliation letters -- confirmed via real
        # dataset generation output as a recurring noise pattern
        # from the political-news idiom "Name, R-Fla." or
        # "Name (R)" (meaning "Republican from Florida"). Our
        # appositive-splitting logic (correctly designed to split
        # genuine coordinate lists like "Amazon, Google, and
        # Microsoft") also fires on this idiom, since spaCy tags
        # the bare party letter as its own entity-like appositive.
        # A standalone "R"/"D" is essentially never a meaningful
        # sentiment target on its own in this corpus -- always
        # noise from this specific idiom.
        "r",
        "d",
        "gop",
        # AP-style state abbreviations -- same idiom, same root
        # cause as the bare party letters above: "Name, R-Fla."
        # attaches the state abbreviation as another entity-like
        # (GPE) appositive, which correctly gets split out as a
        # coordinate candidate, but a bare abbreviated state form
        # like "Fla." or "Ariz." is metadata about the person, not
        # an independent target worth scoring sentiment on. Full
        # state names ("Florida", "Arizona") are NOT included here
        # and remain valid targets elsewhere -- only the
        # abbreviated, period-containing short forms specific to
        # this idiom are filtered.
        "ala.", "ariz.", "ark.", "calif.", "colo.", "conn.",
        "del.", "fla.", "ga.", "ill.", "ind.", "kan.", "ky.",
        "la.", "mass.", "md.", "mich.", "minn.", "miss.", "mo.",
        "mont.", "neb.", "nev.", "okla.", "ore.", "pa.", "tenn.",
        "tex.", "va.", "vt.", "wash.", "wis.", "wyo.",
        "n.c.", "n.d.", "n.h.", "n.j.", "n.m.", "n.y.",
        "r.i.", "s.c.", "s.d.", "w.va.",
        "company",
        "companies",
        "firm",
        "firms",
        "people",
        "officials",
        "official",
        "investors",
        "investor",
        "members",
        "group",
        "groups",
        "organization",
        "organizations",
        "government",
        "government officials",
        "country",
        "nation",
        "the company",
        "the firm",
        "the organization",
        "the government",
        "the country",
        "the nation",
        "the situation",
        "the agreement",
        "the conference",
        "the event",
        "today",
        "yesterday",
        "tomorrow",
        "this week",
        "this month",
        "this year",
    }

    # --------------------------------------------------------
    # Bare pronouns.
    #
    # A pronoun like "it" or "they" is not a usable sentiment
    # target on its own -- it needs to be resolved to the real
    # entity it refers to (e.g. "it" -> "Meta"), which is the
    # job of entity resolution / coreference, NOT of TargetSelector.
    #
    # Rather than emit an unresolved pronoun as if it were a
    # valid target, we filter it out here. Once entity_resolver.py
    # exists, unresolved pronouns can be looked up and reintroduced
    # with their resolved reference instead of being dropped.
    # --------------------------------------------------------

    PRONOUNS = {
        "it",
        "its",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "they",
        "them",
        "their",
        "theirs",
        "this",
        "that",
        "these",
        "those",
        "we",
        "us",
        "our",
        "ours",
        "i",
        "me",
        "my",
        "mine",
        "you",
        "your",
        "yours",
        # Relative pronouns -- e.g. "the president WHO nominated
        # him" -- "who" ends up as nsubj of the relative clause
        # verb and would otherwise leak through as a bare target.
        # Confirmed via real dataset generation output.
        "who",
        "whom",
        "whose",
        "which",
        "what",
    }

    # --------------------------------------------------------
    # Entity types that should never be picked as a fallback
    # target on their own.
    #
    # These are spaCy NER labels for things like dates, counts,
    # and percentages -- technically "entities", but never
    # meaningful sentiment targets by themselves. Without this
    # filter, _fallback() can pick something like "quarter"
    # (DATE) over a much more meaningful candidate like "the
    # technology sector", just because the DATE entity happened
    # to appear first in the entity list.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Safety-net length cap for target text.
    #
    # The primary fix for unwieldy long targets is excluding
    # relative clauses / clausal complements at extraction time
    # (semantic_extractor.py's _own_span_tokens). This cap is a
    # BACKSTOP for what that doesn't cover -- confirmed via real
    # dataset generation: chains of stacked prepositional phrases
    # ("the possibility of X by Y under Z...") and clauses buried
    # inside xcomp/infinitival constructions can still produce
    # 30-40 word "targets" that are unusable for sentiment
    # labeling. Deliberately generous (15 words) since this is a
    # last-resort net, not the primary mechanism -- most real
    # targets are far shorter than this already.
    # --------------------------------------------------------

    MAX_TARGET_WORDS = 15

    WEAK_ENTITY_LABELS = {
        "DATE",
        "TIME",
        "CARDINAL",
        "ORDINAL",
        "PERCENT",
        "QUANTITY",
    }

    # --------------------------------------------------------
    # Main public method
    # --------------------------------------------------------

    def select(
        self,
        graph: SemanticGraph,
    ) -> TargetSelection:
        """
        Select the single best target from a SemanticGraph.

        No gold target is required.

        This now reuses select_all() rather than maintaining a
        separate implementation. Previously select() and
        select_all() had two independent, hand-written copies of
        the same event-walking logic, which meant a fix applied
        to one (e.g. the generic-term or pronoun filtering fixes)
        could silently fail to apply to the other. Now select()
        just asks select_all() for every candidate and picks the
        best one, so the two methods can never disagree.

        "Best" = highest confidence ("high" > "medium" > "low" >
        "none"), and among ties, whichever candidate select_all()
        found first (i.e. earliest event in the sentence).
        """

        if not isinstance(graph, SemanticGraph):
            raise TypeError(
                "graph must be a SemanticGraph"
            )

        candidates = self.select_all(graph)

        if not candidates:
            return TargetSelection(
                target=None,
                reason="No meaningful semantic target was found.",
                event=None,
                role=None,
                evidence=[],
                confidence="none",
            )

        confidence_rank = {
            "high": 3,
            "medium": 2,
            "low": 1,
            "none": 0,
        }

        best = max(
            candidates,
            key=lambda c: confidence_rank.get(c.confidence, 0),
        )

        return best

    # ========================================================
    # Multi-target selection (NEW)
    # ========================================================

    def select_all(
        self,
        graph: SemanticGraph,
    ) -> list[TargetSelection]:
        """
        Select ALL plausible sentiment targets in the sentence,
        not just one.

        This exists because NewsMTSC is a MULTI-target sentiment
        task: a single sentence like

            "Google announced layoffs and Meta announced hiring."

        has two independent targets (Google, Meta) that can and
        should receive different sentiment labels. The original
        select() stops at the first match and silently drops
        every other candidate — which throws away most of the
        signal this pipeline needs to produce.

        Strategy:
            1. Walk every event, and for each one, pick a target
               using the same object-focused / subject-focused
               logic as select() — but do NOT stop after the
               first one.
            2. Deduplicate by normalized target text (case-
               insensitive), keeping the first (highest-confidence)
               selection for each unique target.
            3. If NO events produced any target, fall back to
               scanning entities directly (same as _fallback()).
        """

        if not isinstance(graph, SemanticGraph):
            raise TypeError(
                "graph must be a SemanticGraph"
            )

        results: list[TargetSelection] = []
        seen_targets: set[str] = set()

        def _add(selection: TargetSelection) -> None:
            if selection.target is None:
                return
            key = selection.target.strip().lower()
            if key in seen_targets:
                return
            seen_targets.add(key)
            results.append(selection)

        for event in graph.events:

            lemma = (
                event.lemma or event.trigger
            ).lower().strip()

            obj_candidates = self._filter_candidates(
                event.objects, graph
            )
            subj_candidates = self._filter_candidates(
                event.subjects, graph
            )

            # Decide confidence/reason per role. Whichever role the
            # verb is classified toward gets "high" confidence;
            # the other role (if also present) still gets added,
            # but at "medium" confidence, since it's a secondary
            # participant rather than the verb's primary focus.
            #
            # ALL candidates for a role are added as separate
            # targets, not just one "best" one. This matters for
            # coordination: "Amazon, Google, and Microsoft are
            # investing" should surface all three as independent
            # targets, not collapse to a single winner.

            if lemma in self.OBJECT_FOCUSED_EVENTS:
                obj_confidence = "high"
                subj_confidence = "medium"
            elif lemma in self.SUBJECT_FOCUSED_EVENTS:
                obj_confidence = "medium"
                subj_confidence = "high"
            else:
                obj_confidence = "medium"
                subj_confidence = "medium"

            for obj_target in obj_candidates:

                capped_obj_target = self._apply_length_cap(obj_target, graph)

                if capped_obj_target is None:
                    continue

                _add(
                    TargetSelection(
                        target=capped_obj_target,
                        reason=(
                            f"Selected as a semantic object of "
                            f"'{event.trigger}'."
                            + (
                                " (shortened from a longer phrase "
                                "due to length cap)"
                                if capped_obj_target != obj_target
                                else ""
                            )
                        ),
                        event=event.trigger,
                        role="object",
                        evidence=[
                            f"event={event.trigger}",
                            f"object={capped_obj_target}",
                        ],
                        confidence=obj_confidence,
                        embedded_entities=self._find_embedded_entities(
                            capped_obj_target, graph
                        ),
                    )
                )

            for subj_target in subj_candidates:

                capped_subj_target = self._apply_length_cap(subj_target, graph)

                if capped_subj_target is None:
                    continue

                _add(
                    TargetSelection(
                        target=capped_subj_target,
                        reason=(
                            f"Selected as a semantic subject of "
                            f"'{event.trigger}'."
                            + (
                                " (shortened from a longer phrase "
                                "due to length cap)"
                                if capped_subj_target != subj_target
                                else ""
                            )
                        ),
                        event=event.trigger,
                        role="subject",
                        evidence=[
                            f"event={event.trigger}",
                            f"subject={capped_subj_target}",
                        ],
                        confidence=subj_confidence,
                        embedded_entities=self._find_embedded_entities(
                            capped_subj_target, graph
                        ),
                    )
                )

        # If events produced nothing at all, fall back to entities.
        if not results:
            fallback = self._fallback(graph)
            if fallback.target is not None:
                _add(fallback)

        return results

    def _find_embedded_entities(
        self,
        target_text: str,
        graph: SemanticGraph,
    ) -> list[str]:
        """
        Find named entities recognized INSIDE target_text.

        Returns an empty list if target_text itself already IS one
        of the entities (nothing extra to tag), if no entity text
        appears within it at all, or if the only entities found are
        weak types (DATE, CARDINAL, etc. -- see WEAK_ENTITY_LABELS).

        Weak types are excluded here for the same reason they're
        excluded from _fallback(): something like "quarterly"
        (DATE) or "Thousands" (CARDINAL) is technically a spaCy
        entity, but tagging it as an "embedded entity" inside a
        phrase like "record quarterly revenue" is noise, not
        useful signal -- nobody building a shortened target view
        would want to shorten that phrase down to "quarterly".

        Uses simple case-insensitive substring matching, which is
        good enough given entity spans are usually short and
        already reasonably well-formed. A minimum length guard (2
        chars) prevents a single-character "entity" (e.g. a bare
        party-affiliation letter "R" mis-tagged as NORP) from
        spuriously matching almost any longer text purely by
        coincidence -- confirmed via real dataset generation output:
        "R" was showing up as an "embedded entity" inside "Marco
        Rubio" only because the letter r happens to appear inside
        the word "Rubio", not because it's a genuine embedded
        reference to anything.
        """

        normalized_target = target_text.strip().lower()

        found: list[str] = []
        seen: set[str] = set()

        for entity in graph.entities:

            if entity.label in self.WEAK_ENTITY_LABELS:
                continue

            entity_text = entity.text.strip()
            normalized_entity = entity_text.lower()

            if len(normalized_entity) < 2:
                continue

            if not normalized_entity:
                continue

            # Skip if the target text IS this entity (nothing to tag).
            if normalized_entity == normalized_target:
                continue

            if normalized_entity in normalized_target:

                if normalized_entity in seen:
                    continue

                seen.add(normalized_entity)
                found.append(entity_text)

        return found

    def _apply_length_cap(
        self,
        target_text: str,
        graph: SemanticGraph,
    ) -> str | None:
        """
        Enforce MAX_TARGET_WORDS as a last-resort safety net.

        If target_text is within the cap, returned unchanged. If
        it exceeds the cap:
            - shorten to the FIRST embedded entity found within it
              (e.g. a 40-word phrase containing "the Constitution"
              shortens to just "the Constitution"), or
            - if no embedded entity exists at all, return None
              (signal to the caller to drop this candidate
              entirely, rather than keep an unusably long,
              entity-less phrase).

        This is deliberately a backstop, not the primary mechanism
        -- see MAX_TARGET_WORDS' docstring for why it exists
        despite the clause-boundary exclusion already in
        semantic_extractor.py.
        """

        if len(target_text.split()) <= self.MAX_TARGET_WORDS:
            return target_text

        embedded = self._find_embedded_entities(target_text, graph)

        if embedded:
            return embedded[0]

        return None

    # ========================================================
    # Candidate value selection
    # ========================================================

    @staticmethod
    def _normalize_for_entity_match(
        text: str,
    ) -> str:
        """
        Lowercase and strip a leading article ("the"/"a"/"an")
        before comparing a candidate phrase against entity text.

        Needed because spaCy's NER span and our extracted noun
        phrase don't always agree on whether the article is
        included. Example: the entity "UK" (no article) vs. our
        extracted subject phrase "The UK" (article included, since
        determiners aren't excluded from a token's own span). A
        plain lowercase-equality check would treat "the uk" != "uk"
        and wrongly drop a real entity as if it were noise.
        """

        normalized = text.strip().lower()

        for article in ("the ", "a ", "an "):
            if normalized.startswith(article):
                return normalized[len(article):]

        return normalized

    @staticmethod
    def _is_entity_match(
        candidate: str,
        entity_texts: set[str],
    ) -> bool:
        """
        True if `candidate` corresponds to a recognized entity --
        either an exact match, or (fallback) a substring of a
        longer entity span.

        The substring fallback exists because spaCy's NER
        sometimes merges multiple REAL entities from a
        comma-separated list into ONE malformed entity span --
        e.g. "the Democratic National Committee, Facebook" gets
        tagged as a SINGLE ORG entity, comma included. Neither
        "the Democratic National Committee" nor "Facebook" alone
        exactly matches that merged text, so a pure equality check
        would wrongly treat both as "not real entities" and drop
        them via the entity-priority filter in _filter_candidates
        -- even though they're both clearly genuine entities that
        just got fused together by an NER quirk.

        A minimum length guard (4 chars) avoids short candidate
        strings spuriously matching as a substring of an unrelated
        longer entity span.
        """

        candidate_norm = TargetSelector._normalize_for_entity_match(
            candidate
        )

        if not candidate_norm:
            return False

        if candidate_norm in entity_texts:
            return True

        if len(candidate_norm) < 4:
            return False

        for entity_text in entity_texts:
            if candidate_norm in entity_text:
                return True

        return False

    def _filter_candidates(
        self,
        values: list[str],
        graph: SemanticGraph,
    ) -> list[str]:
        """
        Return the usable candidates from a list of semantic
        participants (e.g. event.objects or event.subjects).

        Step 1 -- basic filtering (applies always):
            - drop empty/whitespace-only values
            - drop bare pronouns ("it", "they", ...)
            - drop generic terms ("the company", "officials", ...)
            - dedupe (case-insensitive), keeping first occurrence

        Step 2 -- disambiguating multiple survivors:
            If more than one candidate remains, we can't just keep
            them all -- our extraction code puts BOTH genuine
            coordination ("Amazon, Google, and Microsoft") and
            unrelated same-role noise into the same flat list, and
            can't currently tell them apart structurally. So:

              - If ANY candidate is a recognized named entity,
                keep ONLY the entity matches, dropping the rest.
                This handles two real cases we hit in testing:
                  * "Amazon, Google, Microsoft" -- all entities,
                    all kept (genuine coordination).
                  * "layoffs, Meta" (from a parser mis-attachment)
                    -- only "Meta" is an entity, "layoffs" is
                    correctly dropped as noise.

              - If NONE of the candidates are entities, we have no
                reliable way to tell coordination from unrelated
                grammatical roles (e.g. "interest rates" + a
                separate object-complement "unchanged" getting
                merged into the same list), so we conservatively
                keep only the FIRST one, same as the old
                single-winner behavior.

            KNOWN LIMITATION: genuine coordination between
            non-entity noun phrases (e.g. "AI and machine
            learning are growing fields") will lose everything
            after the first item under this rule. We're accepting
            that trade-off for now to avoid noise like "unchanged"
            being treated as a real target -- revisit if this
            turns out to matter on more real sentences.
        """

        if not values:
            return []

        seen: set[str] = set()
        candidates: list[str] = []

        for value in values:

            normalized = value.strip().lower()

            if not normalized:
                continue

            if normalized in self.PRONOUNS:
                continue

            if normalized in self.GENERIC_TERMS:
                continue

            if normalized in seen:
                continue

            seen.add(normalized)
            candidates.append(value.strip())

        if len(candidates) <= 1:
            return candidates

        entity_texts = {
            self._normalize_for_entity_match(entity.text)
            for entity in graph.entities
        }

        entity_matches = [
            c for c in candidates
            if self._is_entity_match(c, entity_texts)
        ]

        if entity_matches:
            return entity_matches

        return candidates[:1]

    def _best_value(
        self,
        values: list[str],
        graph: SemanticGraph,
    ) -> str | None:
        """
        Select the SINGLE strongest value from a list of semantic
        participants, preferring named entities. Kept for backward
        compatibility and for callers that genuinely want one
        representative value rather than every candidate (use
        _filter_candidates() for the latter).

        Priority:

            1. Named entity
            2. First remaining non-generic, non-pronoun candidate
        """

        candidates = self._filter_candidates(values, graph)

        if not candidates:
            return None

        entity_texts = {
            self._normalize_for_entity_match(entity.text)
            for entity in graph.entities
        }

        for candidate in candidates:

            if self._is_entity_match(candidate, entity_texts):
                return candidate

        return candidates[0]

    # ========================================================
    # Fallback
    # ========================================================

    def _fallback(
        self,
        graph: SemanticGraph,
    ) -> TargetSelection:
        """
        Fallback when no useful event relation exists.
        """

        # Prefer named entities.

        for entity in graph.entities:

            text = entity.text.strip()

            if not text:
                continue

            if text.lower() in self.GENERIC_TERMS:
                continue

            if entity.label in self.WEAK_ENTITY_LABELS:
                continue

            return TargetSelection(
                target=text,
                reason=(
                    "No suitable event relation was "
                    "available, so the first meaningful "
                    "named entity was selected."
                ),
                event=None,
                role="entity",
                evidence=[
                    f"entity={text}",
                    f"entity_label={entity.label}",
                ],
                confidence="low",
            )

        # Then noun phrases.

        for phrase in graph.noun_phrases:

            text = phrase.strip()

            if not text:
                continue

            if text.lower() in self.PRONOUNS:
                continue

            if text.lower() in self.GENERIC_TERMS:
                continue

            capped_text = self._apply_length_cap(text, graph)

            if capped_text is None:
                # Too long with no embedded entity to shorten to --
                # skip this phrase and try the next one, rather
                # than returning an unusably long fallback target.
                continue

            return TargetSelection(
                target=capped_text,
                reason=(
                    "No event or named entity provided a "
                    "strong target, so a meaningful noun "
                    "phrase was selected."
                    + (
                        " (shortened from a longer phrase due to "
                        "length cap)"
                        if capped_text != text
                        else ""
                    )
                ),
                event=None,
                role="noun_phrase",
                evidence=[
                    f"noun_phrase={capped_text}",
                ],
                confidence="low",
                embedded_entities=self._find_embedded_entities(
                    capped_text, graph
                ),
            )

        return TargetSelection(
            target=None,
            reason="No meaningful semantic target was found.",
            event=None,
            role=None,
            evidence=[],
            confidence="none",
        )


# ============================================================
# Convenience function
# ============================================================


def select_target(
    graph: SemanticGraph,
) -> TargetSelection:
    """
    Convenience wrapper around TargetSelector.
    """

    selector = TargetSelector()

    return selector.select(
        graph
    )


# ============================================================
# Manual test
# ============================================================


if __name__ == "__main__":

    print(
        "TargetSelector module loaded successfully."
    )

    print(
        "\nThis module expects a SemanticGraph."
    )

    print(
        "Run the dedicated target-selector test "
        "to test real sentences."
    )
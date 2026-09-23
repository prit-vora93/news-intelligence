"""
semantic/semantic_extractor.py
------------------------------

Semantic analysis layer for the News Intelligence project.

Purpose:
    Convert a raw sentence into a structured semantic representation.

This is NOT the target-ranking model.

Architecture:

    Sentence
        ↓
    spaCy NLP
        ↓
    ┌───────────────┬───────────────┬───────────────┐
    │   Entities    │   Noun Phrases│    Events     │
    └───────────────┴───────────────┴───────────────┘
        ↓
    Dependencies / Relations
        ↓
    Structured representation

The output will later be consumed by target_selector.py.

The important design decision here is:
    We do NOT decide the final target yet.

We first build a semantic representation of the sentence.
"""


from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


# ============================================================
# spaCy
# ============================================================

try:
    import spacy
except ImportError as exc:
    raise ImportError(
        "spaCy is required. Install it with:\n"
        "    pip install spacy"
    ) from exc


# ============================================================
# Configuration
# ============================================================

DEFAULT_MODEL = "en_core_web_sm"


# ============================================================
# Data structures
# ============================================================


@dataclass
class Entity:
    """Named entity detected in the sentence."""

    text: str
    label: str
    start: int
    end: int


@dataclass
class NounPhrase:
    """Noun phrase detected by the dependency parser."""

    text: str
    root: str
    root_pos: str
    start: int
    end: int


@dataclass
class TokenInfo:
    """Important dependency information for a token."""

    text: str
    lemma: str
    pos: str
    tag: str
    dep: str
    head: str
    start: int
    end: int


@dataclass
class Event:
    """
    Lightweight event representation.

    At this stage an event is primarily a verbal trigger.
    We are deliberately not trying to solve full event
    extraction yet.
    """

    trigger: str
    lemma: str
    pos: str
    start: int
    end: int


@dataclass
class Relation:
    """
    Dependency-based relation between tokens.

    Example:

        Microsoft acquired Activision Blizzard

    may produce information such as:

        subject: Microsoft
        relation: acquired
        object: Activision Blizzard

    event_start:
        Character offset of the verb token this relation was
        derived from. This lets downstream code (SemanticGraphBuilder)
        match a relation back to its EXACT source event, instead of
        matching by lemma text — which breaks when the same lemma
        (e.g. "announced") appears more than once in a sentence.
    """

    subject: str | None
    relation: str
    object: str | None
    event_start: int


@dataclass
class SemanticRepresentation:
    """Complete semantic representation of one sentence."""

    sentence: str
    entities: list[Entity]
    noun_phrases: list[NounPhrase]
    events: list[Event]
    relations: list[Relation]
    tokens: list[TokenInfo]


# ============================================================
# Semantic extractor
# ============================================================


class SemanticExtractor:
    """
    Convert sentences into structured semantic representations.

    Example:

        extractor = SemanticExtractor()

        result = extractor.extract(
            "Microsoft acquired Activision Blizzard."
        )
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
    ) -> None:

        print(
            f"Loading spaCy model: {model_name}"
        )

        try:
            self.nlp = spacy.load(
                model_name
            )
        except OSError as exc:
            raise OSError(
                f"Could not load spaCy model "
                f"'{model_name}'.\n\n"
                f"Install it with:\n"
                f"    python -m spacy download {model_name}"
            ) from exc

        print(
            "spaCy model loaded successfully."
        )

    # ========================================================
    # Public API
    # ========================================================

    def extract(
        self,
        sentence: str,
    ) -> SemanticRepresentation:
        """
        Extract semantic information from one sentence.
        """

        if not isinstance(sentence, str):
            raise TypeError(
                "sentence must be a string"
            )

        sentence = sentence.strip()

        if not sentence:
            return SemanticRepresentation(
                sentence="",
                entities=[],
                noun_phrases=[],
                events=[],
                relations=[],
                tokens=[],
            )

        doc = self.nlp(sentence)

        entities = self._extract_entities(
            doc
        )

        noun_phrases = self._extract_noun_phrases(
            doc
        )

        events = self._extract_events(
            doc
        )

        relations = self._extract_relations(
            doc
        )

        tokens = self._extract_tokens(
            doc
        )

        return SemanticRepresentation(
            sentence=sentence,
            entities=entities,
            noun_phrases=noun_phrases,
            events=events,
            relations=relations,
            tokens=tokens,
        )

    # ========================================================
    # Entity extraction
    # ========================================================

    @staticmethod
    def _extract_entities(
        doc: Any,
    ) -> list[Entity]:

        entities: list[Entity] = []

        for ent in doc.ents:

            entities.append(
                Entity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                )
            )

        return entities

    # ========================================================
    # Noun phrase extraction
    # ========================================================

    @staticmethod
    def _extract_noun_phrases(
        doc: Any,
    ) -> list[NounPhrase]:

        noun_phrases: list[NounPhrase] = []

        for chunk in doc.noun_chunks:

            noun_phrases.append(
                NounPhrase(
                    text=chunk.text,
                    root=chunk.root.text,
                    root_pos=chunk.root.pos_,
                    start=chunk.start_char,
                    end=chunk.end_char,
                )
            )

        return noun_phrases

    # ========================================================
    # Event trigger detection
    # ========================================================

    @staticmethod
    def _is_event_trigger(
        token: Any,
    ) -> bool:
        """
        True if this token should be treated as an event trigger.

        Two cases:

        1. token.pos_ == "VERB" -- the normal case.

        2. token.pos_ == "AUX" AND it's acting as a true COPULA
           (a linking "to be"), not just a modal/auxiliary
           supporting a separate main verb.

           Why this matters: spaCy tags "is"/"was"/"are"/"be" as
           AUX, not VERB -- even when it's the ENTIRE predicate of
           the sentence, e.g. "Trump is a diplomat" has "is" as
           dep_="ROOT" with no other VERB in the clause at all.
           Without this case, copular sentences ("X is Y", "X was
           Y") produce ZERO events and their subject (Trump, Roger,
           Handel, ...) never gets extracted -- confirmed via real
           spaCy output during evaluation against gold NewsMTSC data.

           We deliberately do NOT treat every AUX as a trigger --
           that would wrongly fire on modals like "will" in "will
           invest" (where "invest", the real VERB, is already
           captured separately; "will" is just supporting it, with
           dep_="aux" not "ROOT", and has no attr/acomp child).

           A copula qualifies only when it has BOTH a subject
           (nsubj/nsubjpass) AND a predicate complement
           (attr/acomp) as children -- that combination is specific
           to genuine linking-verb usage.
        """

        if token.pos_ == "VERB":
            return True

        if token.pos_ != "AUX":
            return False

        if token.dep_ not in {"ROOT", "conj"}:
            return False

        has_subject = any(
            c.dep_ in {"nsubj", "nsubjpass"} for c in token.children
        )
        has_predicate_complement = any(
            c.dep_ in {"attr", "acomp"} for c in token.children
        )

        return has_subject and has_predicate_complement

    # ========================================================
    # Event extraction
    # ========================================================

    @staticmethod
    def _extract_events(
        doc: Any,
    ) -> list[Event]:

        events: list[Event] = []

        for token in doc:

            if not SemanticExtractor._is_event_trigger(token):
                continue

            events.append(
                Event(
                    trigger=token.text,
                    lemma=token.lemma_,
                    pos=token.pos_,
                    start=token.idx,
                    end=token.idx + len(token.text),
                )
            )

        return events

    # ========================================================
    # Relation extraction
    # ========================================================

    @staticmethod
    def _extract_relations(
        doc: Any,
    ) -> list[Relation]:
        """
        NOTE: this also follows `conj` chains (coordination via
        "and"/"or") so that sentences like

            "Google and Meta announced layoffs."

        capture BOTH Google and Meta as subjects, not just the
        first one spaCy attaches directly to the verb. Without
        this, coordinated entities are silently dropped — a
        common pattern in news text.

        Also treats true copular AUX tokens as event triggers (see
        _is_event_trigger) so that "X is Y" / "X was Y" sentences
        produce a relation, not just VERB-headed sentences.
        """

        relations: list[Relation] = []

        for token in doc:

            if not SemanticExtractor._is_event_trigger(token):
                continue

            subjects: list[str] = []
            objects: list[str] = []

            for child in token.children:

                # Nominal subjects
                if child.dep_ in {
                    "nsubj",
                    "nsubjpass",
                    "csubj",
                }:

                    subjects.extend(
                        _with_conjuncts(child)
                    )

                # Direct objects (including copular predicate
                # complements: attr = predicate noun, acomp =
                # predicate adjective; and dative = indirect object
                # of ditransitive verbs, e.g. "gave HIM the book")
                elif child.dep_ in {
                    "dobj",
                    "obj",
                    "attr",
                    "oprd",
                    "acomp",
                    "dative",
                }:

                    objects.extend(
                        _with_conjuncts(child)
                    )

                # Passive-agent phrase.
                #
                # child here is the PREPOSITION "by" itself
                # (dep_ == "agent"), not the actor. E.g. in
                # "Twitter was acquired by Elon Musk", the agent
                # token is "by", and the real actor "Elon Musk"
                # is its pobj (object of preposition) child.
                # Using child directly would include "by" in the
                # extracted text ("by Elon Musk" instead of
                # "Elon Musk"). Descend to the pobj child first.
                elif child.dep_ == "agent":

                    pobj_child = _agent_pobj(child)

                    objects.extend(
                        _with_conjuncts(pobj_child)
                    )

                # General prepositional-phrase object, e.g.
                # "flying TO MEXICO" or "AT 21ST CENTURY FOX, we...".
                #
                # Only pulled in when it contains a real named
                # entity (see _contains_strong_entity) -- PPs carry
                # a lot of adjunct noise ("on a moment's notice",
                # "during an interview") we deliberately do NOT
                # want as target candidates. This is intentionally
                # more conservative than our other object paths.
                elif child.dep_ == "prep":

                    pobj_child = _find_pobj(child)

                    if pobj_child is not None and _contains_strong_entity(pobj_child):
                        objects.extend(
                            _with_conjuncts(pobj_child)
                        )

            # Keep at least one relation if either
            # subject or object exists.
            if subjects or objects:

                if subjects:
                    subject_values = subjects
                else:
                    subject_values = [None]

                if objects:
                    object_values = objects
                else:
                    object_values = [None]

                for subject in subject_values:

                    for obj in object_values:

                        relations.append(
                            Relation(
                                subject=subject,
                                relation=token.lemma_,
                                object=obj,
                                event_start=token.idx,
                            )
                        )

        return relations

    # ========================================================
    # Token extraction
    # ========================================================

    @staticmethod
    def _extract_tokens(
        doc: Any,
    ) -> list[TokenInfo]:

        tokens: list[TokenInfo] = []

        for token in doc:

            tokens.append(
                TokenInfo(
                    text=token.text,
                    lemma=token.lemma_,
                    pos=token.pos_,
                    tag=token.tag_,
                    dep=token.dep_,
                    head=token.head.text,
                    start=token.idx,
                    end=token.idx + len(token.text),
                )
            )

        return tokens

    # ========================================================
    # Dictionary representation
    # ========================================================

    def extract_dict(
        self,
        sentence: str,
    ) -> dict[str, Any]:
        """
        Extract semantic representation as a normal dictionary.

        Useful for debugging, JSON serialization and APIs.
        """

        representation = self.extract(
            sentence
        )

        return asdict(
            representation
        )


# ============================================================
# Helper
# ============================================================


def _is_entity_like(
    token: Any,
) -> bool:
    """
    True if this token is part of a spaCy-recognized named entity
    span (token.ent_type_ is non-empty).

    Used to distinguish two different uses of the "appos"
    (appositive) dependency label, which spaCy overloads:

    1. A genuine coordinate list item that got mis-parsed, e.g.
       "Amazon, Google, and Microsoft" -- spaCy attaches "Google"
       as appos to "Amazon" instead of conj (only "Microsoft" gets
       the correct conj label). Here the appos child IS a bare
       named entity -- this should be split into a separate target.

    2. A genuine descriptive appositive, e.g. "Tim Cook, the CEO
       of Apple, spoke" -- "the CEO of Apple" describes the SAME
       entity as "Tim Cook", not a second one. Here the appos
       child is a longer descriptive phrase, not a bare entity --
       this should stay merged into the parent's own text, not be
       split into a second target.
    """

    return bool(token.ent_type_)


def _agent_pobj(
    agent_token: Any,
) -> Any:
    """
    Given the "agent" token (the preposition "by" in a passive
    construction, e.g. "...was acquired BY Elon Musk"), return its
    "pobj" child -- the actual noun phrase that is the real actor
    ("Elon Musk"), not the preposition itself.

    Falls back to the agent_token itself if no pobj child is found
    (shouldn't normally happen, but avoids crashing on unexpected
    parses).
    """

    for child in agent_token.children:
        if child.dep_ == "pobj":
            return child

    return agent_token


_WEAK_ENTITY_LABELS_FOR_PP = {
    "DATE", "TIME", "CARDINAL", "ORDINAL", "PERCENT", "QUANTITY",
}


def _find_pobj(
    prep_token: Any,
) -> Any | None:
    """
    Given a general "prep" token (any preposition, not just the
    passive "agent" case), return its "pobj" child if it has one,
    else None.

    Unlike _agent_pobj, this does NOT fall back to the prep token
    itself -- if the prep's child is "pcomp" instead (e.g. "BY
    flying to Mexico", where "flying" is a VERB, not a noun), we
    deliberately return None and let that VERB be picked up
    independently as its own event by the normal per-token loop in
    _extract_events/_extract_relations, rather than trying to
    recurse into it here.
    """

    for child in prep_token.children:
        if child.dep_ == "pobj":
            return child

    return None


def _contains_strong_entity(
    token: Any,
) -> bool:
    """
    True if `token`'s subtree contains a spaCy-recognized named
    entity that ISN'T a weak type (DATE, CARDINAL, etc.).

    Used to gate which prepositional-phrase objects we're willing
    to pull in as target candidates (see the "prep" branch in
    _extract_relations). PPs carry a lot of adjunct noise we don't
    want ("on a moment's notice", "during an interview") -- unlike
    our existing dobj/attr candidates, which we accept regardless
    of entity status, prep-derived candidates are gated behind
    "does this actually contain a real entity?" specifically to
    keep that noise out while still catching real cases like
    "flying TO MEXICO" or "AT 21ST CENTURY FOX, we...".
    """

    for descendant in token.subtree:
        if descendant.ent_type_ and descendant.ent_type_ not in _WEAK_ENTITY_LABELS_FOR_PP:
            return True

    return False


def _own_span_tokens(
    token: Any,
) -> list[Any]:
    """
    Return this token's subtree EXCLUDING anything that belongs to
    a coordinated conjunct (dep_ == "conj"), the "and"/"or" itself
    (dep_ == "cc"), an entity-like appositive (dep_ == "appos"
    where the appositive is itself a bare named entity -- see
    _is_entity_like), or a clausal modifier attached to this token
    (dep_ in {"relcl", "acl", "ccomp"}).

    Why the conj/cc/appos exclusions are needed: spaCy's
    token.subtree for the FIRST item in "Google and Meta" includes
    "and" and "Meta" as descendants of "Google" (since Meta
    attaches to Google via dep_="conj"). Without this exclusion, a
    naive subtree-text call would return "Google and Meta" as a
    single string instead of letting us split it into two separate
    targets.

    Why the clause exclusion is needed: relative clauses and
    clausal complements attached to a noun can be arbitrarily long
    and were being absorbed whole into a candidate's text. Example
    confirmed via real dataset generation: "any doubts based on the
    judge's opinions...THAT JUDGE GORSUCH WOULD BE A RELIABLE
    CONSERVATIVE COMMITTED TO FOLLOWING THE ORIGINAL UNDERSTANDING
    OF THOSE WHO DRAFTED AND RATIFIED THE CONSTITUTION" -- a
    32-word "target," unusable for sentiment labeling. Excluding
    relcl/acl/ccomp keeps the target itself short while the full
    sentence (always kept alongside target_text in dataset records)
    still provides the complete context for labeling -- nothing is
    lost from the labeling process, only from the target's own
    identifying text.
    """

    exclude_indices: set[int] = set()

    for child in token.children:

        if child.dep_ in {"conj", "cc"}:

            for descendant in child.subtree:
                exclude_indices.add(descendant.i)

        elif child.dep_ == "punct" and child.text in {",", ";"}:

            # Only strip COMMA/SEMICOLON list-separators (e.g. the
            # stray trailing comma in "Amazon, Google, and
            # Microsoft"). Do NOT exclude punctuation generally --
            # that was wrongly also stripping meaningful hyphens
            # ("11-year-old") and quote marks ("'Reid Rule'"),
            # which should stay in the text with correct spacing
            # via _subtree_text's whitespace-aware joining, not be
            # deleted outright.
            exclude_indices.add(child.i)
            for descendant in child.subtree:
                exclude_indices.add(descendant.i)

        elif child.dep_ == "appos" and _is_entity_like(child):

            for descendant in child.subtree:
                exclude_indices.add(descendant.i)

        elif child.dep_ in {"relcl", "acl", "ccomp"}:

            for descendant in child.subtree:
                exclude_indices.add(descendant.i)

    return [
        item
        for item in token.subtree
        if item.i not in exclude_indices
    ]


def _subtree_text(
    token: Any,
) -> str:
    """
    Return the text covered by a dependency subtree, EXCLUDING
    coordinated conjuncts and entity-like appositives (see
    _own_span_tokens). Use _with_conjuncts() when you want each
    conjunct/coordinate item as a separate string.

    Joining uses spaCy's OWN original whitespace information
    (token.whitespace_) rather than guessing per-punctuation-mark
    which characters need "no space before/after". For two tokens
    that are adjacent in the original sentence (consecutive token
    index) AND had no space between them originally, we join them
    directly with no space; otherwise we join with a single space.

    This single mechanism correctly handles every case we've hit
    so far without needing one-off special cases per punctuation
    type:
        - possessives: "China" + "'s" -> "China's" (not "China 's")
        - hyphenated compounds: "11" + "-" + "year" + "-" + "old"
          -> "11-year-old" (not "11 - year - old")
        - quotation marks: "'" + "Reid" + "Rule" + "'"
          -> "'Reid Rule'" (not "' Reid Rule '")

    Previously each of these was (or would have been) a separate
    ad-hoc fix checking for specific token text like "'s" -- fixing
    the general mechanism instead means future punctuation-spacing
    issues we haven't seen yet are also covered automatically.
    """

    tokens = _own_span_tokens(token)

    if not tokens:
        return token.text

    tokens.sort(
        key=lambda item: item.i
    )

    result = tokens[0].text
    prev = tokens[0]

    for item in tokens[1:]:

        is_adjacent_in_original = (item.i == prev.i + 1)

        # whitespace_ == "" means no space followed this token in
        # the ORIGINAL sentence. Default to " " (i.e. assume a
        # space) if the attribute is unavailable, so behavior stays
        # safe/unchanged for anything that doesn't provide it.
        prev_had_trailing_space = (
            getattr(prev, "whitespace_", " ") != ""
        )

        if is_adjacent_in_original and not prev_had_trailing_space:
            result += item.text
        else:
            result += " " + item.text

        prev = item

    return _clean_orphaned_punctuation(result)


_ORPHANED_EDGE_CHARS = "\u2014\u2013\\-:;,"  # em/en dash, plain hyphen, colon, semicolon, comma
_SELF_PAIRING_QUOTES = ("'", '"')
_OPEN_CLOSE_PAIRS = {
    "(": ")",
    "\u201c": "\u201d",  # “ ”
}


def _clean_orphaned_punctuation(text: str) -> str:
    """
    Strip dangling/orphaned punctuation left behind when a clause
    or appositive gets excluded from a candidate's own span (see
    _own_span_tokens), but the PAIRED PUNCTUATION that used to wrap
    that excluded content (parentheses, quotes, dashes) is a
    separate sibling token that doesn't get excluded along with it.

    Confirmed via real dataset generation output: excluding content
    like a party-affiliation aside "(R-Wis.)" or a parenthetical
    "-- something --" leaves the outer punctuation marks stranded,
    producing targets like "Paul Ryan ( )" (empty parens), "the
    first Indian American (" (a dangling UNCLOSED opener), "James
    O'Keefe's \u2014 \u2014" (dangling plain-ASCII dashes), or
    "...chairman \u201d" (a dangling closing quote with no matching
    opener anywhere).

    IMPORTANT: this must NOT strip a genuine matching pair that
    wraps real content, e.g. "'Reid Rule'" (both quotes present,
    real text between them) should stay exactly as-is -- only a
    truly UNPAIRED (orphaned) opener/closer gets removed, checked
    in BOTH directions (a stray opener can end up dangling at
    either edge, not just the "expected" one).
    """

    # Empty parens (with only whitespace inside) can appear even
    # when not right at the edge of the string.
    text = re.sub(r"\(\s*\)", "", text).strip()

    if not text:
        return text

    # Parens / curly quotes: strip an edge occurrence ONLY if its
    # counterpart is genuinely absent elsewhere in the text (i.e.
    # truly orphaned), never when both sides form a real pair.
    # Checked in both directions, since a stray opener or closer
    # can end up dangling at EITHER edge, not just its "natural" one
    # (e.g. "the first Indian American (" has an orphaned OPENER
    # at the trailing edge, not a closer).
    for opener, closer in _OPEN_CLOSE_PAIRS.items():

        if text.startswith(opener) and closer not in text[1:]:
            text = text[1:].lstrip()
        if text.endswith(closer) and opener not in text[:-1]:
            text = text[:-1].rstrip()

        if text.endswith(opener) and closer not in text[:-1]:
            text = text[:-1].rstrip()
        if text.startswith(closer) and opener not in text[1:]:
            text = text[1:].lstrip()

    # Straight quotes (' or ") self-pair: if the SAME character
    # appears at both edges of real content, that's a genuine
    # wrapped quotation -- keep both. Only strip if it appears at
    # just ONE edge (asymmetric = orphaned).
    for quote_char in _SELF_PAIRING_QUOTES:
        starts = text.startswith(quote_char)
        ends = text.endswith(quote_char)
        if starts and not ends:
            text = text[1:].lstrip()
        elif ends and not starts:
            text = text[:-1].rstrip()

    # Dashes/colon/semicolon/comma at the edges don't function as
    # paired wrappers in any case we've observed -- always strip
    # when they trail off the edge of the text.
    edge_pattern = (
        rf"^[\s{_ORPHANED_EDGE_CHARS}]+"
        rf"|[\s{_ORPHANED_EDGE_CHARS}]+$"
    )
    text = re.sub(edge_pattern, "", text)

    return text.strip()


def _with_conjuncts(
    token: Any,
) -> list[str]:
    """
    Return the (exclusion-aware) text of `token` PLUS the text of
    every token coordinated with it via:

      - dep_ == "conj"  (normal "and"/"or" coordination)
      - dep_ == "appos" WHERE the appositive is itself a bare
        named entity (handles spaCy's common mis-parse of
        comma-separated lists like "Amazon, Google, and Microsoft",
        where only the LAST item gets a clean "conj" label and
        earlier items get mislabeled as "appos")

    Followed recursively so chains like "A, B, and C" are all
    captured as separate strings, not just the first or last one.
    Descriptive appositives (e.g. "the CEO of Apple") are
    deliberately NOT split out, since they describe the same
    entity rather than introducing a new one.
    """

    results = [_subtree_text(token)]

    for child in token.children:

        if child.dep_ == "conj":
            results.extend(
                _with_conjuncts(child)
            )

        elif child.dep_ == "appos" and _is_entity_like(child):
            results.extend(
                _with_conjuncts(child)
            )

    return results


# ============================================================
# Simple command-line test
# ============================================================


def _print_result(
    result: SemanticRepresentation,
) -> None:

    print()
    print("=" * 70)
    print("SEMANTIC REPRESENTATION")
    print("=" * 70)

    print()
    print("Sentence:")
    print(result.sentence)

    print()
    print("-" * 70)
    print("ENTITIES")
    print("-" * 70)

    if result.entities:

        for entity in result.entities:

            print(
                f"  {entity.text:<35} "
                f"{entity.label}"
            )

    else:

        print("  None")

    print()
    print("-" * 70)
    print("NOUN PHRASES")
    print("-" * 70)

    if result.noun_phrases:

        for phrase in result.noun_phrases:

            print(
                f"  {phrase.text:<35} "
                f"root={phrase.root:<15} "
                f"POS={phrase.root_pos}"
            )

    else:

        print("  None")

    print()
    print("-" * 70)
    print("EVENTS")
    print("-" * 70)

    if result.events:

        for event in result.events:

            print(
                f"  {event.trigger:<25} "
                f"lemma={event.lemma}"
            )

    else:

        print("  None")

    print()
    print("-" * 70)
    print("RELATIONS")
    print("-" * 70)

    if result.relations:

        for relation in result.relations:

            subject = (
                relation.subject
                if relation.subject
                else "?"
            )

            obj = (
                relation.object
                if relation.object
                else "?"
            )

            print(
                f"  {subject} "
                f"--[{relation.relation}]--> "
                f"{obj}"
            )

    else:

        print("  None")

    print()
    print("-" * 70)
    print("TOKENS")
    print("-" * 70)

    for token in result.tokens:

        print(
            f"  {token.text:<20} "
            f"POS={token.pos:<8} "
            f"DEP={token.dep:<12} "
            f"HEAD={token.head}"
        )


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":

    extractor = SemanticExtractor()

    test_sentences = [

        "Microsoft acquired Activision Blizzard in a major technology deal.",

        "Apple CEO Tim Cook visited India this week.",

        "OpenAI announced a new partnership with Microsoft.",

        "Nvidia reported record quarterly revenue after strong demand for AI chips.",

        "The Federal Reserve kept interest rates unchanged on Wednesday.",
    ]

    for sentence in test_sentences:

        result = extractor.extract(
            sentence
        )

        _print_result(
            result
        )
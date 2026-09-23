"""
semantic/test_semantic_extractor.py
-----------------------------------

Test the semantic analysis layer on real-world style sentences.

This test does NOT:
    - use gold targets
    - train a model
    - rank candidates
    - evaluate against labels

It only checks whether the semantic extractor correctly identifies:

    1. Entities
    2. Noun phrases
    3. Events
    4. Subject/object relations
    5. Token dependencies
"""


from __future__ import annotations

import sys
from pathlib import Path


# ============================================================
# Make project root importable
#
# This file lives at <project_root>/semantic/test_semantic_extractor.py
# so the project root is one level up (parents[1], not parents[2]).
# Run this as: python -m semantic.test_semantic_extractor
# from the project root.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Import semantic extractor (package-qualified, matches
# test_target_selector.py's convention)
# ============================================================

from semantic.semantic_extractor import SemanticExtractor


# ============================================================
# Test sentences
# ============================================================

TEST_SENTENCES = [

    # -------------------------------------------------------    # 1. Company acquisition
    # --------------------------------------------------------

    "Microsoft acquired Activision Blizzard in a major technology deal.",

    # --------------------------------------------------------
    # 2. Political announcement
    # --------------------------------------------------------

    "President Trump announced a new economic policy today.",

    # --------------------------------------------------------
    # 3. Company leadership
    # --------------------------------------------------------

    "Microsoft CEO Satya Nadella spoke at the conference.",

    # --------------------------------------------------------
    # 4. Company + person
    # --------------------------------------------------------

    "Elon Musk announced a new plan for Tesla.",

    # --------------------------------------------------------
    # 5. Business
    # --------------------------------------------------------

    "Nvidia reported record quarterly revenue after strong demand for AI chips.",

    # --------------------------------------------------------
    # 6. Technology partnership
    # --------------------------------------------------------

    "OpenAI announced a new partnership with Microsoft.",

    # --------------------------------------------------------
    # 7. Product announcement
    # --------------------------------------------------------

    "Google introduced several new artificial intelligence products.",

    # --------------------------------------------------------
    # 8. Stock movement
    # --------------------------------------------------------

    "Tesla shares rose after the company reported stronger sales.",

    # --------------------------------------------------------
    # 9. Government
    # --------------------------------------------------------

    "The Federal Reserve kept interest rates unchanged on Wednesday.",

    # --------------------------------------------------------
    # 10. Geopolitics
    # --------------------------------------------------------

    "The United States imposed new sanctions on Russia.",

    # --------------------------------------------------------
    # 11. International meeting
    # --------------------------------------------------------

    "China's President Xi Jinping met with European Union officials.",

    # --------------------------------------------------------
    # 12. Investment
    # --------------------------------------------------------

    "Meta announced that it will invest billions of dollars in artificial intelligence.",

    # --------------------------------------------------------
    # 13. Multiple people
    # --------------------------------------------------------

    "Amazon founder Jeff Bezos met with Indian Prime Minister Narendra Modi.",

    # --------------------------------------------------------
    # 14. Competition
    # --------------------------------------------------------

    "Apple and Google are competing to develop new AI products.",

    # --------------------------------------------------------
    # 15. Commodity market
    # --------------------------------------------------------

    "Oil prices climbed after Saudi Arabia announced additional production cuts.",

    # --------------------------------------------------------
    # 16. Agreement
    # --------------------------------------------------------

    "Officials in Washington discussed the proposed agreement.",

    # --------------------------------------------------------
    # 17. Scientific / business event
    # --------------------------------------------------------

    "Pfizer reported positive results from its latest clinical trial.",

    # --------------------------------------------------------
    # 18. Financial market
    # --------------------------------------------------------

    "Investors sold technology stocks after the Federal Reserve raised interest rates.",

    # --------------------------------------------------------
    # 19. Product launch
    # --------------------------------------------------------

    "Apple launched a new iPhone model during its annual event.",

    # --------------------------------------------------------
    # 20. Corporate restructuring
    # --------------------------------------------------------

    "Google announced plans to cut thousands of jobs worldwide.",
]



# ============================================================
# Focused NER regression test
# ============================================================


def test_activision_blizzard_entity_text_preserved():
    """
    Regression test for spaCy's en_core_web_sm NER behavior.

    spaCy currently labels "Activision Blizzard" as PERSON in
    this sentence, even though the real-world entity is an
    organization.

    This test intentionally does NOT correct the label.

    It verifies that the semantic extractor preserves:
        1. the complete entity text;
        2. one entity span;
        3. correct character offsets.

    The current PERSON label is documented as an upstream
    spaCy NER limitation. No production hardcoded correction
    is introduced here.
    """

    sentence = (
        "Microsoft acquired Activision Blizzard "
        "in a major technology deal."
    )

    extractor = SemanticExtractor()

    result = extractor.extract(sentence)

    activision_entities = [
        entity
        for entity in result.entities
        if entity.text == "Activision Blizzard"
    ]

    # The complete entity must be preserved as one span.
    assert len(activision_entities) == 1, (
        "Expected exactly one entity with text "
        "'Activision Blizzard'."
    )

    entity = activision_entities[0]

    # Preserve the entity text.
    assert entity.text == "Activision Blizzard"

    # Document current spaCy behavior.
    #
    # This is NOT saying Activision Blizzard is a PERSON.
    # It documents the current en_core_web_sm NER output so
    # that we don't accidentally add a hardcoded correction
    # merely to make this test pass.
    assert entity.label == "PERSON", (
        "spaCy's current en_core_web_sm behavior changed. "
        "Review this regression test and the NER layer."
    )

    # Verify that the offsets still identify the complete span.
    assert (
        sentence[entity.start:entity.end]
        == "Activision Blizzard"
    )


# ============================================================
# Printing helpers
# ============================================================


def print_entities(result):
    print("\nENTITIES")
    print("-" * 70)

    if not result.entities:
        print("  None")
        return

    for entity in result.entities:

        print(
            f"  {entity.text!r:<35}"
            f"label={entity.label:<15}"
            f"offset={entity.start}-{entity.end}"
        )


def print_noun_phrases(result):
    print("\nNOUN PHRASES")
    print("-" * 70)

    if not result.noun_phrases:
        print("  None")
        return

    for phrase in result.noun_phrases:

        print(
            f"  {phrase.text!r:<35}"
            f"root={phrase.root:<20}"
            f"POS={phrase.root_pos:<8}"
            f"offset={phrase.start}-{phrase.end}"
        )


def print_events(result):
    print("\nEVENTS")
    print("-" * 70)

    if not result.events:
        print("  None")
        return

    for event in result.events:

        print(
            f"  trigger={event.trigger!r:<20}"
            f"lemma={event.lemma:<20}"
            f"POS={event.pos:<8}"
            f"offset={event.start}-{event.end}"
        )


def print_relations(result):
    print("\nRELATIONS")
    print("-" * 70)

    if not result.relations:
        print("  None")
        return

    for relation in result.relations:

        subject = relation.subject or "None"
        obj = relation.object or "None"

        print(
            f"  {subject!r}"
            f" --[{relation.relation}]--> "
            f"{obj!r}"
        )


def print_tokens(result):
    print("\nTOKENS / DEPENDENCIES")
    print("-" * 70)

    for token in result.tokens:

        print(
            f"  {token.text!r:<20}"
            f"lemma={token.lemma:<18}"
            f"POS={token.pos:<8}"
            f"dep={token.dep:<12}"
            f"head={token.head!r}"
        )


# ============================================================
# Main test
# ============================================================


def main():

    print("=" * 70)
    print("SEMANTIC EXTRACTOR REAL-WORLD TEST")
    print("=" * 70)

    print(
        "\nThis test does NOT use gold targets."
    )

    print(
        "We are checking whether the semantic "
        "representation makes sense."
    )

    print(
        "\nLoading semantic extractor..."
    )

    extractor = SemanticExtractor()

    print(
        "\nExtractor ready."
    )

    # ========================================================
    # Run examples
    # ========================================================

    for index, sentence in enumerate(
        TEST_SENTENCES,
        start=1,
    ):

        print("\n")
        print("=" * 70)
        print(f"EXAMPLE {index}")
        print("=" * 70)

        print("\nSENTENCE")
        print("-" * 70)
        print(sentence)

        try:

            result = extractor.extract(
                sentence
            )

            print_entities(result)

            print_noun_phrases(result)

            print_events(result)

            print_relations(result)

            # ------------------------------------------------
            # Token dependencies are useful during development
            # but can make the output very large.
            #
            # Uncomment the next line if we need detailed
            # dependency debugging.
            # ------------------------------------------------

            # print_tokens(result)

        except Exception as exc:

            print(
                "\nERROR:"
            )

            print(
                f"  {type(exc).__name__}: {exc}"
            )

    print("\n")
    print("=" * 70)
    print("SEMANTIC TEST COMPLETE")
    print("=" * 70)


# ============================================================
# Entry point
# ============================================================


if __name__ == "__main__":
    main()

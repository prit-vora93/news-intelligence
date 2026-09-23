"""
semantic/test_target_selector.py
--------------------------------

Real-world test for the semantic target selector.

No gold targets are used. The purpose is to inspect whether the
selector's semantic reasoning makes sense across a range of
sentence structures, not to score against a fixed answer key.

Uses select_all() (not the old single-target select()), since a
sentence can contain multiple independent targets -- that's the
whole point of NewsMTSC-style multi-target sentiment.

Sentences are grouped into categories (per the architecture review
doc, Section 13) so failures are easier to localize to a specific
pattern rather than lost in one flat list.
"""

from semantic.semantic_extractor import SemanticExtractor
from semantic.semantic_graph import SemanticGraphBuilder
from semantic.target_selector import TargetSelector


CATEGORIES: dict[str, list[str]] = {

    "Single-event sentences": [
        "Microsoft acquired Activision Blizzard.",
        "President Trump announced a new economic policy.",
        "Apple launched a new iPhone model.",
        "Nvidia reported record quarterly revenue.",
    ],

    "Multiple-event sentences": [
        "Google announced layoffs and Meta announced hiring.",
        "Oil prices climbed after Saudi Arabia announced production cuts.",
        "Investors sold technology stocks after the Federal Reserve raised interest rates.",
    ],

    "Multiple participants (coordination)": [
        "Apple and Google are competing to develop new AI products.",
        "The United States and the European Union imposed new sanctions on Russia.",
        "Amazon, Google, and Microsoft are all investing in AI infrastructure.",
        "The UK, France, and Germany agreed on a joint defense plan.",
    ],

    "Pronouns": [
        "Meta announced that it will invest billions of dollars in artificial intelligence.",
        "Tesla shares rose after the company reported stronger sales.",
    ],

    "Person + organization": [
        "Amazon founder Jeff Bezos met with Indian Prime Minister Narendra Modi.",
        "Microsoft CEO Satya Nadella spoke at the conference.",
        "China's President Xi Jinping met with European Union officials.",
    ],

    "Passive voice": [
        "Twitter was acquired by Elon Musk in 2022.",
        "Thousands of workers were laid off by Amazon last quarter.",
    ],

    "Negation": [
        "The Federal Reserve did not raise interest rates this month.",
        "Apple has not confirmed plans for a foldable iPhone.",
    ],

    "No obvious event (fallback behavior)": [
        "Inflation.",
        "A strong quarter for the technology sector.",
        "Uncertainty in global markets.",
    ],
}


def print_targets(targets) -> None:

    if not targets:
        print("  NO TARGETS FOUND")
        return

    for t in targets:
        print(
            f"  target={t.target!r:35} "
            f"role={t.role!s:8} "
            f"event={t.event!s:15} "
            f"confidence={t.confidence}"
        )
        print(f"      reason:   {t.reason}")
        print(f"      evidence: {t.evidence}")
        if t.embedded_entities:
            print(f"      embedded_entities: {t.embedded_entities}")


def main() -> None:

    print("=" * 70)
    print("SEMANTIC TARGET SELECTOR REAL-WORLD TEST (select_all)")
    print("=" * 70)
    print()
    print("No gold targets are provided.")
    print("The selector must reason from the semantic graph.")

    extractor = SemanticExtractor()
    graph_builder = SemanticGraphBuilder()
    selector = TargetSelector()

    total_sentences = 0
    total_no_targets = 0

    for category, sentences in CATEGORIES.items():

        print()
        print("#" * 70)
        print(f"CATEGORY: {category}")
        print("#" * 70)

        for sentence in sentences:

            total_sentences += 1

            print()
            print("-" * 70)
            print(f"SENTENCE: {sentence}")
            print("-" * 70)

            representation = extractor.extract(sentence)
            graph = graph_builder.build(representation)

            print()
            print("SEMANTIC EVENTS")
            for event in graph.events:
                print(
                    f"  {event.trigger!r:15}"
                    f" subjects={event.subjects}"
                    f" objects={event.objects}"
                )
            if not graph.events:
                print("  None")

            print()
            print("ENTITIES")
            for entity in graph.entities:
                print(f"  {entity.text!r:30} label={entity.label}")
            if not graph.entities:
                print("  None")

            print()
            print("TARGETS (select_all)")
            targets = selector.select_all(graph)
            print_targets(targets)

            if not targets:
                total_no_targets += 1

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total sentences tested: {total_sentences}")
    print(f"Sentences with NO targets found: {total_no_targets}")
    print()
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
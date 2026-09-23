"""
semantic/test_entity_resolver_batch2.py
-----------------------------------------

Broader, more diverse test batch for entity_resolver.py. Covers
pronoun categories we haven't tried yet (he/him/his, she/her,
its), multiple generic-description categories (the firm, the
government, the country), a deliberate type-mismatch case (to
confirm the type filter actually PREVENTS a wrong resolution, not
just allows correct ones), and a couple of realistic multi-clause
news sentences.
"""

from semantic.semantic_extractor import SemanticExtractor
from semantic.semantic_graph import SemanticGraphBuilder
from semantic.target_selector import TargetSelector
from semantic.entity_resolver import EntityResolver


CATEGORIES: dict[str, list[str]] = {

    "Male pronouns (he/him/his)": [
        "Tim Cook unveiled the new iPhone. He called it a major milestone.",
        "Elon Musk sold his stake in the company.",
    ],

    "Female pronouns (she/her)": [
        "Christine Lagarde spoke at the summit. She warned of rising inflation.",
    ],

    "Its (organization possessive)": [
        "Apple raised its prices across all markets.",
    ],

    "Definite descriptions beyond 'the company'": [
        "France announced new tariffs. The government said the move was necessary.",
        "Japan's economy grew last quarter. The country's exports also rose.",
        "Pfizer released trial results. The firm said the drug showed strong efficacy.",
    ],

    "Deliberate type mismatch (should stay UNRESOLVED, not guess wrong)": [
        "Tim Cook spoke at the event. It later posted record profits.",
    ],

    "Multi-hop within one sentence (three clauses, same referent)": [
        "Nvidia reported earnings, and it also raised its own forecast for the year.",
    ],

    "Realistic multi-entity news sentences": [
        "Samsung unveiled a new chip design. The company said it would begin mass production next year.",
        "The European Central Bank raised rates. It cited persistent inflation pressures.",
    ],
}


def main() -> None:

    extractor = SemanticExtractor()
    graph_builder = SemanticGraphBuilder()
    selector = TargetSelector()
    resolver = EntityResolver()

    for category, sentences in CATEGORIES.items():

        print()
        print("#" * 70)
        print(f"CATEGORY: {category}")
        print("#" * 70)

        for sentence in sentences:

            print()
            print("-" * 70)
            print(f"SENTENCE: {sentence}")
            print("-" * 70)

            representation = extractor.extract(sentence)
            graph = graph_builder.build(representation)
            targets = selector.select_all(graph)

            result = resolver.resolve([graph], [targets])

            print("TARGETS:", [(t.target, t.role) for t in targets])

            print("RESOLVED:")
            if not result.resolved_entities:
                print("  None")
            for e in result.resolved_entities:
                mentions = [f"{m.text!r}({m.mention_type})" for m in e.mentions]
                print(f"  {e.canonical_name} ({e.entity_type}) <- {mentions}")

            print("UNRESOLVED:", [m.text for m in result.unresolved_mentions] or "None")

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

# """
# semantic/test_entity_resolver.py
# --------------------------------

# Real-world tests for EntityResolver.

# Focus:
#     - named entities
#     - within-sentence pronouns
#     - within-sentence definite descriptions
#     - unresolved references
# """

# from semantic.entity_resolver import EntityResolver
# from semantic.semantic_extractor import SemanticExtractor
# from semantic.semantic_graph import SemanticGraphBuilder
# from semantic.target_selector import TargetSelector


# TEST_SENTENCES = [
#     "Microsoft acquired Activision Blizzard.",
#     "Meta announced that it will invest billions of dollars in artificial intelligence.",
#     "Tesla shares rose after the company reported stronger sales.",
#     "Apple has not confirmed plans for a foldable iPhone.",
#     "Twitter was acquired by Elon Musk in 2022.",
# ]


# def build_inputs(sentence: str):
#     extractor = SemanticExtractor()
#     graph_builder = SemanticGraphBuilder()
#     selector = TargetSelector()

#     representation = extractor.extract(sentence)
#     graph = graph_builder.build(representation)
#     targets = selector.select_all(graph)

#     return graph, targets


# def main():
#     resolver = EntityResolver()

#     print("=" * 70)
#     print("ENTITY RESOLVER REAL-WORLD TEST")
#     print("=" * 70)

#     for sentence in TEST_SENTENCES:

#         print("\n" + "-" * 70)
#         print(f"SENTENCE: {sentence}")
#         print("-" * 70)

#         graph, targets = build_inputs(sentence)

#         result = resolver.resolve(
#             sentence_graphs=[graph],
#             target_lists=[targets],
#         )

#         print("\nRESOLVED ENTITIES")

#         for entity in result.resolved_entities:

#             print(
#                 f"  canonical='{entity.canonical_name}' "
#                 f"type={entity.entity_type}"
#             )

#             for mention in entity.mentions:
#                 print(
#                     f"      mention='{mention.text}' "
#                     f"type={mention.mention_type} "
#                     f"span=({mention.start}, {mention.end})"
#                 )

#         print("\nUNRESOLVED MENTIONS")

#         if not result.unresolved_mentions:
#             print("  None")
#         else:
#             for mention in result.unresolved_mentions:
#                 print(
#                     f"  '{mention.text}' "
#                     f"type={mention.mention_type} "
#                     f"span=({mention.start}, {mention.end})"
#                 )

#     print("\n" + "=" * 70)
#     print("TEST COMPLETE")
#     print("=" * 70)


# if __name__ == "__main__":
#     main()


#----------------------------------claude-----------------------------------

"""
semantic/test_entity_resolver.py
---------------------------------

Real-world test for entity_resolver.py. Runs the full pipeline
(extractor -> graph -> target_selector -> entity_resolver) on
sentences that specifically contain pronouns or generic definite
descriptions, since those are the only things this component acts
on.
"""

from semantic.semantic_extractor import SemanticExtractor
from semantic.semantic_graph import SemanticGraphBuilder
from semantic.target_selector import TargetSelector
from semantic.entity_resolver import EntityResolver


SENTENCES = [
    "Meta announced that it will invest billions of dollars in artificial intelligence.",
    "Tesla shares rose after the company reported stronger sales.",
    "Investors sold Tesla shares and it fell sharply.",
    "Amazon reported earnings and they raised prices.",
    "Microsoft acquired Activision Blizzard.",  # no pronouns -- should resolve nothing
    "Google announced layoffs and Meta announced hiring.",  # no pronouns -- should resolve nothing
]


def main() -> None:

    extractor = SemanticExtractor()
    graph_builder = SemanticGraphBuilder()
    selector = TargetSelector()
    resolver = EntityResolver()

    for index, sentence in enumerate(SENTENCES, start=1):

        print()
        print("=" * 70)
        print(f"[{index}] {sentence}")
        print("=" * 70)

        representation = extractor.extract(sentence)
        graph = graph_builder.build(representation)
        targets = selector.select_all(graph)

        result = resolver.resolve([graph], [targets])

        print()
        print("TARGETS (from select_all, for reference)")
        for t in targets:
            print(f"  {t.target!r} (role={t.role})")

        print()
        print("RESOLVED ENTITIES")
        if not result.resolved_entities:
            print("  None")
        for e in result.resolved_entities:
            print(f"  {e.canonical_name} ({e.entity_type})")
            for m in e.mentions:
                print(f"    <- {m.text!r} ({m.mention_type})")

        print()
        print("UNRESOLVED MENTIONS")
        if not result.unresolved_mentions:
            print("  None")
        for m in result.unresolved_mentions:
            print(f"  {m.text!r} ({m.mention_type})")

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
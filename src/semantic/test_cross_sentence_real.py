from semantic.semantic_extractor import SemanticExtractor
from semantic.semantic_graph import SemanticGraphBuilder
from semantic.target_selector import TargetSelector
from semantic.entity_resolver import EntityResolver

extractor = SemanticExtractor()
builder = SemanticGraphBuilder()
selector = TargetSelector()
resolver = EntityResolver()

# A tiny 2-sentence "article" -- sentence 2's "He" should resolve
# back to "Marco Rubio" from sentence 1.
article_sentences = [
    "Marco Rubio criticized the tax reform bill on Tuesday.",
    "He said it would hurt working families.",
]

graphs = []
target_lists = []

for sentence in article_sentences:
    representation = extractor.extract(sentence)
    graph = builder.build(representation)
    targets = selector.select_all(graph)
    graphs.append(graph)
    target_lists.append(targets)
    print(f"SENTENCE: {sentence}")
    print(f"  targets: {[(t.target, t.role) for t in targets]}")

result = resolver.resolve(graphs, target_lists)

print()
print("RESOLVED ENTITIES:")
for e in result.resolved_entities:
    print(f"  {e.canonical_name} ({e.entity_type}):")
    for m in e.mentions:
        print(f"    <- {m.text!r} (sentence_index={m.sentence_index})")

print()
print("UNRESOLVED MENTIONS:")
for m in result.unresolved_mentions:
    print(f"  {m.text!r} (sentence_index={m.sentence_index})")

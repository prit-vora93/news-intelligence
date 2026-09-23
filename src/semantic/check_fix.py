from semantic.semantic_extractor import SemanticExtractor
from semantic.semantic_graph import SemanticGraphBuilder
from semantic.target_selector import TargetSelector

extractor = SemanticExtractor()
builder = SemanticGraphBuilder()
selector = TargetSelector()

sentence = "Google announced layoffs and Meta announced hiring."
rep = extractor.extract(sentence)
graph = builder.build(rep)

print("EVENTS:")
for e in graph.events:
    print(f"  {e.trigger!r} subjects={e.subjects} objects={e.objects}")

print()
print("select_all():")
for r in selector.select_all(graph):
    print(f"  target={r.target!r} role={r.role} confidence={r.confidence}")

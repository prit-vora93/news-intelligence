import json
from collections import defaultdict

from semantic.semantic_extractor import SemanticExtractor
from semantic.semantic_graph import SemanticGraphBuilder
from semantic.target_selector import TargetSelector
from semantic.entity_resolver import EntityResolver


def parse_gid(gid: str):
    """
    Parse a primary_gid into (article_id, sentence_index, mention_text, char_from, char_to).

    Format confirmed via real data:
        {source}_{article_id_part1}_{article_id_part2}_{sentence_index}_{mention_text}_{char_from}_{char_to}

    The mention_text portion may itself contain spaces (never literal
    underscores in practice), so it's whatever remains between the
    first 3 fields and the last 2 fields.
    """

    parts = gid.split("_")

    if len(parts) < 6:
        return None

    article_id = "_".join(parts[0:3])

    try:
        sentence_index = int(parts[3])
        char_from = int(parts[-2])
        char_to = int(parts[-1])
    except ValueError:
        return None

    mention_text = "_".join(parts[4:-2])

    return article_id, sentence_index, mention_text, char_from, char_to


def load_articles(path: str) -> dict[str, dict[int, dict]]:
    """
    Returns: {article_id: {sentence_index: {"text": str, "targets": [...]}}}
    """

    articles: dict[str, dict[int, dict]] = defaultdict(dict)

    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            parsed = parse_gid(record["primary_gid"])
            if parsed is None:
                continue

            article_id, sentence_index, _, _, _ = parsed
            sentence_text = record["sentence_normalized"]

            if sentence_index not in articles[article_id]:
                articles[article_id][sentence_index] = {
                    "text": sentence_text,
                    "targets": [],
                }

            articles[article_id][sentence_index]["targets"].extend(
                record.get("targets", [])
            )

    return articles


# ============================================================
# Pick a real multi-sentence article and run the real pipeline
# ============================================================

DATASET_PATH = "NewsMTSC/NewsSentiment/controller_data/datasets/NewsMTSC-dataset/train.jsonl"

articles = load_articles(DATASET_PATH)

# The Jared Kushner / Ivanka Trump article -- 4 different sentence
# indices (16, 24, 55, 8), a genuinely richer real multi-sentence case.
target_article_id = "allsides_1044_417"

sentences_in_order = sorted(articles[target_article_id].items())

print(f"Article: {target_article_id}")
print(f"Sentences found (sparse indices): {[idx for idx, _ in sentences_in_order]}")
print()

extractor = SemanticExtractor()
builder = SemanticGraphBuilder()
selector = TargetSelector()
resolver = EntityResolver()

graphs = []
target_lists = []

for sentence_index, data in sentences_in_order:
    sentence = data["text"]
    print(f"[idx {sentence_index}] {sentence}")

    representation = extractor.extract(sentence)
    graph = builder.build(representation)
    targets = selector.select_all(graph)

    graphs.append(graph)
    target_lists.append(targets)

result = resolver.resolve(graphs, target_lists)

print()
print("RESOLVED ENTITIES:")
for e in result.resolved_entities:
    print(f"  {e.canonical_name} ({e.entity_type}):")
    for m in e.mentions:
        print(f"    <- {m.text!r} (sentence_index in this run={m.sentence_index})")

print()
print("UNRESOLVED MENTIONS:")
for m in result.unresolved_mentions:
    print(f"  {m.text!r} (sentence_index in this run={m.sentence_index})")
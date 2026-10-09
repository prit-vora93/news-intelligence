import json

records = [json.loads(line) for line in open("dataset_articles_final.jsonl", encoding="utf-8")]

cross_sentence_examples = []

for r in records:
    sentence = r["sentence"]
    for mention in r.get("resolved_mentions", []):
        # If a resolved mention's text doesn't appear in THIS record's
        # own sentence, it almost certainly came from a different
        # sentence in the same article -- a real cross-sentence link.
        if mention.lower() not in sentence.lower():
            cross_sentence_examples.append((r["article_id"], r["sentence_index"], r["target_text"], mention, sentence))

print(f"Total records: {len(records)}")
print(f"Likely cross-sentence resolutions found: {len(cross_sentence_examples)}")
print()

for article_id, sentence_index, target_text, mention, sentence in cross_sentence_examples[:15]:
    print(f"article={article_id}  sentence_index={sentence_index}")
    print(f"  target={target_text!r}  <- resolved mention {mention!r} (not found in this sentence)")
    print(f"  sentence: {sentence[:100]}")
    print()
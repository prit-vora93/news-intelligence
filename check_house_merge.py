import json

records = [json.loads(line) for line in open("dataset_final.jsonl", encoding="utf-8")]

def canonical_entity(record):
    embedded = record.get("embedded_entities") or []
    if embedded:
        return embedded[0].strip()
    text = (record.get("target_text") or "").strip()
    words = text.split()
    if 1 <= len(words) <= 3 and text[:1].isupper():
        return text
    return None

house_records = []
for r in records:
    entity = canonical_entity(r)
    if entity and entity.lower() in ("house", "white house"):
        house_records.append((entity, r["target_text"]))

from collections import Counter
counts = Counter(e for e, _ in house_records)
print("Breakdown of what feeds into 'house' vs 'white house' buckets BEFORE merging:")
for entity, count in counts.most_common():
    print(f"  {entity!r}: {count}")

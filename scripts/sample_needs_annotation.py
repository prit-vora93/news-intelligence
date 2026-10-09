import json
import random

records = [json.loads(line) for line in open("dataset_v1.jsonl", encoding="utf-8")]
needs = [r for r in records if r["label_source"] == "needs_annotation"]

print(f"Total needs_annotation records: {len(needs)}")
print()

sample = random.sample(needs, min(15, len(needs)))

for r in sample:
    print(f"target={r['target_text']!r:35} role={r['role']:8} conf={r['confidence']}")
    print(f"  sentence: {r['sentence'][:120]}")
    print()

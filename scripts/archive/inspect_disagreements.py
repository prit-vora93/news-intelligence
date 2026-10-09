import json
import random

random.seed(7)

records = [json.loads(line) for line in open("dataset_final.jsonl", encoding="utf-8")]
disagreements = [r for r in records if r.get("sentiment_agreement") is False]

print(f"Total disagreements: {len(disagreements)}")
print()

sample = random.sample(disagreements, min(15, len(disagreements)))

for r in sample:
    print(f"target={r['target_text']!r}")
    print(f"  gold={r['sentiment_label']}  predicted={r['predicted_sentiment_label']} ({r['predicted_confidence']:.1%} confidence)")
    print(f"  sentence: {r['sentence'][:140]}")
    print()

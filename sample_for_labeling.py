"""
sample_for_labeling.py
-----------------------

Pull a representative sample of needs_annotation records from
dataset_v1.jsonl for a manual labeling trial, stratified across
(role, confidence) combinations so the sample isn't accidentally
biased toward one pattern (e.g. all high-confidence subjects).

Outputs a CSV (easier to hand-label in a spreadsheet than raw JSON)
with a blank sentiment_label column for you to fill in.

Usage:
    python3 sample_for_labeling.py
"""

import csv
import json
import random
from collections import defaultdict

random.seed(1337)  # different seed -- fresh sample for round 2 validation

SAMPLE_SIZE = 40

records = [json.loads(line) for line in open("dataset_v1.jsonl", encoding="utf-8")]
needs = [r for r in records if r["label_source"] == "needs_annotation"]

print(f"Total needs_annotation records: {len(needs)}")

# Group by (role, confidence) bucket
buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
for r in needs:
    buckets[(r["role"], r["confidence"])].append(r)

print(f"Number of distinct (role, confidence) buckets: {len(buckets)}")
for key, items in sorted(buckets.items()):
    print(f"  {key}: {len(items)} records")

# Sample roughly evenly across buckets, capped at SAMPLE_SIZE total
per_bucket = max(1, SAMPLE_SIZE // len(buckets))
sample = []
for key, items in buckets.items():
    random.shuffle(items)
    sample.extend(items[:per_bucket])

random.shuffle(sample)
sample = sample[:SAMPLE_SIZE]

print()
print(f"Sampled {len(sample)} records across {len(buckets)} buckets")

output_path = "labeling_trial_2.csv"
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "sentence_id", "sentence", "target_text", "role", "confidence",
        "embedded_entities", "resolved_mentions", "sentiment_label"
    ])
    for r in sample:
        writer.writerow([
            r["sentence_id"],
            r["sentence"],
            r["target_text"],
            r["role"],
            r["confidence"],
            "; ".join(r["embedded_entities"]),
            "; ".join(r["resolved_mentions"]),
            "",  # blank for you to fill in
        ])

print(f"Wrote {output_path}")
print()
print("Open this in a spreadsheet (LibreOffice Calc, Excel, Google Sheets)")
print("and fill in the sentiment_label column. Use exactly these values")
print("(confirmed against NewsMTSC's SentimentClasses.py):")
print("  2.0 = negative   4.0 = neutral   6.0 = positive")
print()
print("This time, specifically watch for: any remaining relative pronouns")
print("(who/which/what/whom/whose), dangling possessive spacing, or any")
print("target still feeling too long/unwieldy to label confidently.")

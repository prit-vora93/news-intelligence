import json

records = [json.loads(line) for line in open("dataset_v1.jsonl", encoding="utf-8")]

lengths = [len(r["target_text"].split()) for r in records]
lengths.sort()

print(f"Total records: {len(records)}")
print(f"Target word-count -- min={lengths[0]}, median={lengths[len(lengths)//2]}, "
      f"max={lengths[-1]}, mean={sum(lengths)/len(lengths):.1f}")

print()
print("Longest 10 targets (should be much shorter than before):")
by_length = sorted(records, key=lambda r: len(r["target_text"].split()), reverse=True)
for r in by_length[:10]:
    wc = len(r["target_text"].split())
    print(f"  ({wc} words) {r['target_text']!r}")

print()
print("=== Checking the two specific problem sentences from before ===")
for r in records:
    if "Gorsuch would be a reliable conservative" in r["sentence"]:
        print(f"  target={r['target_text']!r} ({len(r['target_text'].split())} words)")
    if "recreate a common space" in r["sentence"]:
        print(f"  target={r['target_text']!r} ({len(r['target_text'].split())} words)")

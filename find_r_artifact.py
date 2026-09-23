import json

records = [json.loads(line) for line in open("dataset_final.jsonl", encoding="utf-8")]

matches = [r for r in records if r.get("target_text", "").strip() == "R"]

print(f"Total records where target_text is exactly 'R': {len(matches)}")
print()

for r in matches[:8]:
    print(f"role={r['role']}  event={r.get('event')}  confidence={r['confidence']}")
    print(f"  sentence: {r['sentence']}")
    print()

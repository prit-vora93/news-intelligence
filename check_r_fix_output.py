import json

for line in open("r_fix_test_output.jsonl", encoding="utf-8"):
    r = json.loads(line)
    print(f"target={r['target_text']!r:35} embedded_entities={r['embedded_entities']}")

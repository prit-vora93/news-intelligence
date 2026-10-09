import json

records = [json.loads(line) for line in open("dataset_v1.jsonl", encoding="utf-8")]

for r in records:
    if r["target_text"] == "Hillary 's":
        print("FULL SENTENCE:")
        print(r["sentence"])
        print()
        print("Full record:")
        print(json.dumps(r, indent=2))
        break

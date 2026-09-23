import json

path = "NewsMTSC/NewsSentiment/controller_data/datasets/NewsMTSC-dataset/train.jsonl"

with open(path, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 30:
            break
        r = json.loads(line)
        print(f"{r['primary_gid']!r:70} | {r['sentence_normalized'][:70]}")

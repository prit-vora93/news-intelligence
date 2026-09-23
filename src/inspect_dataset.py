import json
from collections import Counter
from pathlib import Path


DATA_DIR = Path(
    "NewsMTSC/NewsSentiment/experiments/default/"
    "datasets/newsmtsc-rw"
)

TRAIN_PATH = DATA_DIR / "train.jsonl"
DEV_PATH = DATA_DIR / "dev.jsonl"
TEST_PATH = DATA_DIR / "test.jsonl"


def inspect_split(path):
    sentence_count = 0
    target_count = 0
    polarity_counts = Counter()

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            sentence_count += 1
            target_count += len(record["targets"])

            for target in record["targets"]:
                polarity_counts[target["polarity"]] += 1

    return sentence_count, target_count, polarity_counts


for split_name, split_path in [
    ("Train", TRAIN_PATH),
    ("Validation", DEV_PATH),
    ("Test", TEST_PATH),
]:
    sentences, targets, polarities = inspect_split(split_path)

    print(f"\n{split_name}")
    print("-" * len(split_name))
    print(f"Sentences: {sentences}")
    print(f"Targets:   {targets}")

    for polarity, count in sorted(polarities.items()):
        percentage = (count / targets) * 100
        print(
            f"Polarity {polarity}: "
            f"{count} ({percentage:.2f}%)"
        )
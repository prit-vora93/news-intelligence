import json
from pathlib import Path


DATA_DIR = Path(
    "NewsMTSC/NewsSentiment/experiments/default/"
    "datasets/newsmtsc-rw"
)

TRAIN_PATH = DATA_DIR / "train.jsonl"
DEV_PATH = DATA_DIR / "dev.jsonl"
TEST_PATH = DATA_DIR / "test.jsonl"


def load_sentences(path):
    sentences = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            sentences.append(record["sentence_normalized"])

    return sentences

def load_sentence_targets(path):
    examples = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            sentence = record["sentence_normalized"]

            for target in record["targets"]:
                mention = target["mention"]

                examples.append((sentence, mention))

    return examples

def load_target_labels(path):
    examples = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            sentence = record["sentence_normalized"]

            for target in record["targets"]:
                mention = target["mention"]
                polarity = target["polarity"]

                examples.append((sentence, mention, polarity))

    return examples

train_sentences = load_sentences(TRAIN_PATH)
dev_sentences = load_sentences(DEV_PATH)
test_sentences = load_sentences(TEST_PATH)

train_targets = load_sentence_targets(TRAIN_PATH)
dev_targets = load_sentence_targets(DEV_PATH)
test_targets = load_sentence_targets(TEST_PATH)

train_labeled = load_target_labels(TRAIN_PATH)
test_labeled = load_target_labels(TEST_PATH)

train_set = set(train_sentences)
dev_set = set(dev_sentences)
test_set = set(test_sentences)

train_target_set = set(train_targets)
dev_target_set = set(dev_targets)
test_target_set = set(test_targets)

train_labels = {
    (sentence, target): polarity
    for sentence, target, polarity in train_labeled
}

test_labels = {
    (sentence, target): polarity
    for sentence, target, polarity in test_labeled
}

train_dev_overlap = train_set & dev_set
train_test_overlap = train_set & test_set
dev_test_overlap = dev_set & test_set


print("Exact Sentence Overlap")
print("----------------------")
print(f"Train ∩ Validation: {len(train_dev_overlap)}")
print(f"Train ∩ Test:       {len(train_test_overlap)}")
print(f"Validation ∩ Test:  {len(dev_test_overlap)}")

print("\nTrain-Test Duplicate Sentences")
print("------------------------------")

# for sentence in sorted(train_test_overlap):
#     print(sentence)


train_dev_target_overlap = train_target_set & dev_target_set
train_test_target_overlap = train_target_set & test_target_set
dev_test_target_overlap = dev_target_set & test_target_set

print("\nExact Sentence + Target Overlap")
print("--------------------------------")

print(
    f"Train ∩ Validation: "
    f"{len(train_dev_target_overlap)}"
)

print(
    f"Train ∩ Test:       "
    f"{len(train_test_target_overlap)}"
)

print(
    f"Validation ∩ Test:  "
    f"{len(dev_test_target_overlap)}"
)

print("\nTrain-Test Exact Target Duplicates")
print("----------------------------------")

for sentence, target in sorted(train_test_target_overlap):
    print(f"Sentence: {sentence}")
    print(f"Target:   {target}")

print("\nPolarity of Train-Test Duplicate")
print("--------------------------------")

for sentence, target in sorted(train_test_target_overlap):
    train_polarity = train_labels[(sentence, target)]
    test_polarity = test_labels[(sentence, target)]

    print(f"Sentence:        {sentence}")
    print(f"Target:          {target}")
    print(f"Train polarity:  {train_polarity}")
    print(f"Test polarity:   {test_polarity}")
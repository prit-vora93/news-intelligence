import pandas as pd
import spacy


DATA_PATH = "data/processed/test_target.csv"

nlp = spacy.load("en_core_web_sm")

df = pd.read_csv(DATA_PATH)


def get_spacy_entities(sentence):
    doc = nlp(sentence)

    return [
        {
            "text": ent.text,
            "start": ent.start_char,
            "end": ent.end_char,
            "label": ent.label_,
        }
        for ent in doc.ents
    ]


exact = 0
partial = 0
missed = 0

examples = {
    "exact": [],
    "partial": [],
    "missed": [],
}


for _, row in df.iterrows():

    sentence = row["sentence"]

    target = row["target"]

    target_start = int(row["target_from"])
    target_end = int(row["target_to"])

    entities = get_spacy_entities(sentence)

    found_exact = False
    found_partial = False

    for entity in entities:

        entity_start = entity["start"]
        entity_end = entity["end"]

        # Exact character-span match
        if (
            entity_start == target_start
            and entity_end == target_end
        ):
            found_exact = True

            if len(examples["exact"]) < 5:
                examples["exact"].append(
                    {
                        "target": target,
                        "entity": entity,
                    }
                )

            break

        # Partial overlap
        overlap = (
            entity_start < target_end
            and entity_end > target_start
        )

        if overlap:
            found_partial = True

            if len(examples["partial"]) < 5:
                examples["partial"].append(
                    {
                        "target": target,
                        "entity": entity,
                    }
                )

    if found_exact:
        exact += 1

    elif found_partial:
        partial += 1

    else:
        missed += 1


total = len(df)


print("\nNER Evaluation")
print("=" * 60)

print(f"Total targets: {total}")

print(
    f"Exact matches: "
    f"{exact} "
    f"({exact / total * 100:.2f}%)"
)

print(
    f"Partial matches: "
    f"{partial} "
    f"({partial / total * 100:.2f}%)"
)

print(
    f"Missed: "
    f"{missed} "
    f"({missed / total * 100:.2f}%)"
)


print("\nExact Examples")
print("-" * 60)

for example in examples["exact"]:
    print(example)


print("\nPartial Examples")
print("-" * 60)

for example in examples["partial"]:
    print(example)


print("\nMissed Examples")
print("-" * 60)

for example in examples["missed"]:
    print(example)
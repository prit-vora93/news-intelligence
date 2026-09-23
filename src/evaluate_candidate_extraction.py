import pandas as pd
import spacy


DATA_PATH = "data/processed/test_target.csv"

nlp = spacy.load("en_core_web_sm")

df = pd.read_csv(DATA_PATH)


PRONOUNS = {
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
}


def get_candidates(sentence):
    """
    Generate candidate targets using:

    1. spaCy named entities
    2. spaCy noun chunks
    3. Pronouns
    """

    doc = nlp(sentence)

    candidates = []

    # --------------------------------------------------
    # 1. Named entities
    # --------------------------------------------------

    for ent in doc.ents:

        candidates.append({
            "text": ent.text,
            "start": ent.start_char,
            "end": ent.end_char,
            "source": "NER",
            "label": ent.label_,
        })

    # --------------------------------------------------
    # 2. Noun chunks
    # --------------------------------------------------

    for chunk in doc.noun_chunks:

        candidates.append({
            "text": chunk.text,
            "start": chunk.start_char,
            "end": chunk.end_char,
            "source": "NOUN_CHUNK",
            "label": None,
        })

    # --------------------------------------------------
    # 3. Pronouns
    # --------------------------------------------------

    for token in doc:

        if (
            token.text.lower() in PRONOUNS
            and token.pos_ == "PRON"
        ):

            candidates.append({
                "text": token.text,
                "start": token.idx,
                "end": token.idx + len(token.text),
                "source": "PRONOUN",
                "label": None,
            })

    return candidates


def exact_match(
    candidate,
    target_start,
    target_end,
):
    return (
        candidate["start"] == target_start
        and candidate["end"] == target_end
    )


def partial_match(
    candidate,
    target_start,
    target_end,
):
    return (
        candidate["start"] < target_end
        and candidate["end"] > target_start
    )


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

ner_exact = 0
ner_plus_noun_exact = 0
all_candidates_exact = 0

ner_candidate_recall = 0
ner_plus_noun_recall = 0
all_candidate_recall = 0

examples = {
    "ner_only": [],
    "ner_noun": [],
    "all": [],
}


for _, row in df.iterrows():

    sentence = row["sentence"]

    target = row["target"]

    target_start = int(row["target_from"])
    target_end = int(row["target_to"])

    candidates = get_candidates(sentence)

    ner_candidates = [
        c
        for c in candidates
        if c["source"] == "NER"
    ]

    ner_noun_candidates = [
        c
        for c in candidates
        if c["source"] in {
            "NER",
            "NOUN_CHUNK",
        }
    ]

    all_candidates = candidates

    # --------------------------------------------------
    # NER only
    # --------------------------------------------------

    ner_found = any(
        exact_match(
            c,
            target_start,
            target_end,
        )
        for c in ner_candidates
    )

    if ner_found:
        ner_exact += 1
        ner_candidate_recall += 1

    # --------------------------------------------------
    # NER + noun chunks
    # --------------------------------------------------

    ner_noun_found = any(
        exact_match(
            c,
            target_start,
            target_end,
        )
        for c in ner_noun_candidates
    )

    if ner_noun_found:
        ner_plus_noun_exact += 1
        ner_plus_noun_recall += 1

    # --------------------------------------------------
    # NER + noun chunks + pronouns
    # --------------------------------------------------

    all_found = any(
        exact_match(
            c,
            target_start,
            target_end,
        )
        for c in all_candidates
    )

    if all_found:
        all_candidates_exact += 1
        all_candidate_recall += 1

    # --------------------------------------------------
    # Save interesting examples
    # --------------------------------------------------

    if (
        not ner_found
        and ner_noun_found
        and len(examples["ner_noun"]) < 5
    ):

        examples["ner_noun"].append({
            "target": target,
            "candidates": [
                c
                for c in ner_noun_candidates
                if partial_match(
                    c,
                    target_start,
                    target_end,
                )
            ],
        })

    if (
        not ner_noun_found
        and all_found
        and len(examples["all"]) < 5
    ):

        examples["all"].append({
            "target": target,
            "candidates": [
                c
                for c in all_candidates
                if partial_match(
                    c,
                    target_start,
                    target_end,
                )
            ],
        })


# --------------------------------------------------
# Results
# --------------------------------------------------

total = len(df)


print("\nCandidate Extraction Evaluation")
print("=" * 60)

print(f"Total targets: {total}")


print("\nExact Candidate Recall")
print("-" * 60)

print(
    f"NER only: "
    f"{ner_candidate_recall} / {total} "
    f"({ner_candidate_recall / total * 100:.2f}%)"
)

print(
    f"NER + noun chunks: "
    f"{ner_plus_noun_recall} / {total} "
    f"({ner_plus_noun_recall / total * 100:.2f}%)"
)

print(
    f"NER + noun chunks + pronouns: "
    f"{all_candidate_recall} / {total} "
    f"({all_candidate_recall / total * 100:.2f}%)"
)


print("\nAdditional Examples")
print("-" * 60)

print("\nNER missed, noun chunk recovered:")

for example in examples["ner_noun"]:

    print(
        f"\nTarget: {example['target']}"
    )

    for candidate in example["candidates"]:
        print(
            f"  {candidate}"
        )


print("\nNoun chunks missed, pronoun recovered:")

for example in examples["all"]:

    print(
        f"\nTarget: {example['target']}"
    )

    for candidate in example["candidates"]:
        print(
            f"  {candidate}"
        )
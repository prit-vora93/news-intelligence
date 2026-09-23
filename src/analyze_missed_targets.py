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

    doc = nlp(sentence)

    candidates = []

    # NER
    for ent in doc.ents:
        candidates.append(
            (
                ent.start_char,
                ent.end_char,
                ent.text,
                "NER",
            )
        )

    # Noun chunks
    for chunk in doc.noun_chunks:
        candidates.append(
            (
                chunk.start_char,
                chunk.end_char,
                chunk.text,
                "NOUN_CHUNK",
            )
        )

    # Pronouns
    for token in doc:
        if (
            token.text.lower() in PRONOUNS
            and token.pos_ == "PRON"
        ):
            candidates.append(
                (
                    token.idx,
                    token.idx + len(token.text),
                    token.text,
                    "PRONOUN",
                )
            )

    return candidates


def exact_match(
    candidate,
    target_start,
    target_end,
):

    return (
        candidate[0] == target_start
        and candidate[1] == target_end
    )


missed = []


for _, row in df.iterrows():

    sentence = row["sentence"]
    target = row["target"]

    target_start = int(row["target_from"])
    target_end = int(row["target_to"])

    candidates = get_candidates(sentence)

    found = any(
        exact_match(
            candidate,
            target_start,
            target_end,
        )
        for candidate in candidates
    )

    if not found:

        missed.append(
            {
                "target": target,
                "sentence": sentence,
                "target_words": len(
                    str(target).split()
                ),
                "target_chars": len(
                    str(target)
                ),
                "candidates": [
                    candidate
                    for candidate in candidates
                    if (
                        candidate[0] < target_end
                        and candidate[1] > target_start
                    )
                ],
            }
        )


print("\nMissed Target Analysis")
print("=" * 70)

print(
    f"Total missed targets: {len(missed)}"
)


# --------------------------------------------------
# Word-count distribution
# --------------------------------------------------

print("\nMissed Target Word Counts")
print("-" * 70)

word_counts = {}

for item in missed:

    count = item["target_words"]

    word_counts[count] = (
        word_counts.get(count, 0) + 1
    )


for words, count in sorted(
    word_counts.items()
):

    print(
        f"{words} word(s): "
        f"{count}"
    )


# --------------------------------------------------
# Long targets
# --------------------------------------------------

print("\nMissed Targets Sorted by Length")
print("-" * 70)

for item in sorted(
    missed,
    key=lambda x: x["target_words"],
    reverse=True,
)[:30]:

    print(
        f"\nTarget ({item['target_words']} words): "
        f"{item['target']}"
    )

    print(
        f"Sentence: "
        f"{item['sentence']}"
    )

    if item["candidates"]:

        print("Overlapping candidates:")

        for candidate in item["candidates"]:
            print(
                f"  {candidate}"
            )

    else:

        print(
            "Overlapping candidates: NONE"
        )


# --------------------------------------------------
# Short missed targets
# --------------------------------------------------

print("\nShort Missed Targets")
print("-" * 70)

short_missed = [
    item
    for item in missed
    if item["target_words"] <= 2
]

for item in short_missed[:30]:

    print(
        f"\nTarget: {item['target']}"
    )

    print(
        f"Sentence: {item['sentence']}"
    )

    if item["candidates"]:

        print("Overlapping candidates:")

        for candidate in item["candidates"]:
            print(
                f"  {candidate}"
            )

    else:

        print(
            "Overlapping candidates: NONE"
        )
import re

import pandas as pd
import spacy


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/processed/train_target.csv"

OUTPUT_PATH = "data/processed/candidate_ranker_train.csv"

nlp = spacy.load("en_core_web_sm")


# ============================================================
# Pronouns
# ============================================================

PRONOUNS = {
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
}


# ============================================================
# Honorifics
# ============================================================

HONORIFICS = {
    "mr",
    "mr.",
    "mrs",
    "mrs.",
    "ms",
    "ms.",
    "miss",
    "dr",
    "dr.",
    "prof",
    "prof.",
    "president",
    "senator",
    "sen.",
    "governor",
    "gov.",
    "rep",
    "rep.",
    "representative",
    "gen",
    "gen.",
    "general",
    "judge",
    "rev",
    "rev.",
}


# ============================================================
# Text normalization
# ============================================================

def normalize_text(text):

    text = str(text).strip()

    # Remove leading punctuation/quotes
    text = re.sub(
        r'^[\s"“”\'‘’`´]+',
        "",
        text,
    )

    # Remove trailing punctuation/quotes
    text = re.sub(
        r'[\s"“”\'‘’`´]+$',
        "",
        text,
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# Candidate helper
# ============================================================

def add_candidate(
    candidates,
    start,
    end,
    text,
    source,
    label=None,
):

    if start < 0:
        return

    if end <= start:
        return

    candidates.append(
        {
            "text": text,
            "start": start,
            "end": end,
            "source": source,
            "label": label,
        }
    )


def add_normalized_candidate(
    candidates,
    sentence,
    start,
    end,
    source,
    label=None,
):

    original = sentence[start:end]

    add_candidate(
        candidates,
        start,
        end,
        original,
        source,
        label,
    )

    normalized = normalize_text(
        original
    )

    if not normalized:
        return

    local_start = original.find(
        normalized
    )

    if local_start == -1:
        return

    normalized_start = (
        start + local_start
    )

    normalized_end = (
        normalized_start
        + len(normalized)
    )

    add_candidate(
        candidates,
        normalized_start,
        normalized_end,
        normalized,
        source + "_NORMALIZED",
        label,
    )


# ============================================================
# Sub-span candidates
# ============================================================

def add_subspan_candidates(
    candidates,
    doc,
    sentence,
):

    chunks = (
        list(doc.ents)
        + list(doc.noun_chunks)
    )

    for chunk in chunks:

        tokens = list(chunk)

        # ----------------------------------------------------
        # Individual proper nouns / pronouns
        # ----------------------------------------------------

        for token in tokens:

            if token.pos_ in {
                "PROPN",
                "PRON",
            }:

                add_candidate(
                    candidates,
                    token.idx,
                    token.idx + len(token.text),
                    token.text,
                    "SUBSPAN_TOKEN",
                )

        # ----------------------------------------------------
        # Consecutive proper names
        # ----------------------------------------------------

        start = None
        end = None

        for token in tokens:

            if token.pos_ == "PROPN":

                if start is None:
                    start = token.idx

                end = (
                    token.idx
                    + len(token.text)
                )

            else:

                if start is not None:

                    add_candidate(
                        candidates,
                        start,
                        end,
                        sentence[start:end],
                        "SUBSPAN_PROPER_NAME",
                    )

                    start = None
                    end = None

        if start is not None:

            add_candidate(
                candidates,
                start,
                end,
                sentence[start:end],
                "SUBSPAN_PROPER_NAME",
            )


# ============================================================
# Candidate extraction
# ============================================================

def get_candidates(sentence):

    doc = nlp(sentence)

    candidates = []

    # --------------------------------------------------------
    # 1. NER
    # --------------------------------------------------------

    for ent in doc.ents:

        add_normalized_candidate(
            candidates,
            sentence,
            ent.start_char,
            ent.end_char,
            "NER",
            ent.label_,
        )

    # --------------------------------------------------------
    # 2. Noun chunks
    # --------------------------------------------------------

    for chunk in doc.noun_chunks:

        add_normalized_candidate(
            candidates,
            sentence,
            chunk.start_char,
            chunk.end_char,
            "NOUN_CHUNK",
        )

    # --------------------------------------------------------
    # 3. Pronouns
    # --------------------------------------------------------

    for token in doc:

        if (
            token.text.lower()
            in PRONOUNS
            and token.pos_ == "PRON"
        ):

            add_candidate(
                candidates,
                token.idx,
                token.idx
                + len(token.text),
                token.text,
                "PRONOUN",
            )

    # --------------------------------------------------------
    # 4. Title expansion
    # --------------------------------------------------------

    for ent in doc.ents:

        if ent.label_ not in {
            "PERSON",
            "ORG",
        }:
            continue

        if ent.start == 0:
            continue

        previous = doc[
            ent.start - 1
        ]

        if (
            previous.text.lower()
            in HONORIFICS
        ):

            expanded_start = (
                previous.idx
            )

            expanded_end = (
                ent.end_char
            )

            add_candidate(
                candidates,
                expanded_start,
                expanded_end,
                sentence[
                    expanded_start:
                    expanded_end
                ],
                "TITLE_EXPANSION",
            )

    # --------------------------------------------------------
    # 5. Coordination
    # --------------------------------------------------------

    for token in doc:

        if token.dep_ != "conj":
            continue

        head = token.head

        if head.pos_ not in {
            "PROPN",
            "NOUN",
            "PRON",
        }:
            continue

        start = min(
            head.idx,
            token.idx,
        )

        end = max(
            head.idx + len(head.text),
            token.idx + len(token.text),
        )

        left = min(
            head.i,
            token.i,
        )

        right = max(
            head.i,
            token.i,
        )

        middle = doc[
            left + 1:right
        ]

        conjunction = any(
            t.text.lower()
            in {"and", "or"}
            for t in middle
        )

        if conjunction:

            add_candidate(
                candidates,
                start,
                end,
                sentence[start:end],
                "COORDINATION",
            )

    # --------------------------------------------------------
    # 6. Appositive expansion
    # --------------------------------------------------------

    for ent in doc.ents:

        if ent.label_ != "PERSON":
            continue

        person_end = ent.end

        for i in range(
            person_end,
            min(
                person_end + 15,
                len(doc),
            ),
        ):

            token = doc[i]

            if token.text in {
                ".",
                "!",
                "?",
            }:
                break

            if token.text != ",":
                continue

            end_token = None

            for j in range(
                i + 1,
                min(
                    i + 15,
                    len(doc),
                ),
            ):

                next_token = doc[j]

                if next_token.text in {
                    ",",
                    ".",
                    "!",
                    "?",
                }:

                    end_token = next_token
                    break

            if end_token is None:
                break

            expanded_start = (
                ent.start_char
            )

            expanded_end = (
                end_token.idx
            )

            add_candidate(
                candidates,
                expanded_start,
                expanded_end,
                sentence[
                    expanded_start:
                    expanded_end
                ],
                "APPOSITIVE_EXPANSION",
            )

            break

    # --------------------------------------------------------
    # 7. Sub-spans
    # --------------------------------------------------------

    add_subspan_candidates(
        candidates,
        doc,
        sentence,
    )

    # --------------------------------------------------------
    # Remove duplicate candidates
    # --------------------------------------------------------

    unique = {}

    for candidate in candidates:

        key = (
            candidate["start"],
            candidate["end"],
            candidate["text"],
        )

        if key not in unique:
            unique[key] = candidate

    return list(
        unique.values()
    )


# ============================================================
# Target matching
# ============================================================

def target_matches(
    candidate,
    target,
):

    candidate_text = normalize_text(
        candidate["text"]
    )

    target_text = normalize_text(
        target
    )

    return (
        candidate_text.lower()
        == target_text.lower()
    )


# ============================================================
# Create ranking dataset
# ============================================================

print("\nBuilding Candidate Ranking Dataset")
print("=" * 70)

df = pd.read_csv(
    DATA_PATH
)

print(
    f"Training rows: {len(df)}"
)


rows = []

total_targets = 0
targets_with_candidate = 0

positive_examples = 0
negative_examples = 0


for index, row in df.iterrows():

    sentence = row["sentence"]

    target = row["target"]

    total_targets += 1

    candidates = get_candidates(
        sentence
    )

    found_gold = False

    for candidate in candidates:

        is_gold = target_matches(
            candidate,
            target,
        )

        if is_gold:
            found_gold = True

        rows.append(
            {
                "sentence": sentence,
                "target": target,
                "candidate": candidate["text"],
                "candidate_start": candidate["start"],
                "candidate_end": candidate["end"],
                "source": candidate["source"],
                "ner_label": candidate.get(
                    "label"
                ),
                "label": int(is_gold),
            }
        )

        if is_gold:
            positive_examples += 1
        else:
            negative_examples += 1

    if found_gold:
        targets_with_candidate += 1

    if (
        index + 1
    ) % 500 == 0:

        print(
            f"Processed "
            f"{index + 1} / {len(df)}"
        )


# ============================================================
# Save dataset
# ============================================================

candidate_df = pd.DataFrame(
    rows
)

candidate_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ============================================================
# Statistics
# ============================================================

print("\nDataset Created")
print("-" * 70)

print(
    f"Original targets: "
    f"{total_targets}"
)

print(
    f"Targets with candidate: "
    f"{targets_with_candidate}"
)

print(
    f"Candidate recall: "
    f"{targets_with_candidate / total_targets * 100:.2f}%"
)

print(
    f"Total candidate rows: "
    f"{len(candidate_df)}"
)

print(
    f"Positive examples: "
    f"{positive_examples}"
)

print(
    f"Negative examples: "
    f"{negative_examples}"
)

print(
    f"Positive ratio: "
    f"{positive_examples / len(candidate_df) * 100:.2f}%"
)

print(
    f"\nSaved to:"
    f"\n{OUTPUT_PATH}"
)
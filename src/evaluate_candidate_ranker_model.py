import re

import joblib
import pandas as pd
import spacy


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/processed/validation_target.csv"

MODEL_PATH = "models/candidate_ranker.pkl"

VECTORIZER_PATH = (
    "models/candidate_ranker_vectorizer.pkl"
)

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

    text = re.sub(
        r'^[\s"“”\'‘’`´]+',
        "",
        text,
    )

    text = re.sub(
        r'[\s"“”\'‘’`´]+$',
        "",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# Candidate helpers
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
                token.idx + len(token.text),
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

            expanded_start = previous.idx

            expanded_end = ent.end_char

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

            expanded_start = ent.start_char

            expanded_end = end_token.idx

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
    # Remove duplicates
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
# Feature engineering
# ============================================================

def build_features(row):

    candidate = str(
        row["candidate"]
    )

    sentence = str(
        row["sentence"]
    )

    source = str(
        row["source"]
    )

    ner_label = str(
        row["ner_label"]
    )

    start = int(
        row["candidate_start"]
    )

    end = int(
        row["candidate_end"]
    )

    sentence_length = max(
        len(sentence),
        1,
    )

    candidate_words = len(
        candidate.split()
    )

    first_word = (
        candidate.split()[0].lower()
        if candidate.split()
        else ""
    )

    features = {

        "candidate_length":
            len(candidate),

        "candidate_words":
            candidate_words,

        "start_position":
            start,

        "end_position":
            end,

        "relative_position":
            start / sentence_length,

        # Source
        "source":
            source,

        # NER
        "ner_label":
            ner_label,

        "is_person":
            int(ner_label == "PERSON"),

        "is_org":
            int(ner_label == "ORG"),

        "is_gpe":
            int(ner_label == "GPE"),

        "is_norp":
            int(ner_label == "NORP"),

        "is_date":
            int(ner_label == "DATE"),

        "is_time":
            int(ner_label == "TIME"),

        "is_cardinal":
            int(ner_label == "CARDINAL"),

        # Length
        "is_single_word":
            int(candidate_words == 1),

        "is_multi_word":
            int(candidate_words > 1),

        # Candidate text
        "starts_with_the":
            int(
                candidate.lower().startswith(
                    "the "
                )
            ),

        "starts_with_title":
            int(
                first_word
                in HONORIFICS
            ),

        "contains_hyphen":
            int("-" in candidate),

        "contains_apostrophe":
            int(
                "'" in candidate
                or "’" in candidate
            ),

        # Capitalization
        "first_word_capitalized":
            int(
                candidate.split()[0][0].isupper()
                if candidate.split()
                else False
            ),

        "all_upper":
            int(candidate.isupper()),

        # Position
        "candidate_occurs_at_start":
            int(start <= 2),

        "candidate_occurs_at_end":
            int(
                end >=
                len(sentence) - 2
            ),
    }

    return features


# ============================================================
# Target matching
# ============================================================

def target_matches(
    candidate,
    target,
):

    return (
        normalize_text(candidate).lower()
        == normalize_text(target).lower()
    )


# ============================================================
# Main evaluation
# ============================================================

print("\nLearned Candidate Ranker Evaluation")
print("=" * 70)


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("\nLoading model...")

model = joblib.load(
    MODEL_PATH
)

vectorizer = joblib.load(
    VECTORIZER_PATH
)

print("Model loaded.")


# ------------------------------------------------------------
# Load validation data
# ------------------------------------------------------------

df = pd.read_csv(
    DATA_PATH
)

print(
    f"Validation targets: {len(df)}"
)


# ============================================================
# Evaluation variables
# ============================================================

total_targets = 0

targets_with_candidate = 0

top1 = 0
top3 = 0
top5 = 0

gold_ranks = []

missed_examples = []

ranking_examples = []


# ============================================================
# Process validation examples
# ============================================================

for index, row in df.iterrows():

    sentence = row["sentence"]

    target = row["target"]

    total_targets += 1

    candidates = get_candidates(
        sentence
    )

    if not candidates:
        missed_examples.append(
            {
                "target": target,
                "sentence": sentence,
            }
        )

        continue

    # --------------------------------------------------------
    # Build feature rows
    # --------------------------------------------------------

    feature_rows = []

    candidate_rows = []

    for candidate in candidates:

        feature_row = {
            "sentence": sentence,
            "candidate": candidate["text"],
            "candidate_start": candidate["start"],
            "candidate_end": candidate["end"],
            "source": candidate["source"],
            "ner_label": candidate.get(
                "label"
            ),
        }

        feature_rows.append(
            build_features(
                feature_row
            )
        )

        candidate_rows.append(
            candidate
        )

    # --------------------------------------------------------
    # Vectorize
    # --------------------------------------------------------

    X = vectorizer.transform(
        feature_rows
    )

    # --------------------------------------------------------
    # Predict probability
    # --------------------------------------------------------

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    # --------------------------------------------------------
    # Attach scores
    # --------------------------------------------------------

    ranked = []

    for candidate, probability in zip(
        candidate_rows,
        probabilities,
    ):

        item = candidate.copy()

        item["score"] = float(
            probability
        )

        item["is_gold"] = target_matches(
            candidate["text"],
            target,
        )

        ranked.append(
            item
        )

    # --------------------------------------------------------
    # Sort candidates
    # --------------------------------------------------------

    ranked.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    # --------------------------------------------------------
    # Find gold candidate
    # --------------------------------------------------------

    gold_rank = None

    for rank, candidate in enumerate(
        ranked,
        start=1,
    ):

        if candidate["is_gold"]:

            gold_rank = rank

            break

    # --------------------------------------------------------
    # Gold not generated
    # --------------------------------------------------------

    if gold_rank is None:

        missed_examples.append(
            {
                "target": target,
                "sentence": sentence,
            }
        )

        continue

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    targets_with_candidate += 1

    gold_ranks.append(
        gold_rank
    )

    if gold_rank <= 1:
        top1 += 1

    if gold_rank <= 3:
        top3 += 1

    if gold_rank <= 5:
        top5 += 1

    # --------------------------------------------------------
    # Save first few examples
    # --------------------------------------------------------

    if len(ranking_examples) < 10:

        ranking_examples.append(
            {
                "sentence": sentence,
                "target": target,
                "ranked": ranked,
            }
        )

    if (
        index + 1
    ) % 50 == 0:

        print(
            f"Processed "
            f"{index + 1} / {len(df)}"
        )


# ============================================================
# Results
# ============================================================

print("\n")
print("=" * 70)
print("Validation Ranking Results")
print("=" * 70)


candidate_recall = (
    targets_with_candidate
    / total_targets
)


ranking_denominator = max(
    targets_with_candidate,
    1,
)


top1_accuracy = (
    top1
    / ranking_denominator
)

top3_recall = (
    top3
    / ranking_denominator
)

top5_recall = (
    top5
    / ranking_denominator
)


print(
    f"\nTotal targets: "
    f"{total_targets}"
)

print(
    f"Targets with candidate: "
    f"{targets_with_candidate}"
)

print(
    f"Candidate recall: "
    f"{candidate_recall * 100:.2f}%"
)


print("\nRanking")
print("-" * 70)

print(
    f"Top-1 accuracy: "
    f"{top1_accuracy * 100:.2f}%"
)

print(
    f"Top-3 recall: "
    f"{top3_recall * 100:.2f}%"
)

print(
    f"Top-5 recall: "
    f"{top5_recall * 100:.2f}%"
)


if gold_ranks:

    print(
        f"Average gold rank: "
        f"{sum(gold_ranks) / len(gold_ranks):.2f}"
    )

    sorted_ranks = sorted(
        gold_ranks
    )

    middle = len(
        sorted_ranks
    ) // 2

    if len(sorted_ranks) % 2 == 0:

        median_rank = (
            sorted_ranks[middle - 1]
            + sorted_ranks[middle]
        ) / 2

    else:

        median_rank = (
            sorted_ranks[middle]
        )

    print(
        f"Median gold rank: "
        f"{median_rank:.2f}"
    )


# ============================================================
# Example rankings
# ============================================================

print("\n")
print("=" * 70)
print("Example Rankings")
print("=" * 70)


for example_number, example in enumerate(
    ranking_examples,
    start=1,
):

    print(
        f"\nExample {example_number}"
    )

    print("-" * 70)

    print(
        f"Sentence:\n"
        f"{example['sentence']}"
    )

    print(
        f"\nGold target:\n"
        f"{example['target']}"
    )

    print(
        "\nTop candidates:"
    )

    for rank, candidate in enumerate(
        example["ranked"][:5],
        start=1,
    ):

        marker = (
            " <-- GOLD"
            if candidate["is_gold"]
            else ""
        )

        print(
            f"{rank:>2}. "
            f"{candidate['text']!r:<40} "
            f"score={candidate['score']:.4f} "
            f"{candidate['source']:<25}"
            f"{marker}"
        )


# ============================================================
# Missed targets
# ============================================================

print("\n")
print("=" * 70)
print("Missed Targets")
print("=" * 70)

print(
    f"Total missed: "
    f"{len(missed_examples)}"
)

for example in missed_examples[:20]:

    print(
        f"\nTarget: {example['target']}"
    )

    print(
        f"Sentence: {example['sentence']}"
    )


print("\nEvaluation complete.")
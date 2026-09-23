import os
import sys
import joblib
import pandas as pd

# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/candidate_ranker.pkl"
VECTORIZER_PATH = "models/candidate_ranker_vectorizer.pkl"

DATA_PATH = "data/processed/candidate_ranker_train.csv"


# ============================================================
# Load model
# ============================================================

print("\nLoading trained candidate ranker")
print("=" * 70)

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found: {MODEL_PATH}")
    sys.exit(1)

if not os.path.exists(VECTORIZER_PATH):
    print(f"ERROR: Vectorizer not found: {VECTORIZER_PATH}")
    sys.exit(1)

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

print(f"Model loaded:      {MODEL_PATH}")
print(f"Vectorizer loaded: {VECTORIZER_PATH}")


# ============================================================
# Feature engineering
#
# IMPORTANT:
# This must match train_candidate_ranker_model.py
# ============================================================

def build_features(row):

    candidate = str(row["candidate"])
    sentence = str(row["sentence"])

    source = str(row["source"])
    ner_label = str(row["ner_label"])

    start = int(row["candidate_start"])
    end = int(row["candidate_end"])

    sentence_length = max(len(sentence), 1)

    candidate_length = len(candidate)
    candidate_words = len(candidate.split())

    # --------------------------------------------------------
    # Safe text helpers
    # --------------------------------------------------------

    words = candidate.split()

    first_word = (
        words[0]
        if words
        else ""
    )

    candidate_lower = candidate.lower()

    # --------------------------------------------------------
    # Relative measurements
    # --------------------------------------------------------

    relative_position = (
        start / sentence_length
    )

    relative_end_position = (
        end / sentence_length
    )

    relative_candidate_length = (
        candidate_length / sentence_length
    )

    # --------------------------------------------------------
    # Target information
    #
    # During real inference there may be no target.
    # For inspection we have it available.
    #
    # We deliberately DO NOT use target-derived features here.
    # Otherwise evaluation would leak the answer.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Candidate/source characteristics
    # --------------------------------------------------------

    is_ner = int(
        source.startswith("NER")
    )

    is_subspan = int(
        "SUBSPAN" in source
    )

    is_normalized = int(
        "NORMALIZED" in source
    )

    is_appositive = int(
        "APPOSITIVE" in source
    )

    is_coordination = int(
        source == "COORDINATION"
    )

    is_person = int(
        ner_label == "PERSON"
    )

    is_org = int(
        ner_label == "ORG"
    )

    is_gpe = int(
        ner_label == "GPE"
    )

    is_norp = int(
        ner_label == "NORP"
    )

    is_date = int(
        ner_label == "DATE"
    )

    is_time = int(
        ner_label == "TIME"
    )

    is_cardinal = int(
        ner_label == "CARDINAL"
    )

    is_product = int(
        ner_label == "PRODUCT"
    )

    is_law = int(
        ner_label == "LAW"
    )

    is_fac = int(
        ner_label == "FAC"
    )

    is_single_word = int(
        candidate_words == 1
    )

    is_multi_word = int(
        candidate_words > 1
    )

    is_three_words = int(
        candidate_words == 3
    )

    is_short_candidate = int(
        candidate_length <= 10
    )

    is_long_candidate = int(
        candidate_length >= 30
    )

    starts_with_the = int(
        candidate_lower.startswith("the ")
    )

    starts_with_a = int(
        candidate_lower.startswith("a ")
    )

    starts_with_an = int(
        candidate_lower.startswith("an ")
    )

    starts_with_title = int(
        first_word.lower()
        in {
            "mr",
            "mr.",
            "mrs",
            "mrs.",
            "ms",
            "ms.",
            "dr",
            "dr.",
            "president",
            "senator",
            "sen.",
            "governor",
            "gov.",
            "rep",
            "rep.",
            "judge",
            "general",
            "gen",
        }
    )

    contains_hyphen = int(
        "-" in candidate
    )

    contains_apostrophe = int(
        "'" in candidate
        or "’" in candidate
    )

    contains_uppercase = int(
        any(
            char.isupper()
            for char in candidate
        )
    )

    all_upper = int(
        candidate.isupper()
    )

    all_lower = int(
        candidate.islower()
    )

    first_word_capitalized = int(
        first_word[0].isupper()
        if first_word
        else False
    )

    # --------------------------------------------------------
    # Punctuation/context
    # --------------------------------------------------------

    starts_with_quote = int(
        candidate.startswith(
            ('"', "'", "“", "‘")
        )
    )

    ends_with_quote = int(
        candidate.endswith(
            ('"', "'", "”", "’")
        )
    )

    starts_with_dash = int(
        candidate.startswith(
            ("-", "–", "—")
        )
    )

    preceded_by_quote = int(
        0 < start <= len(sentence)
        and sentence[start - 1]
        in {'"', "'", "“", "‘"}
    )

    followed_by_quote = int(
        end < len(sentence)
        and sentence[end:end + 1]
        in {'"', "'", "”", "’"}
    )

    preceded_by_dash = int(
        0 < start <= len(sentence)
        and sentence[start - 1] in {'"', "'", "“", "‘", "-", "–", "—"}
    )

    followed_by_dash = int(
        end < len(sentence)
        and sentence[end:end + 1]
        in {"-", "–", "—"}
    )

    has_edge_punctuation = int(
        bool(candidate)
        and (
            not candidate[0].isalnum()
            or not candidate[-1].isalnum()
        )
    )

    # --------------------------------------------------------
    # Position
    # --------------------------------------------------------

    candidate_occurs_at_start = int(
        start <= 2
    )

    candidate_occurs_at_end = int(
        end >= sentence_length - 2
    )

    # --------------------------------------------------------
    # Build dictionary
    # --------------------------------------------------------

    features = {

        # Basic
        "candidate_length":
            candidate_length,

        "candidate_words":
            candidate_words,

        # Position
        "start_position":
            start,

        "end_position":
            end,

        "relative_position":
            relative_position,

        "relative_end_position":
            relative_end_position,

        "relative_candidate_length":
            relative_candidate_length,

        # Source
        "source":
            source,

        "is_ner":
            is_ner,

        "is_subspan":
            is_subspan,

        "is_normalized":
            is_normalized,

        "is_appositive":
            is_appositive,

        "is_coordination":
            is_coordination,

        # NER
        "ner_label":
            ner_label,

        "is_person":
            is_person,

        "is_org":
            is_org,

        "is_gpe":
            is_gpe,

        "is_norp":
            is_norp,

        "is_date":
            is_date,

        "is_time":
            is_time,

        "is_cardinal":
            is_cardinal,

        "is_product":
            is_product,

        "is_law":
            is_law,

        "is_fac":
            is_fac,

        # Length
        "is_single_word":
            is_single_word,

        "is_multi_word":
            is_multi_word,

        "is_three_words":
            is_three_words,

        "is_short_candidate":
            is_short_candidate,

        "is_long_candidate":
            is_long_candidate,

        # Text characteristics
        "starts_with_the":
            starts_with_the,

        "starts_with_a":
            starts_with_a,

        "starts_with_an":
            starts_with_an,

        "starts_with_title":
            starts_with_title,

        "contains_hyphen":
            contains_hyphen,

        "contains_apostrophe":
            contains_apostrophe,

        "contains_uppercase":
            contains_uppercase,

        "all_upper":
            all_upper,

        "all_lower":
            all_lower,

        "first_word_capitalized":
            first_word_capitalized,

        # Punctuation
        "starts_with_quote":
            starts_with_quote,

        "ends_with_quote":
            ends_with_quote,

        "starts_with_dash":
            starts_with_dash,

        "preceded_by_quote":
            preceded_by_quote,

        "followed_by_quote":
            followed_by_quote,

        "preceded_by_dash":
            preceded_by_dash,

        "followed_by_dash":
            followed_by_dash,

        "has_edge_punctuation":
            has_edge_punctuation,

        # Context
        "candidate_occurs_at_start":
            candidate_occurs_at_start,

        "candidate_occurs_at_end":
            candidate_occurs_at_end,
    }

    return features


# ============================================================
# Candidate ranking
# ============================================================

def rank_candidates(sentence, candidates):

    rows = []

    for candidate in candidates:

        row = {
            "sentence": sentence,
            "candidate": candidate["text"],
            "candidate_start": candidate["start"],
            "candidate_end": candidate["end"],
            "source": candidate["source"],
            "ner_label": candidate.get(
                "ner_label",
                None,
            ),
        }

        rows.append(row)

    if not rows:
        return []

    feature_dicts = [
        build_features(row)
        for row in rows
    ]

    X = vectorizer.transform(
        feature_dicts
    )

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    results = []

    for row, probability in zip(
        rows,
        probabilities,
    ):

        results.append({
            **row,
            "score": float(
                probability
            ),
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return results


# ============================================================
# Dataset-wide evaluation
# ============================================================

def evaluate_dataset():

    print("\nDataset-wide Ranking Evaluation")
    print("=" * 70)

    if not os.path.exists(DATA_PATH):

        print(
            f"Dataset not found: {DATA_PATH}"
        )

        return

    df = pd.read_csv(DATA_PATH)

    grouped = (
        df.groupby(
            ["sentence", "target"],
            sort=False,
        )
    )

    total_targets = 0

    matching_targets = 0

    top1 = 0
    top3 = 0
    top5 = 0

    ranks = []

    for (
        sentence,
        target
    ), group in grouped:

        total_targets += 1

        rows = []

        for _, row in group.iterrows():

            rows.append({
                "text":
                    str(row["candidate"]),

                "start":
                    int(row["candidate_start"]),

                "end":
                    int(row["candidate_end"]),

                "source":
                    str(row["source"]),

                "ner_label":
                    row["ner_label"],
            })

        ranked = rank_candidates(
            sentence,
            rows,
        )

        target_normalized = (
            str(target)
            .strip()
            .lower()
        )

        matching_rank = None

        for rank, candidate in enumerate(
            ranked,
            start=1,
        ):

            candidate_normalized = (
                candidate["candidate"]
                .strip()
                .lower()
            )

            if (
                candidate_normalized
                == target_normalized
            ):

                matching_rank = rank
                break

        if matching_rank is None:
            continue

        matching_targets += 1

        ranks.append(
            matching_rank
        )

        if matching_rank == 1:
            top1 += 1

        if matching_rank <= 3:
            top3 += 1

        if matching_rank <= 5:
            top5 += 1

    print(
        f"\nTargets with matching candidate: "
        f"{matching_targets} / {total_targets}"
    )

    if not ranks:

        print(
            "No matching candidates found."
        )

        return

    print(
        f"Top-1 accuracy: "
        f"{top1 / total_targets * 100:.2f}%"
    )

    print(
        f"Top-3 recall: "
        f"{top3 / total_targets * 100:.2f}%"
    )

    print(
        f"Top-5 recall: "
        f"{top5 / total_targets * 100:.2f}%"
    )

    print(
        f"Average gold rank: "
        f"{sum(ranks) / len(ranks):.2f}"
    )

    sorted_ranks = sorted(ranks)

    median = sorted_ranks[
        len(sorted_ranks) // 2
    ]

    print(
        f"Median gold rank: "
        f"{median}"
    )


# ============================================================
# Inspection examples
# ============================================================

def inspect_examples():

    examples = [

        {
            "sentence":
                "President Trump praised the new economic policy.",

            "target":
                "President Trump",

            "candidates": [
                {
                    "text":
                        "President Trump",
                    "start":
                        0,
                    "end":
                        15,
                    "source":
                        "NOUN_CHUNK",
                    "ner_label":
                        None,
                },
                {
                    "text":
                        "Trump",
                    "start":
                        9,
                    "end":
                        14,
                    "source":
                        "NER",
                    "ner_label":
                        "PERSON",
                },
                {
                    "text":
                        "President",
                    "start":
                        0,
                    "end":
                        9,
                    "source":
                        "SUBSPAN_TOKEN",
                    "ner_label":
                        None,
                },
                {
                    "text":
                        "the new economic policy",
                    "start":
                        30,
                    "end":
                        54,
                    "source":
                        "NOUN_CHUNK",
                    "ner_label":
                        None,
                },
            ],
        },

        {
            "sentence":
                "Critics said President Trump handled the situation poorly.",

            "target":
                "President Trump",

            "candidates": [
                {
                    "text":
                        "President Trump",
                    "start":
                        13,
                    "end":
                        28,
                    "source":
                        "NOUN_CHUNK",
                    "ner_label":
                        None,
                },
                {
                    "text":
                        "Trump",
                    "start":
                        23,
                    "end":
                        28,
                    "source":
                        "NER",
                    "ner_label":
                        "PERSON",
                },
                {
                    "text":
                        "President",
                    "start":
                        13,
                    "end":
                        22,
                    "source":
                        "SUBSPAN_TOKEN",
                    "ner_label":
                        None,
                },
                {
                    "text":
                        "the situation",
                    "start":
                        37,
                    "end":
                        51,
                    "source":
                        "NOUN_CHUNK",
                    "ner_label":
                        None,
                },
                {
                    "text":
                        "Critics",
                    "start":
                        0,
                    "end":
                        7,
                    "source":
                        "NOUN_CHUNK",
                    "ner_label":
                        None,
                },
            ],
        },

        {
            "sentence":
                "The company announced strong profits today.",

            "target":
                "company",

            "candidates": [
                {
                    "text":
                        "The company",
                    "start":
                        0,
                    "end":
                        11,
                    "source":
                        "NOUN_CHUNK",
                    "ner_label":
                        None,
                },
                {
                    "text":
                        "strong profits",
                    "start":
                        29,
                    "end":
                        44,
                    "source":
                        "NOUN_CHUNK",
                    "ner_label":
                        None,
                },
                {
                    "text":
                        "today",
                    "start":
                        45,
                    "end":
                        50,
                    "source":
                        "NER",
                    "ner_label":
                        "DATE",
                },
            ],
        },

        {
            "sentence":
                "Wa Lone and Kyaw Soe Oo were arrested in December.",

            "target":
                "Wa Lone and Kyaw Soe Oo",

            "candidates": [
                {
                    "text":
                        "Kyaw Soe Oo",
                    "start":
                        16,
                    "end":
                        27,
                    "source":
                        "NER",
                    "ner_label":
                        "PERSON",
                },
                {
                    "text":
                        "Lone",
                    "start":
                        3,
                    "end":
                        7,
                    "source":
                        "NOUN_CHUNK",
                    "ner_label":
                        None,
                },
                {
                    "text":
                        "Lone and Kyaw Soe Oo",
                    "start":
                        3,
                    "end":
                        27,
                    "source":
                        "COORDINATION",
                    "ner_label":
                        None,
                },
                {
                    "text":
                        "Kyaw",
                    "start":
                        16,
                    "end":
                        20,
                    "source":
                        "SUBSPAN_TOKEN",
                    "ner_label":
                        None,
                },
                {
                    "text":
                        "Oo",
                    "start":
                        25,
                    "end":
                        27,
                    "source":
                        "SUBSPAN_TOKEN",
                    "ner_label":
                        None,
                },
                {
                    "text":
                        "Soe",
                    "start":
                        21,
                    "end":
                        24,
                    "source":
                        "SUBSPAN_TOKEN",
                    "ner_label":
                        None,
                },
                {
                    "text":
                        "December",
                    "start":
                        43,
                    "end":
                        51,
                    "source":
                        "NER",
                    "ner_label":
                        "DATE",
                },
            ],
        },

        {
            "sentence":
                "John McCain is one of this country's best-known politicians.",

            "target":
                "McCain",

            "candidates": [
                {
                    "text":
                        "John McCain",
                    "start":
                        0,
                    "end":
                        11,
                    "source":
                        "NER",
                    "ner_label":
                        "PERSON",
                },
                {
                    "text":
                        "John",
                    "start":
                        0,
                    "end":
                        4,
                    "source":
                        "SUBSPAN_TOKEN",
                    "ner_label":
                        None,
                },
                {
                    "text":
                        "McCain",
                    "start":
                        5,
                    "end":
                        11,
                    "source":
                        "SUBSPAN_TOKEN",
                    "ner_label":
                        None,
                },
                {
                    "text":
                        "this country's best-known politicians",
                    "start":
                        27,
                    "end":
                        66,
                    "source":
                        "NOUN_CHUNK",
                    "ner_label":
                        None,
                },
                {
                    "text":
                        "one",
                    "start":
                        16,
                    "end":
                        19,
                    "source":
                        "NER",
                    "ner_label":
                        "CARDINAL",
                },
            ],
        },

        {
            "sentence":
                "Trump picks Sean Spicer for press secretary.",

            "target":
                "Spicer",

            "candidates": [
                {
                    "text":
                        "Sean Spicer",
                    "start":
                        12,
                    "end":
                        23,
                    "source":
                        "NER",
                    "ner_label":
                        "PERSON",
                },
                {
                    "text":
                        "Spicer",
                    "start":
                        17,
                    "end":
                        23,
                    "source":
                        "SUBSPAN_TOKEN",
                    "ner_label":
                        None,
                },
                {
                    "text":
                        "Sean",
                    "start":
                        12,
                    "end":
                        16,
                    "source":
                        "SUBSPAN_TOKEN",
                    "ner_label":
                        None,
                },
                {
                    "text":
                        "Trump",
                    "start":
                        0,
                    "end":
                        5,
                    "source":
                        "NOUN_CHUNK",
                    "ner_label":
                        None,
                },
                {
                    "text":
                        "picks",
                    "start":
                        6,
                    "end":
                        11,
                    "source":
                        "NOUN_CHUNK",
                    "ner_label":
                        None,
                },
                {
                    "text":
                        "press secretary",
                    "start":
                        28,
                    "end":
                        44,
                    "source":
                        "NOUN_CHUNK",
                    "ner_label":
                        None,
                },
            ],
        },

        {
            "sentence":
                "Members of Italian-American groups are incensed by New York City Mayor Bill de Blasio's refusal.",

            "target":
                "de Blasio",

            "candidates": [
                {
                    "text":
                        "New York City",
                    "start":
                        58,
                    "end":
                        72,
                    "source":
                        "NER",
                    "ner_label":
                        "GPE",
                },
                {
                    "text":
                        "Bill de Blasio's",
                    "start":
                        85,
                    "end":
                        101,
                    "source":
                        "NER",
                    "ner_label":
                        "PERSON",
                },
                {
                    "text":
                        "Bill de Blasio",
                    "start":
                        85,
                    "end":
                        99,
                    "source":
                        "SUBSPAN_PROPER_NAME",
                    "ner_label":
                        None,
                },
                {
                    "text":
                        "Mayor",
                    "start":
                        73,
                    "end":
                        78,
                    "source":
                        "SUBSPAN_TOKEN",
                    "ner_label":
                        None,
                },
                {
                    "text":
                        "Italian",
                    "start":
                        11,
                    "end":
                        18,
                    "source":
                        "NER",
                    "ner_label":
                        "NORP",
                },
                {
                    "text":
                        "New York City Mayor Bill de Blasio",
                    "start":
                        58,
                    "end":
                        99,
                    "source":
                        "SUBSPAN_PROPER_NAME",
                    "ner_label":
                        None,
                },
                {
                    "text":
                        "de",
                    "start":
                        93,
                    "end":
                        95,
                    "source":
                        "SUBSPAN_TOKEN",
                    "ner_label":
                        None,
                },
                {
                    "text":
                        "Bill",
                    "start":
                        85,
                    "end":
                        89,
                    "source":
                        "SUBSPAN_TOKEN",
                    "ner_label":
                        None,
                },
                {
                    "text":
                        "Blasio",
                    "start":
                        96,
                    "end":
                        102,
                    "source":
                        "SUBSPAN_TOKEN",
                    "ner_label":
                        None,
                },
                {
                    "text":
                        "New York City Mayor Bill de Blasio's refusal",
                    "start":
                        58,
                    "end":
                        109,
                    "source":
                        "NOUN_CHUNK",
                    "ner_label":
                        None,
                },
                {
                    "text":
                        "New",
                    "start":
                        58,
                    "end":
                        61,
                    "source":
                        "SUBSPAN_TOKEN",
                    "ner_label":
                        None,
                },
                {
                    "text":
                        "York",
                    "start":
                        62,
                    "end":
                        66,
                    "source":
                        "SUBSPAN_TOKEN",
                    "ner_label":
                        None,
                },
                {
                    "text":
                        "Members",
                    "start":
                        0,
                    "end":
                        7,
                    "source":
                        "NOUN_CHUNK",
                    "ner_label":
                        None,
                },
                {
                    "text":
                        "Italian-American groups",
                    "start":
                        11,
                    "end":
                        35,
                    "source":
                        "NOUN_CHUNK",
                    "ner_label":
                        None,
                },
                {
                    "text":
                        "City",
                    "start":
                        69,
                    "end":
                        73,
                    "source":
                        "SUBSPAN_TOKEN",
                    "ner_label":
                        None,
                },
            ],
        },
    ]

    print("\nCandidate Ranking Inspection")
    print("=" * 70)

    for index, example in enumerate(
        examples,
        start=1,
    ):

        sentence = example["sentence"]
        target = example["target"]

        ranked = rank_candidates(
            sentence,
            example["candidates"],
        )

        print(
            f"\nExample {index}"
        )

        print("-" * 70)

        print(
            "Sentence:"
        )

        print(sentence)

        print(
            "\nGold target:"
        )

        print(target)

        print(
            f"\nRanked candidates "
            f"({len(ranked)}):"
        )

        for rank, candidate in enumerate(
            ranked,
            start=1,
        ):

            text_value = candidate[
                "candidate"
            ]

            score = candidate[
                "score"
            ]

            source = candidate[
                "source"
            ]

            ner_label = candidate[
                "ner_label"
            ]

            is_gold = (
                text_value.strip().lower()
                == target.strip().lower()
            )

            marker = (
                "<-- GOLD"
                if is_gold
                else ""
            )

            print(
                f"{rank:2d}. "
                f"{text_value!r:<45} "
                f"score={score:7.4f} "
                f"{source:<25} "
                f"label={str(ner_label):<10} "
                f"{marker}"
            )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    inspect_examples()

    evaluate_dataset()
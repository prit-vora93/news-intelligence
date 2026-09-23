import pandas as pd
import joblib
import numpy as np


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/processed/candidate_ranker_train.csv"

MODEL_PATH = "models/candidate_ranker.pkl"
VECTORIZER_PATH = "models/candidate_ranker_vectorizer.pkl"


# ============================================================
# Load model + vectorizer
# ============================================================

print("\nLoading trained candidate ranker")
print("=" * 70)

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

print(f"Model loaded:      {MODEL_PATH}")
print(f"Vectorizer loaded: {VECTORIZER_PATH}")


# ============================================================
# Load dataset
# ============================================================

print("\nLoading candidate dataset")
print("-" * 70)

df = pd.read_csv(DATA_PATH)

print(f"Rows: {len(df)}")


# ============================================================
# Feature engineering
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

    # Safe offset handling
    prev_char = (
        sentence[start - 1]
        if 0 < start <= len(sentence)
        else ""
    )

    next_char = (
        sentence[end]
        if 0 <= end < len(sentence)
        else ""
    )

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
            start / sentence_length,

        "relative_end_position":
            end / sentence_length,

        "relative_candidate_length":
            candidate_length / sentence_length,

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

        "is_product":
            int(ner_label == "PRODUCT"),

        "is_law":
            int(ner_label == "LAW"),

        "is_fac":
            int(ner_label == "FAC"),

        # Candidate characteristics
        "is_single_word":
            int(candidate_words == 1),

        "is_multi_word":
            int(candidate_words > 1),

        "is_two_words":
            int(candidate_words == 2),

        "is_three_words":
            int(candidate_words == 3),

        "is_short_candidate":
            int(candidate_length <= 5),

        "is_long_candidate":
            int(candidate_length >= 30),

        "starts_with_the":
            int(
                candidate.lower().startswith("the ")
            ),

        "starts_with_a":
            int(
                candidate.lower().startswith("a ")
            ),

        "starts_with_an":
            int(
                candidate.lower().startswith("an ")
            ),

        "starts_with_title":
            int(
                candidate.lower().split()[0]
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
                if candidate.split()
                else False
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

        "all_lower":
            int(candidate.islower()),

        "contains_uppercase":
            int(any(c.isupper() for c in candidate)),

        # Punctuation
        "starts_with_quote":
            int(
                candidate.startswith(
                    ('"', "'", "“", "‘")
                )
            ),

        "ends_with_quote":
            int(
                candidate.endswith(
                    ('"', "'", "”", "’")
                )
            ),

        "starts_with_dash":
            int(
                candidate.startswith(
                    ("-", "–", "—")
                )
            ),

        "has_edge_punctuation":
            int(
                candidate[:1] in "\"'“‘-–—"
                or
                candidate[-1:] in "\"'”’-–—"
            ),

        # Context
        "preceded_by_quote":
            int(
                prev_char in {
                    '"',
                    "'",
                    "“",
                    "‘",
                }
            ),

        "followed_by_quote":
            int(
                next_char in {
                    '"',
                    "'",
                    "”",
                    "’",
                }
            ),

        "preceded_by_dash":
            int(
                prev_char in {
                    "-",
                    "–",
                    "—",
                }
            ),

        "followed_by_dash":
            int(
                next_char in {
                    "-",
                    "–",
                    "—",
                }
            ),

        "candidate_occurs_at_start":
            int(start <= 2),

        "candidate_occurs_at_end":
            int(
                end >= len(sentence) - 2
            ),

        # Source characteristics
        "is_ner":
            int(source == "NER"),

        "is_subspan":
            int(
                source
                in {
                    "SUBSPAN_TOKEN",
                    "SUBSPAN_PROPER_NAME",
                }
            ),

        "is_normalized":
            int(
                "NORMALIZED" in source
            ),

        "is_coordination":
            int(
                source == "COORDINATION"
            ),

        "is_appositive":
            int(
                source == "APPOSITIVE_EXPANSION"
            ),

        # Useful span-shape signals
        "has_internal_space":
            int(" " in candidate.strip()),

        "starts_with_capital":
            int(
                candidate[:1].isupper()
            ),

        "ends_with_s":
            int(
                candidate.lower().endswith("s")
            ),

    }

    return features


# ============================================================
# Candidate ranking
# ============================================================

def rank_candidates(group):

    feature_dicts = []

    for _, row in group.iterrows():

        feature_dicts.append(
            build_features(row)
        )

    X = vectorizer.transform(
        feature_dicts
    )

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    ranked = group.copy()

    ranked["score"] = probabilities

    ranked = ranked.sort_values(
        "score",
        ascending=False,
    )

    return ranked


# ============================================================
# Error classification
# ============================================================

def classify_error(
    gold,
    predicted,
    candidates,
):

    if predicted == gold:
        return "CORRECT"

    gold_words = len(gold.split())
    predicted_words = len(predicted.split())

    gold_lower = gold.lower()
    predicted_lower = predicted.lower()

    # --------------------------------------------------------
    # Larger span
    # --------------------------------------------------------

    if (
        gold_lower in predicted_lower
        and predicted_words > gold_words
    ):
        return "LARGER_SPAN"

    # --------------------------------------------------------
    # Smaller span
    # --------------------------------------------------------

    if (
        predicted_lower in gold_lower
        and predicted_words < gold_words
    ):
        return "SMALLER_SPAN"

    # --------------------------------------------------------
    # Same entity / possessive variation
    # --------------------------------------------------------

    if (
        gold_lower.rstrip("'s")
        == predicted_lower.rstrip("'s")
    ):
        return "POSSESSIVE_VARIATION"

    # --------------------------------------------------------
    # NER vs subspan
    # --------------------------------------------------------

    gold_row = candidates[
        candidates["candidate"] == gold
    ]

    predicted_row = candidates[
        candidates["candidate"] == predicted
    ]

    if (
        len(gold_row) > 0
        and len(predicted_row) > 0
    ):

        gold_source = str(
            gold_row.iloc[0]["source"]
        )

        predicted_source = str(
            predicted_row.iloc[0]["source"]
        )

        if (
            "SUBSPAN" in gold_source
            and predicted_source == "NER"
        ):
            return "NER_VS_SUBSPAN"

        if (
            "NER" in gold_source
            and "SUBSPAN" in predicted_source
        ):
            return "SUBSPAN_VS_NER"

    # --------------------------------------------------------
    # Coordination
    # --------------------------------------------------------

    if (
        "COORDINATION"
        in str(
            predicted_row.iloc[0]["source"]
        )
        if len(predicted_row) > 0
        else False
    ):
        return "COORDINATION_ERROR"

    return "OTHER"


# ============================================================
# Evaluate dataset
# ============================================================

print("\nRunning ranking error analysis")
print("=" * 70)

results = []

grouped = df.groupby(
    ["sentence", "target"],
    sort=False,
)

total = len(grouped)

for index, ((sentence, target), group) in enumerate(
    grouped,
    start=1,
):

    positives = group[
        group["label"] == 1
    ]

    if len(positives) == 0:
        continue

    gold = str(
        positives.iloc[0]["candidate"]
    )

    ranked = rank_candidates(group)

    predicted = str(
        ranked.iloc[0]["candidate"]
    )

    gold_matches = ranked[
        ranked["candidate"] == gold
    ]

    if len(gold_matches) == 0:
        continue

    gold_rank = (
        ranked.index.get_loc(
            gold_matches.index[0]
        )
        + 1
    )

    error_type = classify_error(
        gold,
        predicted,
        ranked,
    )

    results.append({
        "sentence": sentence,
        "gold": gold,
        "predicted": predicted,
        "gold_rank": gold_rank,
        "error_type": error_type,
        "predicted_score":
            float(ranked.iloc[0]["score"]),
        "gold_score":
            float(
                gold_matches.iloc[0]["score"]
            ),
    })

    if index % 500 == 0:
        print(
            f"Processed {index} / {total}"
        )


results_df = pd.DataFrame(results)


# ============================================================
# Summary
# ============================================================

print("\n")
print("=" * 70)
print("RANKING ERROR SUMMARY")
print("=" * 70)

print(
    f"\nEvaluated targets: "
    f"{len(results_df)}"
)

print("\nError categories")
print("-" * 70)

counts = (
    results_df["error_type"]
    .value_counts()
)

for error_type, count in counts.items():

    percentage = (
        count / len(results_df) * 100
    )

    print(
        f"{error_type:<25}"
        f"{count:>6}"
        f"  ({percentage:5.2f}%)"
    )


# ============================================================
# Wrong predictions only
# ============================================================

errors = results_df[
    results_df["error_type"] != "CORRECT"
].copy()


print("\n")
print("=" * 70)
print("TOP RANKING ERRORS")
print("=" * 70)

print(
    f"\nTotal errors: {len(errors)}"
)


for error_type in [
    "LARGER_SPAN",
    "SMALLER_SPAN",
    "NER_VS_SUBSPAN",
    "SUBSPAN_VS_NER",
    "POSSESSIVE_VARIATION",
    "COORDINATION_ERROR",
    "OTHER",
]:

    subset = errors[
        errors["error_type"] == error_type
    ]

    if len(subset) == 0:
        continue

    print("\n")
    print("-" * 70)
    print(
        f"{error_type} "
        f"({len(subset)} examples)"
    )
    print("-" * 70)

    for _, row in subset.head(10).iterrows():

        print(
            f"\nSentence:\n"
            f"{row['sentence']}"
        )

        print(
            f"Gold:      "
            f"{row['gold']}"
        )

        print(
            f"Predicted: "
            f"{row['predicted']}"
        )

        print(
            f"Gold rank: "
            f"{row['gold_rank']}"
        )

        print(
            f"Scores: predicted="
            f"{row['predicted_score']:.4f}, "
            f"gold="
            f"{row['gold_score']:.4f}"
        )


# ============================================================
# Save error report
# ============================================================

OUTPUT_PATH = (
    "data/processed/"
    "candidate_ranking_errors.csv"
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
)

print("\n")
print("=" * 70)
print("Analysis complete")
print("=" * 70)

print(
    f"\nSaved error report:"
    f"\n{OUTPUT_PATH}"
)
import os
import sys
import joblib
import pandas as pd

# ============================================================
# Make project root importable
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Configuration
# ============================================================

DATA_PATH = (
    "data/processed/candidate_ranker_train.csv"
)

MODEL_PATH = (
    "models/candidate_ranker.pkl"
)

VECTORIZER_PATH = (
    "models/candidate_ranker_vectorizer.pkl"
)


# ============================================================
# Load model
# ============================================================

print()
print("=" * 70)
print("Trained Candidate Ranker Evaluation")
print("=" * 70)

print()
print("Loading dataset")
print("-" * 70)

df = pd.read_csv(DATA_PATH)

print(f"Rows: {len(df)}")


print()
print("Loading trained model")
print("-" * 70)

model = joblib.load(MODEL_PATH)

vectorizer = joblib.load(
    VECTORIZER_PATH
)

print("Model loaded successfully.")
print("Vectorizer loaded successfully.")


# ============================================================
# Feature engineering
#
# IMPORTANT:
# These features MUST NOT use the gold target.
#
# At real inference time we know:
#   sentence
#   candidate
#   offsets
#   source
#   NER label
#
# We DO NOT know:
#   target
#
# ============================================================

def build_inference_features(row):

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

    candidate_length = len(
        candidate
    )

    candidate_words = len(
        candidate.split()
    )

    first_word = (
        candidate.split()[0]
        if candidate.split()
        else ""
    )

    # --------------------------------------------------------
    # Basic candidate features
    # --------------------------------------------------------

    features = {

        "candidate_length":
            candidate_length,

        "candidate_words":
            candidate_words,

        "start_position":
            start,

        "end_position":
            end,

        "relative_position":
            start / sentence_length,

        # ----------------------------------------------------
        # Source
        # ----------------------------------------------------

        "source":
            source,

        # ----------------------------------------------------
        # NER
        # ----------------------------------------------------

        "ner_label":
            ner_label,

        "is_person":
            int(
                ner_label == "PERSON"
            ),

        "is_org":
            int(
                ner_label == "ORG"
            ),

        "is_gpe":
            int(
                ner_label == "GPE"
            ),

        "is_norp":
            int(
                ner_label == "NORP"
            ),

        "is_date":
            int(
                ner_label == "DATE"
            ),

        "is_time":
            int(
                ner_label == "TIME"
            ),

        "is_cardinal":
            int(
                ner_label == "CARDINAL"
            ),

        # ----------------------------------------------------
        # Candidate characteristics
        # ----------------------------------------------------

        "is_single_word":
            int(
                candidate_words == 1
            ),

        "is_multi_word":
            int(
                candidate_words > 1
            ),

        "starts_with_the":
            int(
                candidate.lower().startswith(
                    "the "
                )
            ),

        "starts_with_title":
            int(
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
            ),

        "contains_hyphen":
            int(
                "-" in candidate
            ),

        "contains_apostrophe":
            int(
                "'" in candidate
                or "’" in candidate
            ),

        # ----------------------------------------------------
        # Capitalization
        # ----------------------------------------------------

        "first_word_capitalized":
            int(
                first_word[0].isupper()
                if first_word
                else False
            ),

        "all_upper":
            int(
                candidate.isupper()
            ),

        "contains_uppercase":
            int(
                any(
                    c.isupper()
                    for c in candidate
                )
            ),

        # ----------------------------------------------------
        # Punctuation
        # ----------------------------------------------------

        "starts_with_quote":
            int(
                candidate.startswith(
                    (
                        '"',
                        "'",
                        "“",
                        "‘",
                    )
                )
            ),

        "ends_with_quote":
            int(
                candidate.endswith(
                    (
                        '"',
                        "'",
                        "”",
                        "’",
                    )
                )
            ),

        "starts_with_dash":
            int(
                candidate.startswith(
                    (
                        "-",
                        "–",
                        "—",
                    )
                )
            ),

        "has_edge_punctuation":
            int(
                candidate[:1] in
                ".,!?;:'\"“”‘’()[]{}–—-"
                or
                candidate[-1:]
                in
                ".,!?;:'\"“”‘’()[]{}–—-"
            ),

        # ----------------------------------------------------
        # Sentence position
        # ----------------------------------------------------

        "candidate_occurs_at_start":
            int(
                start <= 2
            ),

        "candidate_occurs_at_end":
            int(
                end >=
                len(sentence) - 2
            ),

        # ----------------------------------------------------
        # Structural features
        # ----------------------------------------------------

        "is_ner":
            int(
                source == "NER"
            ),

        "is_subspan":
            int(
                source.startswith(
                    "SUBSPAN"
                )
            ),

        "is_appositive":
            int(
                source ==
                "APPOSITIVE_EXPANSION"
            ),
    }

    return features


# ============================================================
# Build inference features
# ============================================================

print()
print("Building inference features")
print("-" * 70)

X_dict = []

for index, row in df.iterrows():

    X_dict.append(
        build_inference_features(row)
    )

    if (
        index + 1
    ) % 20000 == 0:

        print(
            f"Processed "
            f"{index + 1} / {len(df)}"
        )


# ============================================================
# Vectorize
# ============================================================

print()
print("Vectorizing inference features")
print("-" * 70)

X = vectorizer.transform(
    X_dict
)

# scikit-learn sparse matrix compatibility
if hasattr(X, "indices"):
    X.indices = X.indices.astype(
        "int32"
    )

if hasattr(X, "indptr"):
    X.indptr = X.indptr.astype(
        "int32"
    )

print(
    f"Feature matrix: {X.shape}"
)


# ============================================================
# Predict ranking scores
# ============================================================

print()
print("Generating ranking scores")
print("-" * 70)

probabilities = (
    model.predict_proba(X)[:, 1]
)

df["rank_score"] = probabilities


# ============================================================
# Evaluate ranking
#
# Each target/sentence can have multiple candidates.
#
# We rank candidates by model probability.
#
# The gold candidate is:
# candidate == target
#
# IMPORTANT:
# target is used ONLY for evaluation.
# It is NOT used to create model features.
# ============================================================

print()
print("=" * 70)
print("Ranking Evaluation")
print("=" * 70)


groups = df.groupby(
    ["sentence", "target"],
    sort=False
)


total_targets = 0
targets_with_candidate = 0

top1 = 0
top3 = 0
top5 = 0

ranks = []


for (sentence, target), group in groups:

    total_targets += 1

    target_text = str(
        target
    )

    candidates = group.copy()

    # --------------------------------------------------------
    # Find exact candidate match
    # --------------------------------------------------------

    exact_matches = candidates[
        candidates["candidate"].astype(str)
        == target_text
    ]

    if len(exact_matches) == 0:
        continue

    targets_with_candidate += 1

    # --------------------------------------------------------
    # Rank candidates
    # --------------------------------------------------------

    ranked = candidates.sort_values(
        "rank_score",
        ascending=False,
    )

    ranked_candidates = (
        ranked["candidate"]
        .astype(str)
        .tolist()
    )

    # --------------------------------------------------------
    # Find gold rank
    # --------------------------------------------------------

    gold_rank = None

    for position, candidate in enumerate(
        ranked_candidates,
        start=1,
    ):

        if candidate == target_text:

            gold_rank = position
            break

    if gold_rank is None:
        continue

    ranks.append(
        gold_rank
    )

    if gold_rank == 1:
        top1 += 1

    if gold_rank <= 3:
        top3 += 1

    if gold_rank <= 5:
        top5 += 1


# ============================================================
# Results
# ============================================================

print()
print("-" * 70)

print(
    f"Total targets: "
    f"{total_targets}"
)

print(
    f"Targets with matching candidate: "
    f"{targets_with_candidate}"
)

if targets_with_candidate > 0:

    print()

    print(
        f"Top-1 accuracy: "
        f"{top1 / targets_with_candidate * 100:.2f}%"
    )

    print(
        f"Top-3 recall: "
        f"{top3 / targets_with_candidate * 100:.2f}%"
    )

    print(
        f"Top-5 recall: "
        f"{top5 / targets_with_candidate * 100:.2f}%"
    )

    print(
        f"Average gold rank: "
        f"{sum(ranks) / len(ranks):.2f}"
    )

    sorted_ranks = sorted(
        ranks
    )

    middle = len(
        sorted_ranks
    ) // 2

    if len(sorted_ranks) % 2:

        median_rank = (
            sorted_ranks[middle]
        )

    else:

        median_rank = (
            sorted_ranks[middle - 1]
            + sorted_ranks[middle]
        ) / 2

    print(
        f"Median gold rank: "
        f"{median_rank:.2f}"
    )


# ============================================================
# Show difficult examples
# ============================================================

print()
print("=" * 70)
print("Worst Ranked Examples")
print("=" * 70)

worst_examples = []

for (sentence, target), group in groups:

    target_text = str(
        target
    )

    exact_matches = group[
        group["candidate"].astype(str)
        == target_text
    ]

    if len(exact_matches) == 0:
        continue

    ranked = group.sort_values(
        "rank_score",
        ascending=False,
    )

    ranked_candidates = (
        ranked["candidate"]
        .astype(str)
        .tolist()
    )

    gold_rank = None

    for position, candidate in enumerate(
        ranked_candidates,
        start=1,
    ):

        if candidate == target_text:

            gold_rank = position
            break

    if gold_rank is None:
        continue

    worst_examples.append(
        (
            gold_rank,
            target_text,
            sentence,
            ranked,
        )
    )


worst_examples.sort(
    key=lambda x: x[0],
    reverse=True,
)


for example_number, example in enumerate(
    worst_examples[:10],
    start=1,
):

    gold_rank, target, sentence, ranked = (
        example
    )

    print()
    print(
        f"Example {example_number}"
    )

    print(
        f"Gold rank: {gold_rank}"
    )

    print(
        f"Target: {target}"
    )

    print(
        f"Sentence: {sentence}"
    )

    print(
        "Top candidates:"
    )

    for position, (_, row) in enumerate(
        ranked.head(5).iterrows(),
        start=1,
    ):

        marker = (
            " <-- GOLD"
            if str(row["candidate"])
            == target
            else ""
        )

        print(
            f"  {position}. "
            f"{str(row['candidate'])!r:45}"
            f"score={row['rank_score']:.4f}"
            f"{marker}"
        )


print()
print("=" * 70)
print("Evaluation Complete")
print("=" * 70)


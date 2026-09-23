import os
import joblib
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/processed/candidate_ranker_train.csv"

MODEL_PATH = "models/candidate_ranker.pkl"

VECTORIZER_PATH = "models/candidate_ranker_vectorizer.pkl"


# ============================================================
# Load dataset
# ============================================================

print("\nLoading candidate ranking dataset")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")


# ============================================================
# Feature engineering
#
# IMPORTANT:
# These features MUST NOT use the gold target.
#
# During real inference we only know:
#   sentence
#   candidate
#   candidate_start
#   candidate_end
#   source
#   ner_label
#
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

    candidate_words_list = candidate.split()

    candidate_words = len(candidate_words_list)

    candidate_lower = candidate.lower()

    # --------------------------------------------------------
    # Basic features
    # --------------------------------------------------------

    features = {

        # ----------------------------------------------------
        # Candidate length
        # ----------------------------------------------------

        "candidate_length":
            candidate_length,

        "candidate_words":
            candidate_words,

        "candidate_length_squared":
            candidate_length ** 2,

        "candidate_words_squared":
            candidate_words ** 2,

        # ----------------------------------------------------
        # Position
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Source
        # ----------------------------------------------------

        "source":
            source,

        "is_ner":
            int(source == "NER"),

        "is_noun_chunk":
            int(source == "NOUN_CHUNK"),

        "is_pronoun":
            int(source == "PRONOUN"),

        "is_subspan":
            int(
                source.startswith("SUBSPAN")
            ),

        "is_subspan_token":
            int(
                source == "SUBSPAN_TOKEN"
            ),

        "is_subspan_proper_name":
            int(
                source == "SUBSPAN_PROPER_NAME"
            ),

        "is_coordination":
            int(
                source == "COORDINATION"
            ),

        "is_appositive":
            int(
                source == "APPOSITIVE_EXPANSION"
            ),

        "is_title_expansion":
            int(
                source == "TITLE_EXPANSION"
            ),

        "is_normalized":
            int(
                "NORMALIZED" in source
            ),

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

        "is_product":
            int(
                ner_label == "PRODUCT"
            ),

        "is_event":
            int(
                ner_label == "EVENT"
            ),

        "is_work_of_art":
            int(
                ner_label == "WORK_OF_ART"
            ),

        "is_law":
            int(
                ner_label == "LAW"
            ),

        "is_loc":
            int(
                ner_label == "LOC"
            ),

        "is_fac":
            int(
                ner_label == "FAC"
            ),

        # ----------------------------------------------------
        # Candidate word structure
        # ----------------------------------------------------

        "is_single_word":
            int(
                candidate_words == 1
            ),

        "is_two_words":
            int(
                candidate_words == 2
            ),

        "is_three_words":
            int(
                candidate_words == 3
            ),

        "is_multi_word":
            int(
                candidate_words > 1
            ),

        "is_long_candidate":
            int(
                candidate_words >= 4
            ),

        "is_very_long_candidate":
            int(
                candidate_words >= 8
            ),

        # ----------------------------------------------------
        # Articles / titles
        # ----------------------------------------------------

        "starts_with_the":
            int(
                candidate_lower.startswith("the ")
            ),

        "starts_with_a":
            int(
                candidate_lower.startswith("a ")
            ),

        "starts_with_an":
            int(
                candidate_lower.startswith("an ")
            ),

        "starts_with_title":
            int(
                candidate_lower.split()[0]
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
                    "mayor",
                    "minister",
                    "prime",
                    "professor",
                    "prof.",
                    "chief",
                    "captain",
                    "col",
                    "colonel",
                }
                if candidate_words_list
                else False
            ),

        # ----------------------------------------------------
        # Punctuation
        # ----------------------------------------------------

        "contains_hyphen":
            int(
                "-" in candidate
            ),

        "contains_apostrophe":
            int(
                "'" in candidate
                or "’" in candidate
            ),

        "starts_with_quote":
            int(
                candidate.startswith((
                    '"',
                    "'",
                    "“",
                    "‘",
                ))
            ),

        "ends_with_quote":
            int(
                candidate.endswith((
                    '"',
                    "'",
                    "”",
                    "’",
                ))
            ),

        "starts_with_dash":
            int(
                candidate.startswith((
                    "-",
                    "–",
                    "—",
                ))
            ),

        "has_edge_punctuation":
            int(
                (
                    len(candidate) > 0
                    and (
                        not candidate[0].isalnum()
                        or not candidate[-1].isalnum()
                    )
                )
            ),

        # ----------------------------------------------------
        # Capitalization
        # ----------------------------------------------------

        "first_word_capitalized":
            int(
                candidate_words_list[0][0].isupper()
                if candidate_words_list
                and candidate_words_list[0]
                else False
            ),

        "contains_uppercase":
            int(
                any(
                    char.isupper()
                    for char in candidate
                )
            ),

        "all_upper":
            int(
                candidate.isupper()
            ),

        "all_lower":
            int(
                candidate.islower()
            ),

        "title_case":
            int(
                candidate.istitle()
            ),

        # ----------------------------------------------------
        # Sentence position
        # ----------------------------------------------------

        "candidate_occurs_at_start":
            int(
                start <= 2
            ),

        "candidate_occurs_near_start":
            int(
                start <= 20
            ),

        "candidate_occurs_at_end":
            int(
                end >= sentence_length - 2
            ),

        "candidate_occurs_near_end":
            int(
                end >= sentence_length - 20
            ),

        # ----------------------------------------------------
        # Local sentence context
        #
        # These use only sentence + candidate position.
        # ----------------------------------------------------

        "preceded_by_space":
            int(
                start > 0
                and sentence[start - 1].isspace()
            ),

        "followed_by_space":
            int(
                end < len(sentence)
                and sentence[end:end + 1].isspace()
            ),

        "preceded_by_quote":
            int(
                start > 0
                and sentence[start - 1]
                in {'"', "'", "“", "‘"}
            ),

        "followed_by_quote":
            int(
                end < len(sentence)
                and sentence[end:end + 1]
                in {'"', "'", "”", "’"}
            ),

        "preceded_by_dash":
            int(
                start > 0
                and sentence[start - 1]
                in {"-", "–", "—"}
            ),

        "followed_by_dash":
            int(
                end < len(sentence)
                and sentence[end:end + 1]
                in {"-", "–", "—"}
            ),
    }

    return features


# ============================================================
# Build feature dictionaries
# ============================================================

print("\nBuilding features")
print("-" * 70)

X_dict = []

total_rows = len(df)

for index, row in df.iterrows():

    X_dict.append(
        build_features(row)
    )

    if (index + 1) % 20000 == 0:

        print(
            f"Processed "
            f"{index + 1} / {total_rows}"
        )


y = df["label"].astype(int)


# ============================================================
# Vectorize
# ============================================================

print("\nVectorizing features")
print("-" * 70)

vectorizer = DictVectorizer(
    sparse=True
)

X = vectorizer.fit_transform(
    X_dict
)


# ============================================================
# Fix sparse matrix indices
#
# Required by some sklearn/scipy combinations.
# ============================================================

X.indices = X.indices.astype(
    "int32",
    copy=False
)

X.indptr = X.indptr.astype(
    "int32",
    copy=False
)


print(
    f"Feature matrix: {X.shape}"
)

print(
    f"Positive examples: {y.sum()}"
)

print(
    f"Negative examples: "
    f"{len(y) - y.sum()}"
)


# ============================================================
# Train Logistic Regression
# ============================================================

print("\nTraining candidate ranker")
print("-" * 70)

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    C=1.0,
    solver="liblinear",
)


model.fit(
    X,
    y,
)


print(
    "Training complete."
)


# ============================================================
# Show strongest features
# ============================================================

print("\nStrongest ranking features")
print("-" * 70)

feature_names = vectorizer.get_feature_names_out()

coefficients = model.coef_[0]

feature_importance = sorted(
    zip(
        feature_names,
        coefficients
    ),
    key=lambda x: abs(x[1]),
    reverse=True,
)


for feature, coefficient in feature_importance[:40]:

    print(
        f"{feature:<40}"
        f"{coefficient:>10.4f}"
    )


# ============================================================
# Training evaluation
#
# NOTE:
# This is NOT the real ranking evaluation.
# It is only a sanity check that the classifier trained.
# ============================================================

predictions = model.predict(X)

probabilities = (
    model.predict_proba(X)[:, 1]
)


print("\nTraining Classification Report")
print("-" * 70)

print(
    classification_report(
        y,
        predictions,
        target_names=[
            "Negative Candidate",
            "Gold Candidate",
        ],
    )
)


print(
    "Training ROC-AUC:",
    f"{roc_auc_score(y, probabilities):.4f}",
)


# ============================================================
# Create model directory
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)


# ============================================================
# Save model
# ============================================================

joblib.dump(
    model,
    MODEL_PATH,
)


joblib.dump(
    vectorizer,
    VECTORIZER_PATH,
)


# ============================================================
# Finished
# ============================================================

print("\nSaved")
print("-" * 70)

print(
    f"Model:      {MODEL_PATH}"
)

print(
    f"Vectorizer: {VECTORIZER_PATH}"
)

print("\nTraining finished successfully.")
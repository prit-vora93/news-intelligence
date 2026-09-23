"""
ranker_offsets.py
-----------------
Learn useful offset/ranking signals from the candidate ranking dataset.

Goal:
    Understand how candidate position, length, source, and NER information
    relate to whether a candidate is the gold target.

This script is an analysis tool. It does NOT modify the trained ranker.
"""

import pandas as pd


# ============================================================
# 1. Load candidate ranking dataset
# ============================================================

DATA_PATH = "data/processed/candidate_ranker_train.csv"

print("=" * 70)
print("Ranker Offset Analysis")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"\nLoaded rows: {len(df)}")
print(f"Columns: {list(df.columns)}")


# ============================================================
# 2. Basic candidate information
# ============================================================

df["candidate"] = df["candidate"].fillna("").astype(str)
df["target"] = df["target"].fillna("").astype(str)

df["candidate_start"] = pd.to_numeric(
    df["candidate_start"],
    errors="coerce",
)

df["candidate_end"] = pd.to_numeric(
    df["candidate_end"],
    errors="coerce",
)


# Candidate character length
df["candidate_length"] = (
    df["candidate_end"] - df["candidate_start"]
)


# Target character length
df["target_length"] = df["target"].str.len()


# Candidate word count
df["candidate_words"] = (
    df["candidate"]
    .str.split()
    .str.len()
)


# Target word count
df["target_words"] = (
    df["target"]
    .str.split()
    .str.len()
)


# ============================================================
# 3. Candidate position inside sentence
# ============================================================

df["sentence_length"] = (
    df["sentence"]
    .fillna("")
    .astype(str)
    .str.len()
)

df["relative_start"] = (
    df["candidate_start"]
    / df["sentence_length"].replace(0, 1)
)


# ============================================================
# 4. Exact target/candidate relationship
# ============================================================

df["candidate_lower"] = (
    df["candidate"]
    .str.lower()
    .str.strip()
)

df["target_lower"] = (
    df["target"]
    .str.lower()
    .str.strip()
)

df["text_exact_match"] = (
    df["candidate_lower"]
    == df["target_lower"]
)


df["target_inside_candidate"] = df.apply(
    lambda row:
        row["target_lower"] in row["candidate_lower"],
    axis=1,
)


# ============================================================
# 5. Positive vs negative candidates
# ============================================================

positive = df[df["label"] == 1].copy()
negative = df[df["label"] == 0].copy()

print("\n" + "=" * 70)
print("Candidate Statistics")
print("=" * 70)

print(f"\nPositive candidates: {len(positive)}")
print(f"Negative candidates: {len(negative)}")


# ============================================================
# 6. Offset statistics
# ============================================================

print("\n" + "-" * 70)
print("Candidate Start Offset")
print("-" * 70)

print(
    f"Positive mean start: "
    f"{positive['candidate_start'].mean():.2f}"
)

print(
    f"Negative mean start: "
    f"{negative['candidate_start'].mean():.2f}"
)


print("\n" + "-" * 70)
print("Relative Candidate Position")
print("-" * 70)

print(
    f"Positive mean relative position: "
    f"{positive['relative_start'].mean():.4f}"
)

print(
    f"Negative mean relative position: "
    f"{negative['relative_start'].mean():.4f}"
)


# ============================================================
# 7. Candidate length
# ============================================================

print("\n" + "-" * 70)
print("Candidate Length")
print("-" * 70)

print(
    f"Positive mean characters: "
    f"{positive['candidate_length'].mean():.2f}"
)

print(
    f"Negative mean characters: "
    f"{negative['candidate_length'].mean():.2f}"
)

print(
    f"Positive mean words: "
    f"{positive['candidate_words'].mean():.2f}"
)

print(
    f"Negative mean words: "
    f"{negative['candidate_words'].mean():.2f}"
)


# ============================================================
# 8. Exact text matching
# ============================================================

print("\n" + "-" * 70)
print("Exact Candidate Text Match")
print("-" * 70)

exact_positive = positive["text_exact_match"].mean()

print(
    f"Positive candidates exactly matching target: "
    f"{exact_positive * 100:.2f}%"
)


# ============================================================
# 9. Target contained inside candidate
# ============================================================

print("\n" + "-" * 70)
print("Target Containment")
print("-" * 70)

contained_positive = (
    positive["target_inside_candidate"].mean()
)

print(
    f"Positive candidates containing target: "
    f"{contained_positive * 100:.2f}%"
)


# ============================================================
# 10. Source analysis
# ============================================================

print("\n" + "=" * 70)
print("Source Analysis")
print("=" * 70)

source_stats = (
    df.groupby("source")["label"]
    .agg(
        count="count",
        positives="sum",
        positive_rate="mean",
    )
    .sort_values(
        "positive_rate",
        ascending=False,
    )
)

print()

for source, row in source_stats.iterrows():

    print(
        f"{source:<25}"
        f" count={int(row['count']):6d}"
        f" positives={int(row['positives']):6d}"
        f" rate={row['positive_rate'] * 100:6.2f}%"
    )


# ============================================================
# 11. NER label analysis
# ============================================================

print("\n" + "=" * 70)
print("NER Label Analysis")
print("=" * 70)

df["ner_label"] = (
    df["ner_label"]
    .fillna("NONE")
    .astype(str)
)

ner_stats = (
    df.groupby("ner_label")["label"]
    .agg(
        count="count",
        positives="sum",
        positive_rate="mean",
    )
    .sort_values(
        "positive_rate",
        ascending=False,
    )
)

print()

for label, row in ner_stats.iterrows():

    print(
        f"{label:<15}"
        f" count={int(row['count']):6d}"
        f" positives={int(row['positives']):6d}"
        f" rate={row['positive_rate'] * 100:6.2f}%"
    )


# ============================================================
# 12. Candidate length buckets
# ============================================================

print("\n" + "=" * 70)
print("Candidate Length Buckets")
print("=" * 70)


def length_bucket(words):

    if words == 1:
        return "1 word"

    if words == 2:
        return "2 words"

    if words <= 4:
        return "3-4 words"

    if words <= 8:
        return "5-8 words"

    return "9+ words"


df["length_bucket"] = (
    df["candidate_words"]
    .apply(length_bucket)
)

length_stats = (
    df.groupby("length_bucket")["label"]
    .agg(
        count="count",
        positives="sum",
        positive_rate="mean",
    )
)

print()

for bucket, row in length_stats.iterrows():

    print(
        f"{bucket:<12}"
        f" count={int(row['count']):6d}"
        f" positives={int(row['positives']):6d}"
        f" rate={row['positive_rate'] * 100:6.2f}%"
    )


# ============================================================
# 13. Show positive examples with offsets
# ============================================================

print("\n" + "=" * 70)
print("Sample Positive Candidates")
print("=" * 70)

sample = positive[
    [
        "target",
        "candidate",
        "candidate_start",
        "candidate_end",
        "candidate_words",
        "source",
        "ner_label",
    ]
].head(20)

print()

for _, row in sample.iterrows():

    print(
        f"Target:     {row['target']}"
    )

    print(
        f"Candidate:  {row['candidate']}"
    )

    print(
        f"Offset:     "
        f"{int(row['candidate_start'])}"
        f"-"
        f"{int(row['candidate_end'])}"
    )

    print(
        f"Words:      {int(row['candidate_words'])}"
    )

    print(
        f"Source:     {row['source']}"
    )

    print(
        f"NER label:  {row['ner_label']}"
    )

    print("-" * 50)


# ============================================================
# 14. Finish
# ============================================================

print("\n" + "=" * 70)
print("Analysis Complete")
print("=" * 70)
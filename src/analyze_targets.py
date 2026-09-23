import pandas as pd
import re
from collections import Counter


DATA_PATH = "data/processed/test_target.csv"

df = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# Basic target statistics
# --------------------------------------------------

df["target_words"] = df["target"].astype(str).str.split().str.len()
df["target_chars"] = df["target"].astype(str).str.len()


print("\nTarget Analysis")
print("=" * 60)

print(f"Total targets: {len(df)}")

print(
    f"Unique targets: "
    f"{df['target'].nunique()}"
)

print(
    f"Average target length: "
    f"{df['target_words'].mean():.2f} words"
)


# --------------------------------------------------
# Single vs multi-word
# --------------------------------------------------

single_word = (df["target_words"] == 1).sum()
multi_word = (df["target_words"] > 1).sum()

print("\nTarget Length")
print("-" * 60)

print(
    f"Single-word targets: "
    f"{single_word} "
    f"({single_word / len(df) * 100:.2f}%)"
)

print(
    f"Multi-word targets: "
    f"{multi_word} "
    f"({multi_word / len(df) * 100:.2f}%)"
)


# --------------------------------------------------
# Target word-count distribution
# --------------------------------------------------

print("\nWord Count Distribution")
print("-" * 60)

word_counts = df["target_words"].value_counts().sort_index()

for words, count in word_counts.items():

    print(
        f"{words} word(s): "
        f"{count} "
        f"({count / len(df) * 100:.2f}%)"
    )


# --------------------------------------------------
# Pronoun detection
# --------------------------------------------------

pronouns = {
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "you",
    "your",
    "yours",
    "yourself",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
}


def is_pronoun(target):

    words = str(target).lower().split()

    return (
        len(words) == 1
        and words[0] in pronouns
    )


df["is_pronoun"] = df["target"].apply(is_pronoun)

pronoun_count = df["is_pronoun"].sum()

print("\nPronoun Targets")
print("-" * 60)

print(
    f"Pronoun targets: "
    f"{pronoun_count} "
    f"({pronoun_count / len(df) * 100:.2f}%)"
)


# --------------------------------------------------
# Common noun / organization-style keywords
# --------------------------------------------------

common_target_words = {
    "company",
    "companies",
    "government",
    "president",
    "presidents",
    "minister",
    "ministers",
    "investor",
    "investors",
    "people",
    "public",
    "police",
    "official",
    "officials",
    "administration",
    "party",
    "parties",
    "market",
    "markets",
}


def contains_common_target_word(target):

    words = set(
        str(target).lower().split()
    )

    return bool(
        words.intersection(
            common_target_words
        )
    )


df["common_noun_target"] = df["target"].apply(
    contains_common_target_word
)

common_count = df["common_noun_target"].sum()

print("\nCommon/Noun-like Targets")
print("-" * 60)

print(
    f"Targets containing common target words: "
    f"{common_count} "
    f"({common_count / len(df) * 100:.2f}%)"
)


# --------------------------------------------------
# Top target mentions
# --------------------------------------------------

print("\nTop 30 Target Mentions")
print("-" * 60)

target_counts = (
    df["target"]
    .value_counts()
    .head(30)
)

for target, count in target_counts.items():

    print(
        f"{count:4d}  {target}"
    )


# --------------------------------------------------
# Longest targets
# --------------------------------------------------

print("\nLongest Targets")
print("-" * 60)

longest = (
    df.sort_values(
        "target_words",
        ascending=False
    )
    [["target", "target_words"]]
    .drop_duplicates()
    .head(20)
)

for _, row in longest.iterrows():

    print(
        f"{int(row['target_words']):2d} words  "
        f"{row['target']}"
    )


# --------------------------------------------------
# Save summary
# --------------------------------------------------

summary = {
    "total_targets": len(df),
    "unique_targets": df["target"].nunique(),
    "single_word": int(single_word),
    "multi_word": int(multi_word),
    "pronouns": int(pronoun_count),
    "common_noun_like": int(common_count),
    "average_target_words": float(
        df["target_words"].mean()
    ),
}

print("\nSummary")
print("-" * 60)

for key, value in summary.items():
    print(f"{key}: {value}")
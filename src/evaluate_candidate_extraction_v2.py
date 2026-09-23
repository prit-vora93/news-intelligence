
import re

import pandas as pd
import spacy


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/processed/test_target.csv"

nlp = spacy.load("en_core_web_sm")


# ============================================================
# Pronouns
# ============================================================

PRONOUNS = {
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


# ============================================================
# Titles / Honorifics
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
# Utility: normalize candidate text
# ============================================================

def normalize_text(text):
    """
    Remove punctuation surrounding a candidate.

    Examples:

        '" Powell'
            -> 'Powell'

        '“President Trump'
            -> 'President Trump'

        'Powell,"'
            -> 'Powell'
    """

    text = str(text)

    # Remove whitespace first.
    text = text.strip()

    # Remove leading quotes / punctuation.
    text = re.sub(
        r'^[\s"“”\'‘’`´]+',
        "",
        text,
    )

    # Remove trailing quotes / punctuation.
    text = re.sub(
        r'[\s"“”\'‘’`´]+$',
        "",
        text,
    )

    # Normalize repeated whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# Utility: create candidate
# ============================================================

def add_candidate(
    candidates,
    start,
    end,
    text,
    source,
):
    """
    Add a candidate using its exact character offsets.

    Invalid spans are ignored.
    """

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
        }
    )


# ============================================================
# Utility: add normalized version
# ============================================================

def add_normalized_candidate(
    candidates,
    sentence,
    start,
    end,
    source,
):
    """
    Add the original candidate and a normalized candidate.

    This is important for targets such as:

        " Powell

    where the dataset span includes the quote but the
    linguistic entity is actually Powell.
    """

    original = sentence[start:end]

    # Original candidate
    add_candidate(
        candidates,
        start,
        end,
        original,
        source,
    )

    normalized = normalize_text(original)

    if not normalized:
        return

    # Find normalized text inside original text.
    local_start = original.find(normalized)

    if local_start == -1:
        return

    normalized_start = start + local_start
    normalized_end = normalized_start + len(normalized)

    add_candidate(
        candidates,
        normalized_start,
        normalized_end,
        normalized,
        source + "_NORMALIZED",
    )


# ============================================================
# Sub-span candidates
# ============================================================

def add_subspan_candidates(
    candidates,
    doc,
    sentence,
):
    """
    Generate smaller candidates from NER entities and noun chunks.

    Examples:

        John McCain
            -> John
            -> McCain
            -> John McCain

        Sean Spicer
            -> Sean
            -> Spicer
            -> Sean Spicer
    """

    chunks = list(doc.ents) + list(doc.noun_chunks)

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
        # Consecutive proper-name sequence
        # ----------------------------------------------------

        start = None
        end = None

        for token in tokens:

            if token.pos_ == "PROPN":

                if start is None:
                    start = token.idx

                end = token.idx + len(token.text)

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

        # Flush final proper-name sequence.
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

    # ========================================================
    # 1. Named entities
    # ========================================================

    for ent in doc.ents:

        add_normalized_candidate(
            candidates,
            sentence,
            ent.start_char,
            ent.end_char,
            "NER",
        )

    # ========================================================
    # 2. Noun chunks
    # ========================================================

    for chunk in doc.noun_chunks:

        add_normalized_candidate(
            candidates,
            sentence,
            chunk.start_char,
            chunk.end_char,
            "NOUN_CHUNK",
        )

    # ========================================================
    # 3. Pronouns
    # ========================================================

    for token in doc:

        if (
            token.text.lower() in PRONOUNS
            and token.pos_ == "PRON"
        ):

            add_candidate(
                candidates,
                token.idx,
                token.idx + len(token.text),
                token.text,
                "PRONOUN",
            )

    # ========================================================
    # 4. Honorific / title expansion
    #
    # Example:
    #
    # Trump
    #     ↓
    # President Trump
    #
    # John McCain
    #     ↓
    # Mr John McCain
    # ========================================================

    for ent in doc.ents:

        if ent.label_ not in {
            "PERSON",
            "ORG",
        }:
            continue

        if ent.start == 0:
            continue

        previous = doc[ent.start - 1]

        if previous.text.lower() in HONORIFICS:

            expanded_start = previous.idx
            expanded_end = ent.end_char

            add_candidate(
                candidates,
                expanded_start,
                expanded_end,
                sentence[
                    expanded_start:expanded_end
                ],
                "TITLE_EXPANSION",
            )

    # ========================================================
    # 5. Coordination
    #
    # Example:
    #
    # Obama and Cameron
    #
    # Generate:
    #
    # Obama
    # Cameron
    # Obama and Cameron
    # ========================================================

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
            left + 1:
            right
        ]

        conjunction = any(
            t.text.lower() in {
                "and",
                "or",
            }
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

    # ========================================================
    # 6. Person + appositive expansion
    #
    # Example:
    #
    # Michael Caputo,
    # a former communications official
    #
    # Generate:
    #
    # Michael Caputo
    # Michael Caputo, a former communications official
    # ========================================================

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

            # Stop at sentence boundary.
            if token.text in {
                ".",
                "!",
                "?",
            }:
                break

            # Looking for comma after person.
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
                    expanded_start:expanded_end
                ],
                "APPOSITIVE_EXPANSION",
            )

            break

    # ========================================================
    # 7. NEW: sub-span candidates
    # ========================================================

    add_subspan_candidates(
        candidates,
        doc,
        sentence,
    )

    # ========================================================
    # Remove duplicate candidates
    # ========================================================

    unique = {}

    for candidate in candidates:

        key = (
            candidate["start"],
            candidate["end"],
            candidate["text"],
        )

        if key not in unique:
            unique[key] = candidate

    return list(unique.values())


# ============================================================
# Exact matching
# ============================================================

def exact_match(
    candidate,
    target_start,
    target_end,
):
    return (
        candidate["start"] == target_start
        and candidate["end"] == target_end
    )


# ============================================================
# Load dataset
# ============================================================

df = pd.read_csv(DATA_PATH)
candidate_counts = []
total = len(df)


# ============================================================
# Evaluate
# ============================================================

v2_recall = 0

still_missed = []


for _, row in df.iterrows():

    sentence = row["sentence"]

    target = row["target"]

    target_start = int(
        row["target_from"]
    )

    target_end = int(
        row["target_to"]
    )

    candidates = get_candidates(
        sentence
    )
    candidate_counts.append(len(candidates))
    found = any(
        exact_match(
            candidate,
            target_start,
            target_end,
        )
        for candidate in candidates
    )

    if found:

        v2_recall += 1

    else:

        still_missed.append(
            {
                "target": target,
                "sentence": sentence,
                "words": len(
                    str(target).split()
                ),
            }
        )


# ============================================================
# Results
# ============================================================

print("\nCandidate Extraction V2")
print("=" * 70)

print(
    f"Total targets: {total}"
)


print("\nPrevious Baseline")
print("-" * 70)

print(
    "NER + noun chunks + pronouns: "
    "86.05%"
)

print(
    "V2 before sub-spans: "
    "87.80%"
)


print("\nV2 With Sub-span Candidates")
print("-" * 70)

print(
    f"Exact recall: "
    f"{v2_recall} / {total} "
    f"({v2_recall / total * 100:.2f}%)"
)

print(
    f"Missed: "
    f"{len(still_missed)}"
)


# ============================================================
# Missed target distribution
# ============================================================

print("\nRemaining Missed Targets")
print("-" * 70)

word_counts = {}

for item in still_missed:

    words = item["words"]

    word_counts[words] = (
        word_counts.get(words, 0)
        + 1
    )


for words, count in sorted(
    word_counts.items()
):

    print(
        f"{words} word(s): "
        f"{count}"
    )


# ============================================================
# Show remaining misses
# ============================================================

print("\nFirst 50 Remaining Misses")
print("-" * 70)

for item in still_missed[:50]:

    print(
        f"\nTarget: {item['target']}"
    )

    print(
        f"Sentence: "
        f"{item['sentence']}"
    )

# ============================================================
# Candidate Count Analysis
# ============================================================

import statistics


print("\nCandidate Count Analysis")
print("=" * 70)

print(
    f"Average candidates per sentence: "
    f"{statistics.mean(candidate_counts):.2f}"
)

print(
    f"Median candidates per sentence: "
    f"{statistics.median(candidate_counts):.2f}"
)

print(
    f"Minimum candidates: "
    f"{min(candidate_counts)}"
)

print(
    f"Maximum candidates: "
    f"{max(candidate_counts)}"
)


# ------------------------------------------------------------
# Candidate count buckets
# ------------------------------------------------------------

buckets = {
    "1-5": 0,
    "6-10": 0,
    "11-20": 0,
    "21-30": 0,
    "31+": 0,
}


for count in candidate_counts:

    if count <= 5:
        buckets["1-5"] += 1

    elif count <= 10:
        buckets["6-10"] += 1

    elif count <= 20:
        buckets["11-20"] += 1

    elif count <= 30:
        buckets["21-30"] += 1

    else:
        buckets["31+"] += 1


print("\nCandidate Count Distribution")
print("-" * 70)

total_sentences = len(candidate_counts)

for bucket, count in buckets.items():

    percentage = (
        count / total_sentences * 100
    )

    print(
        f"{bucket:>6}: "
        f"{count:4d} "
        f"({percentage:.2f}%)"
    )
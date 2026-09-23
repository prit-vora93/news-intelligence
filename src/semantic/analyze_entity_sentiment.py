"""
semantic/analyze_entity_sentiment.py
--------------------------------------

Aggregate predicted sentiment BY ENTITY across the whole corpus --
the actual "intelligence" deliverable of this project: not just
per-sentence predictions, but "what's the overall sentiment toward
X across everything we've processed?"

Canonicalization: for each record, the "entity" grouped on is:
  1. The first embedded_entity, if present (already a clean short
     form spaCy recognized, e.g. "Trump" from "President Trump").
  2. Else target_text itself, IF it looks like a short (<=3 word)
     capitalized entity-like phrase (a fallback for bare entity
     targets spaCy didn't separately tag but which still look like
     names).
  3. Otherwise excluded from entity-level aggregation entirely --
     full descriptive phrases ("the possibility of...") aren't
     useful entity buckets and would just add noise.

Uses predicted_sentiment_label (always available when the dataset
was generated with --predict) rather than gold sentiment_label
(only available for ~21% of records), since the whole point is
aggregating across the FULL corpus, including sentences no human
ever labeled -- that's the actual value of having a trained model
in the loop.

Usage:
    PYTHONPATH=src python src/semantic/analyze_entity_sentiment.py \\
        dataset_final.jsonl --top 20 --min-mentions 3
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


# Words that function as institutional/adjectival modifiers, not
# given names -- if the LONGER key's first word is one of these,
# do NOT merge it into a shorter suffix bucket. Confirmed necessary
# via real data: "White House" (80 mentions, the executive branch)
# was incorrectly merging into "House" (70 mentions, almost
# certainly House of Representatives -- a different institution
# entirely) purely because both end in the word "house". This only
# blocks the case where the modifier word is FIRST (the adjective
# role in institutional names); it does NOT block a real person
# whose surname happens to be one of these words appearing as the
# LAST word of a longer name (e.g. "John White" merging into
# "White" is unaffected, since "White" there is in the suffix
# position, not the blocked prefix position).
_INSTITUTIONAL_MODIFIER_WORDS = {
    "white", "supreme", "united", "national", "federal", "state",
    "democratic", "republican", "u.s.", "us", "house", "senate",
}


def canonical_entity(record: dict) -> str | None:

    embedded = record.get("embedded_entities") or []
    if embedded:
        return embedded[0].strip()

    text = (record.get("target_text") or "").strip()
    words = text.split()

    if 1 <= len(words) <= 3 and text[:1].isupper():
        return text

    return None


def merge_aliases(stats: dict[str, dict]) -> dict[str, dict]:
    """
    Merge longer name variants into a shorter, already-existing
    bucket when the longer key ENDS WITH the shorter key as its
    trailing word(s) -- e.g. "donald trump" merges into "trump",
    "hillary clinton" merges into "clinton".

    This handles the common Western surname-last naming pattern
    without needing entity-type information or core pipeline
    changes -- confirmed as a real gap via the actual aggregate
    analysis output ("Trump": 1132 mentions and "Donald Trump": 204
    mentions sitting as two separate buckets for the same person).

    Known limitation, not fixed here: doesn't merge cases with NO
    shared trailing word at all (e.g. a nickname vs. full name), and
    could rarely over-merge two different people who happen to
    share a surname -- an acceptable, documented tradeoff given how
    much more common the "same person, different mention length"
    case is in this corpus.

    Only merges into a key that ALREADY EXISTS as its own bucket
    (not into an arbitrary constructed suffix) and only when the
    total mention count exceeds min_mentions filtering happens
    later, so a coincidental short surname with very few mentions
    won't quietly absorb something bigger -- the merge target must
    already be a real, independently-seen bucket.
    """

    merged = dict(stats)

    # Process longer keys first, since a 3-word key might need to
    # check both its 2-word and 1-word suffixes.
    for long_key in sorted(merged.keys(), key=lambda k: -len(k.split())):

        if long_key not in merged:
            continue

        words = long_key.split()

        if len(words) < 2:
            continue

        if words[0].lower() in _INSTITUTIONAL_MODIFIER_WORDS:
            continue

        for suffix_len in range(1, len(words)):

            suffix = " ".join(words[-suffix_len:])

            if suffix == long_key or suffix not in merged:
                continue

            source = merged.pop(long_key, None)
            if source is None:
                break

            target = merged[suffix]

            for label in ("Positive", "Neutral", "Negative"):
                target[label] += source[label]

            target["total"] += source["total"]
            target["confidences"].extend(source["confidences"])

            break

    return merged


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Aggregate predicted sentiment by entity across the corpus."
    )
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--min-mentions", type=int, default=3)
    args = parser.parse_args()

    print(f"Loading {args.dataset_path}...")

    records = []
    with open(args.dataset_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Loaded {len(records)} records")

    stats: dict[str, dict] = defaultdict(
        lambda: {"Positive": 0, "Neutral": 0, "Negative": 0, "total": 0, "confidences": []}
    )

    skipped_no_prediction = 0
    skipped_no_entity = 0

    for r in records:

        # Accept either field-naming schema: dataset_final.jsonl (from
        # build_dataset.py --predict) uses predicted_sentiment_label /
        # predicted_confidence; live_results.jsonl (from
        # live_ingest.py) uses sentiment / sentiment_confidence. One
        # analysis script works against both rather than maintaining
        # near-duplicate scripts per data source.
        pred = r.get("predicted_sentiment_label") or r.get("sentiment")
        if pred is None:
            skipped_no_prediction += 1
            continue

        entity = canonical_entity(r)
        if entity is None:
            skipped_no_entity += 1
            continue

        key = entity.lower()
        stats[key]["total"] += 1
        stats[key][pred] += 1
        stats[key]["confidences"].append(
            r.get("predicted_confidence") or r.get("sentiment_confidence") or 0.0
        )
        stats[key]["display_name"] = entity

    print(f"Records with no live prediction: {skipped_no_prediction}")
    print(f"Records excluded from entity aggregation (not entity-like): {skipped_no_entity}")
    print()

    stats = merge_aliases(stats)

    eligible = {k: v for k, v in stats.items() if v["total"] >= args.min_mentions}
    ranked = sorted(eligible.items(), key=lambda kv: kv[1]["total"], reverse=True)

    print("=" * 90)
    print(f"TOP {args.top} ENTITIES BY MENTION COUNT (predicted sentiment breakdown)")
    print("=" * 90)

    header = (
        f"{'ENTITY':30}{'MENTIONS':>10}{'POSITIVE':>10}{'NEUTRAL':>10}"
        f"{'NEGATIVE':>10}{'AVG CONF':>10}"
    )
    print(header)
    print("-" * len(header))

    for key, v in ranked[: args.top]:

        name = v["display_name"]
        total = v["total"]
        pos_pct = v["Positive"] / total
        neu_pct = v["Neutral"] / total
        neg_pct = v["Negative"] / total
        avg_conf = (
            sum(v["confidences"]) / len(v["confidences"])
            if v["confidences"] else 0.0
        )

        print(
            f"{name:30}{total:>10}"
            f"{pos_pct:>9.0%} {neu_pct:>9.0%} {neg_pct:>9.0%} {avg_conf:>9.0%}"
        )

    print()
    print(f"Total distinct entities (>= {args.min_mentions} mentions): {len(eligible)}")


if __name__ == "__main__":
    main()
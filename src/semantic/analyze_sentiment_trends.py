"""
semantic/analyze_sentiment_trends.py
---------------------------------------

Day-over-day sentiment trends by entity, using the fetched_at
timestamp that live_ingest.py records on every entry. This is only
meaningful once live_results.jsonl has accumulated data across
MULTIPLE separate runs/days -- a single snapshot has no trend to
show. Reuses canonical_entity() and merge_aliases() from
analyze_entity_sentiment.py rather than duplicating that logic, so
both scripts stay consistent with each other.

Usage:
    PYTHONPATH=src python src/semantic/analyze_sentiment_trends.py \\
        live_results.jsonl --entity Burnham

    # Or see which entities have enough multi-day data to be worth
    # looking at trends for at all:
    PYTHONPATH=src python src/semantic/analyze_sentiment_trends.py \\
        live_results.jsonl --list-entities
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from semantic.analyze_entity_sentiment import canonical_entity, merge_aliases, derive_alias_mapping


def parse_date(fetched_at: str) -> str | None:
    """
    Extract just the DATE portion (YYYY-MM-DD) from an ISO 8601
    timestamp, so records from the same day bucket together
    regardless of exact time.
    """

    if not fetched_at:
        return None

    return fetched_at[:10] if len(fetched_at) >= 10 else None


def build_trend_stats(records: list[dict]) -> dict[str, dict[str, dict]]:
    """
    Returns: {entity_key: {date: {"Positive": n, "Neutral": n, "Negative": n, "total": n, "display_name": str}}}
    """

    stats: dict[str, dict[str, dict]] = defaultdict(dict)

    for r in records:

        pred = r.get("predicted_sentiment_label") or r.get("sentiment")
        if pred is None:
            continue

        entity = canonical_entity(r)
        if entity is None:
            continue

        date = parse_date(r.get("fetched_at", ""))
        if date is None:
            continue

        key = entity.lower()

        if date not in stats[key]:
            stats[key][date] = {
                "Positive": 0, "Neutral": 0, "Negative": 0,
                "total": 0, "display_name": entity,
            }

        stats[key][date][pred] += 1
        stats[key][date]["total"] += 1

    return stats


def merge_aliases_per_day(
    per_day_stats: dict[str, dict[str, dict]],
) -> dict[str, dict[str, dict]]:
    """
    Apply the SAME alias-merging logic used for the aggregate view
    (analyze_entity_sentiment.merge_aliases), but per-day, so
    "Donald Trump" on day 1 still merges with "Trump" on day 2 under
    one consistent entity key across the whole trend.
    """

    mapping = derive_alias_mapping(set(per_day_stats.keys()))

    merged_per_day: dict[str, dict[str, dict]] = defaultdict(dict)

    for entity_key, days in per_day_stats.items():

        target_key = mapping.get(entity_key, entity_key)

        for date, day_stats in days.items():

            if date not in merged_per_day[target_key]:
                merged_per_day[target_key][date] = {
                    "Positive": 0, "Neutral": 0, "Negative": 0,
                    "total": 0, "display_name": day_stats["display_name"],
                }

            for label in ("Positive", "Neutral", "Negative"):
                merged_per_day[target_key][date][label] += day_stats[label]
            merged_per_day[target_key][date]["total"] += day_stats["total"]

    return merged_per_day


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Show day-over-day sentiment trends by entity from accumulated live_results.jsonl data."
    )
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--entity", default=None, help="Show the trend for one specific entity (case-insensitive).")
    parser.add_argument("--list-entities", action="store_true", help="List entities with data across 2+ distinct days.")
    parser.add_argument("--min-mentions", type=int, default=2, help="Minimum total mentions to include an entity.")
    args = parser.parse_args()

    print(f"Loading {args.dataset_path}...")

    records = []
    with open(args.dataset_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Loaded {len(records)} records")

    per_day_stats = build_trend_stats(records)
    per_day_stats = merge_aliases_per_day(per_day_stats)

    multi_day_entities = {
        key: days for key, days in per_day_stats.items()
        if len(days) >= 2 and sum(d["total"] for d in days.values()) >= args.min_mentions
    }

    if args.list_entities:
        print()
        print(f"Entities with data across 2+ distinct days (>= {args.min_mentions} total mentions):")
        for key, days in sorted(multi_day_entities.items(), key=lambda kv: -sum(d["total"] for d in kv[1].values())):
            display_name = next(iter(days.values()))["display_name"]
            dates = sorted(days.keys())
            print(f"  {display_name:30} across {len(dates)} day(s): {', '.join(dates)}")
        return

    if not args.entity:
        parser.error("Provide --entity NAME or use --list-entities to see what's available.")

    target_key = args.entity.lower()

    if target_key not in per_day_stats:
        print(f"\nNo data found for '{args.entity}'.")
        return

    days = per_day_stats[target_key]
    display_name = next(iter(days.values()))["display_name"]

    print()
    print("=" * 70)
    print(f"SENTIMENT TREND: {display_name}")
    print("=" * 70)

    if len(days) < 2:
        print("(Only 1 day of data so far -- not enough to show a trend yet.)")

    header = f"{'DATE':12}{'MENTIONS':>10}{'POSITIVE':>10}{'NEUTRAL':>10}{'NEGATIVE':>10}"
    print(header)
    print("-" * len(header))

    for date in sorted(days.keys()):

        d = days[date]
        total = d["total"]
        pos_pct = d["Positive"] / total if total else 0.0
        neu_pct = d["Neutral"] / total if total else 0.0
        neg_pct = d["Negative"] / total if total else 0.0

        print(f"{date:12}{total:>10}{pos_pct:>9.0%} {neu_pct:>9.0%} {neg_pct:>9.0%}")


if __name__ == "__main__":
    main()
"""
sample_live_for_labeling.py
---------------------------

Draw a stratified sample from live_results.jsonl for hand-labeling, so
the model's accuracy on *current* news can be measured.

Writes two files:
  * a labeling CSV with a blank `gold_label` column (no model prediction,
    so it cannot bias your labels)
  * a key CSV with the model's predictions, joined back by
    evaluate_live_labels.py

Sampling: stratified by outlet x predicted label. Positive/Negative
predictions get larger quotas than Neutral (Neutral is ~60% of live
output and would otherwise swamp the sample). At most --max-per-sentence
targets are taken from one sentence.

Usage:
    python3 scripts/sample_live_for_labeling.py
    python3 scripts/sample_live_for_labeling.py --input live_results.jsonl --size 150
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

OUTLETS = {
    "feeds.npr.org": "NPR",
    "feeds.bbci.co.uk": "BBC",
    "www.thehindu.com": "The Hindu",
}
LABELS = ("Negative", "Neutral", "Positive")


def outlet_of(record: dict) -> str:
    host = urlparse(record["feed_url"]).netloc
    return OUTLETS.get(host, host)


def record_id(record: dict) -> str:
    raw = "|".join((record["article_url"], record["sentence"], record["target_text"]))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("error") or r.get("sentiment") not in LABELS:
                continue
            records.append(r)
    return records


def draw_sample(
    records: list[dict], size: int, max_per_sentence: int, rng: random.Random
) -> list[dict]:
    # De-duplicate, and cap targets per sentence.
    seen_ids: set[str] = set()
    per_sentence: dict[str, int] = defaultdict(int)
    shuffled = records[:]
    rng.shuffle(shuffled)
    pool = []
    for r in shuffled:
        rid = record_id(r)
        if rid in seen_ids or per_sentence[r["sentence"]] >= max_per_sentence:
            continue
        seen_ids.add(rid)
        per_sentence[r["sentence"]] += 1
        pool.append(r)

    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in pool:
        strata[(outlet_of(r), r["sentiment"])].append(r)

    outlets = sorted({k[0] for k in strata})
    # Neutral gets half the quota of Positive/Negative.
    weights = {"Negative": 2, "Positive": 2, "Neutral": 1}
    total_weight = sum(weights[lab] for lab in LABELS) * len(outlets)

    sample: list[dict] = []
    for outlet in outlets:
        for lab in LABELS:
            quota = round(size * weights[lab] / total_weight)
            sample.extend(strata[(outlet, lab)][:quota])

    # Top up (short strata) or trim (rounding) to exactly `size`.
    chosen = {record_id(r) for r in sample}
    if len(sample) < size:
        extras = [r for r in pool if record_id(r) not in chosen]
        sample.extend(extras[: size - len(sample)])
    rng.shuffle(sample)
    return sample[:size]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", type=Path, default=Path("data/labeling/live_results_sample.jsonl"))
    ap.add_argument("--size", type=int, default=150)
    ap.add_argument("--max-per-sentence", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-dir", type=Path, default=Path("data/labeling"))
    args = ap.parse_args()

    records = load_records(args.input)
    print(f"Usable records in {args.input}: {len(records)}")

    sample = draw_sample(records, args.size, args.max_per_sentence, random.Random(args.seed))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    label_path = args.out_dir / "live_labeling_set.csv"
    key_path = args.out_dir / "live_labeling_key.csv"

    with label_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["record_id", "outlet", "sentence", "target_text", "role",
                    "extraction_confidence", "gold_label"])
        for r in sample:
            w.writerow([record_id(r), outlet_of(r), r["sentence"], r["target_text"],
                        r["role"], r["extraction_confidence"], ""])

    with key_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["record_id", "predicted_label", "predicted_confidence"])
        for r in sample:
            w.writerow([record_id(r), r["sentiment"], r.get("sentiment_confidence")])

    print(f"Sampled {len(sample)} records")
    by = defaultdict(int)
    for r in sample:
        by[(outlet_of(r), r["sentiment"])] += 1
    for k in sorted(by):
        print(f"  {k[0]:10s} {k[1]:9s} {by[k]}")
    print(f"\nWrote {label_path}  (fill in the gold_label column)")
    print(f"Wrote {key_path}  (model predictions; do not look at it while labeling)")
    print("Allowed gold_label values: negative, neutral, positive")
    print("Label the sentiment toward target_text in that sentence, as the article presents it.")


if __name__ == "__main__":
    main()

"""
semantic/evaluate_sentiment_accuracy.py
------------------------------------------

Measure REAL sentiment-prediction accuracy: for every dataset
record where we already have a TRUSTWORTHY gold sentiment label
(label_source in {"gold_exact", "gold_embedded_or_resolved"} --
see dataset_schema_v1.md), run the trained model's prediction for
the same (sentence, target) pair and compare against the gold
label.

This is the first rigorous accuracy check of the actual sentiment
step -- previous checks were eyeballing a handful of predictions,
this compares against thousands of real, pre-existing gold labels.

Usage:
    PYTHONPATH=src python src/semantic/evaluate_sentiment_accuracy.py \\
        dataset_full.jsonl --limit 500

    --limit N restricts evaluation to the first N gold-matched
    records. Start small (e.g. 200-500) before running against
    every gold-matched record, since each one requires a real model
    inference call.

    --dump-errors <path> writes every misclassified example (gold
    label, predicted label, sentence, target) to a JSONL file for
    manual inspection.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from transformer_predictor import TransformerPredictor
from semantic.predict_pipeline import locate_target_text


LABEL_TO_NUMERIC = {"Negative": 2.0, "Neutral": 4.0, "Positive": 6.0}
NUMERIC_TO_LABEL = {v: k for k, v in LABEL_TO_NUMERIC.items()}

TRUSTWORTHY_SOURCES = {"gold_exact", "gold_embedded_or_resolved"}


def load_gold_matched_records(path: Path, limit: int | None) -> list[dict]:

    records = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if (
                r.get("label_source") in TRUSTWORTHY_SOURCES
                and r.get("sentiment_label") is not None
            ):
                records.append(r)

    if limit is not None:
        records = records[:limit]

    return records


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Evaluate sentiment prediction accuracy against existing gold labels."
    )
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dump-errors", type=Path, default=None)
    args = parser.parse_args()

    print(f"Loading gold-matched records from {args.dataset_path}...")
    records = load_gold_matched_records(args.dataset_path, args.limit)
    print(f"Found {len(records)} records with trustworthy gold sentiment labels")

    if not records:
        print("Nothing to evaluate -- exiting.")
        return

    print("Loading trained sentiment model...")
    predictor = TransformerPredictor()

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_source: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    errors: list[dict] = []
    skipped = 0

    total = len(records)

    for i, r in enumerate(records, start=1):

        if i == 1 or i % 100 == 0 or i == total:
            print(f"  ...{i}/{total} processed")

        sentence = r["sentence"]
        target = r["target_text"]
        gold_numeric = r["sentiment_label"]
        gold_label = NUMERIC_TO_LABEL.get(gold_numeric)

        if gold_label is None:
            skipped += 1
            continue

        located = locate_target_text(sentence, target)

        if located is None:
            skipped += 1
            continue

        try:
            prediction = predictor.predict(sentence, located)
        except Exception:
            skipped += 1
            continue

        pred_label = prediction["prediction"]

        confusion[gold_label][pred_label] += 1

        source = r.get("label_source", "unknown")
        by_source[source]["total"] += 1

        if pred_label == gold_label:
            by_source[source]["correct"] += 1
        else:
            errors.append(
                {
                    "sentence": sentence,
                    "target": target,
                    "gold": gold_label,
                    "predicted": pred_label,
                    "confidence": prediction["confidence"],
                }
            )

    total_evaluated = sum(v["total"] for v in by_source.values())
    total_correct = sum(v["correct"] for v in by_source.values())

    print()
    print("=" * 70)
    print("SENTIMENT ACCURACY EVALUATION")
    print("=" * 70)
    print(f"Records with gold labels: {total}")
    print(f"Skipped (target not locatable / prediction failed): {skipped}")
    print(f"Evaluated: {total_evaluated}")
    print()

    overall_acc = total_correct / total_evaluated if total_evaluated else 0.0
    print(f"Overall accuracy: {total_correct}/{total_evaluated} = {overall_acc:.1%}")
    print()

    print("By label source:")
    for source, v in by_source.items():
        acc = v["correct"] / v["total"] if v["total"] else 0.0
        print(f"  {source:28} {v['correct']:5}/{v['total']:<5} = {acc:.1%}")

    labels = ["Negative", "Neutral", "Positive"]

    print()
    print("Confusion matrix (rows=gold, cols=predicted):")
    header = " " * 12 + "".join(f"{l:>10}" for l in labels)
    print(header)
    for gold_l in labels:
        row = f"{gold_l:12}" + "".join(
            f"{confusion[gold_l][pred_l]:>10}" for pred_l in labels
        )
        print(row)

    print()
    print("Per-class precision/recall:")
    for label in labels:
        tp = confusion[label][label]
        fn = sum(confusion[label][p] for p in labels if p != label)
        fp = sum(confusion[g][label] for g in labels if g != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        print(f"  {label:10} precision={precision:.1%}  recall={recall:.1%}")

    if args.dump_errors is not None and errors:
        with open(args.dump_errors, "w", encoding="utf-8") as f:
            for e in errors:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print()
        print(f"Wrote {len(errors)} misclassified examples to {args.dump_errors}")

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

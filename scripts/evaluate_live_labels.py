"""
evaluate_live_labels.py
-----------------------

Compare hand labels (live_labeling_set.csv, gold_label filled in) with
the model's predictions (live_labeling_key.csv).

Prints overall accuracy, per-class precision/recall, a confusion matrix,
and breakdowns by outlet and by extraction confidence. Also writes the
result to outputs/live_eval/.

Usage:
    python3 scripts/evaluate_live_labels.py
    python3 scripts/evaluate_live_labels.py --labels data/labeling/live_labeling_set.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ("negative", "neutral", "positive")
ALIASES = {"neg": "negative", "neu": "neutral", "pos": "positive"}


def norm(label: str) -> str:
    label = label.strip().lower()
    return ALIASES.get(label, label)


def load(labels_path: Path, key_path: Path) -> tuple[list[dict], int]:
    with key_path.open(encoding="utf-8", newline="") as f:
        key = {r["record_id"]: r for r in csv.DictReader(f)}

    rows, unlabeled = [], 0
    with labels_path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            gold = norm(r.get("gold_label", ""))
            if not gold:
                unlabeled += 1
                continue
            if gold not in CLASSES:
                raise SystemExit(
                    f"Invalid gold_label {r['gold_label']!r} for record {r['record_id']} "
                    f"(allowed: {', '.join(CLASSES)})"
                )
            if r["record_id"] not in key:
                raise SystemExit(f"record_id {r['record_id']} not found in {key_path}")
            r["gold"] = gold
            r["pred"] = norm(key[r["record_id"]]["predicted_label"])
            rows.append(r)
    return rows, unlabeled


def accuracy(rows: list[dict]) -> float:
    return sum(r["gold"] == r["pred"] for r in rows) / len(rows) if rows else float("nan")


def per_class(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for c in CLASSES:
        tp = sum(r["gold"] == c and r["pred"] == c for r in rows)
        pred_n = sum(r["pred"] == c for r in rows)
        gold_n = sum(r["gold"] == c for r in rows)
        out[c] = {
            "precision": tp / pred_n if pred_n else None,
            "recall": tp / gold_n if gold_n else None,
            "support": gold_n,
        }
    return out


def fmt(x: float | None) -> str:
    return "  n/a" if x is None else f"{x:5.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", type=Path, default=Path("data/labeling/live_labeling_set.csv"))
    ap.add_argument("--key", type=Path, default=Path("data/labeling/live_labeling_key.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/live_eval"))
    args = ap.parse_args()

    rows, unlabeled = load(args.labels, args.key)
    if not rows:
        raise SystemExit("No labeled rows found: fill in the gold_label column first.")

    n = len(rows)
    print(f"Labeled rows: {n}   (unlabeled, skipped: {unlabeled})")
    if n < 100:
        print("Warning: fewer than 100 labels; treat the numbers as rough.")
    acc = accuracy(rows)
    print(f"\nOverall accuracy: {acc:.3f}  ({sum(r['gold'] == r['pred'] for r in rows)}/{n})")

    pc = per_class(rows)
    print("\nPer class        precision recall support")
    for c in CLASSES:
        print(f"  {c:12s}   {fmt(pc[c]['precision'])}    {fmt(pc[c]['recall'])}  {pc[c]['support']:5d}")

    conf = Counter((r["gold"], r["pred"]) for r in rows)
    print("\nConfusion matrix (rows = your label, columns = model)")
    print("              " + "".join(f"{c:>10s}" for c in CLASSES))
    for g in CLASSES:
        print(f"  {g:10s}  " + "".join(f"{conf[(g, p)]:10d}" for p in CLASSES))

    def breakdown(field: str) -> dict[str, dict]:
        groups = defaultdict(list)
        for r in rows:
            groups[r[field]].append(r)
        return {k: {"n": len(v), "accuracy": accuracy(v)} for k, v in sorted(groups.items())}

    by_outlet = breakdown("outlet")
    by_conf = breakdown("extraction_confidence")
    for title, data in (("outlet", by_outlet), ("extraction confidence", by_conf)):
        print(f"\nBy {title}")
        for k, v in data.items():
            print(f"  {k:12s} n={v['n']:4d}  accuracy={v['accuracy']:.3f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "labeled": n, "unlabeled": unlabeled, "accuracy": acc, "per_class": pc,
        "confusion": {f"{g}->{p}": conf[(g, p)] for g in CLASSES for p in CLASSES},
        "by_outlet": by_outlet, "by_extraction_confidence": by_conf,
    }
    (args.out_dir / "live_eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    with (args.out_dir / "live_eval_errors.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["record_id", "outlet", "sentence", "target_text", "gold", "predicted"])
        for r in rows:
            if r["gold"] != r["pred"]:
                w.writerow([r["record_id"], r["outlet"], r["sentence"], r["target_text"],
                            r["gold"], r["pred"]])
    print(f"\nWrote {args.out_dir}/live_eval.json and live_eval_errors.csv")


if __name__ == "__main__":
    main()

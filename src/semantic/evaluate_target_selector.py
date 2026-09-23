"""
semantic/evaluate_target_selector.py
------------------------------------

Pipeline step 8: evaluate whether:

    SemanticExtractor
        ↓
    SemanticGraphBuilder
        ↓
    TargetSelector
        ↓
    EntityResolver

can recover REAL gold targets from:

    data/processed/train_target.csv

This evaluator is a RECALL evaluation.

For every gold (sentence, target) pair, we report the best
matching tier:

    1. EXACT
        Gold target exactly matches a selected target.

    2. RESOLVED_OR_EMBEDDED
        Gold target matches:
          - a TargetSelection.embedded_entities entry
          - an EntityResolver canonical name
          - an EntityResolver mention surface form
          - an unresolved mention surface form

    3. SUBSTRING
        Gold target and candidate partially overlap.

    4. NOT_FOUND
        No matching candidate exists.

Important performance change:
    spaCy inference uses nlp.pipe(..., batch_size=N).

The existing SemanticExtractor.extract() logic is preserved by
feeding each already-parsed spaCy Doc back through extract().
This means batching changes the spaCy inference stage without
silently bypassing extractor-specific post-processing.

Outputs:

    predictions CSV:
        one row per (sentence, gold_target, model)

    model comparison CSV:
        one row per model with aggregate metrics

Examples:

    Single model:

        PYTHONPATH=src python src/semantic/evaluate_target_selector.py \
            data/processed/train_target.csv \
            --model en_core_web_sm

    Two-model comparison:

        PYTHONPATH=src python src/semantic/evaluate_target_selector.py \
            data/processed/train_target.csv \
            --compare-models en_core_web_sm en_core_web_trf

    First 300 sentences:

        PYTHONPATH=src python src/semantic/evaluate_target_selector.py \
            data/processed/train_target.csv \
            --compare-models en_core_web_sm en_core_web_trf \
            --limit 300 \
            --batch-size 16

    Custom output directory:

        PYTHONPATH=src python src/semantic/evaluate_target_selector.py \
            data/processed/train_target.csv \
            --compare-models en_core_web_sm en_core_web_trf \
            --output-dir outputs/target_benchmark

    Dump every NOT_FOUND target:

        PYTHONPATH=src python src/semantic/evaluate_target_selector.py \
            data/processed/train_target.csv \
            --compare-models en_core_web_sm en_core_web_trf \
            --dump-not-found outputs/target_benchmark/not_found.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from semantic.entity_resolver import EntityResolver
from semantic.semantic_extractor import (
    SemanticExtractor,
    SemanticRepresentation,
)
from semantic.semantic_graph import SemanticGraphBuilder
from semantic.target_selector import TargetSelector


# ============================================================
# Matching helpers
# ============================================================


def normalize(text: str) -> str:
    """Normalize text for case-insensitive matching."""

    return " ".join(text.strip().lower().split())


def load_gold_data(csv_path: Path) -> dict[str, list[str]]:
    """
    Read the gold CSV and group targets by sentence.

    Preserves first-seen sentence order.
    """

    gold: dict[str, list[str]] = defaultdict(list)

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        required_columns = {"sentence", "target"}
        fieldnames = set(reader.fieldnames or [])

        missing = required_columns - fieldnames
        if missing:
            raise ValueError(
                f"Gold CSV is missing required columns: "
                f"{sorted(missing)}"
            )

        for row in reader:
            sentence = (row.get("sentence") or "").strip()
            target = (row.get("target") or "").strip()

            if sentence and target:
                gold[sentence].append(target)

    return gold


# ============================================================
# spaCy batching
# ============================================================


def iter_representations(
    sentences: list[str],
    extractor: SemanticExtractor,
    batch_size: int,
) -> Iterable[tuple[str, SemanticRepresentation]]:
    """
    Parse sentences through spaCy using nlp.pipe() batching, while
    preserving the existing SemanticExtractor.extract() logic.

    Why this structure?

    The current extractor does more than just:
        doc = self.nlp(sentence)

    Calling the private _extract_* methods directly here could
    accidentally bypass future extractor-specific preprocessing
    or contextual NER recovery.

    Instead:

        1. nlp.pipe() performs batched spaCy inference.
        2. The resulting Doc is temporarily supplied to
           SemanticExtractor.extract().
        3. The rest of the extractor pipeline remains unchanged.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    original_nlp = extractor.nlp

    docs = original_nlp.pipe(
        sentences,
        batch_size=batch_size,
    )

    for sentence, doc in zip(sentences, docs):

        def prepared_nlp(
            _text: str,
            _doc=doc,
        ):
            return _doc

        extractor.nlp = prepared_nlp

        try:
            representation = extractor.extract(sentence)
        finally:
            extractor.nlp = original_nlp

        yield sentence, representation


# ============================================================
# Candidate pool
# ============================================================


def build_candidate_pool(
    graph,
    targets,
    resolution,
) -> tuple[
    dict[str, str],
    dict[str, str],
]:
    """
    Build matching pools.

    exact_pool:
        normalized selected-target text -> original target text

    embedded_pool:
        normalized embedded/resolved mention -> original surface text
    """

    exact_pool: dict[str, str] = {}
    embedded_pool: dict[str, str] = {}

    # --------------------------------------------------------
    # TargetSelector outputs
    # --------------------------------------------------------

    for target in targets:

        if target.target:
            exact_pool[
                normalize(target.target)
            ] = target.target

        for entity in target.embedded_entities:
            embedded_pool.setdefault(
                normalize(entity),
                entity,
            )

    # --------------------------------------------------------
    # EntityResolver outputs
    # --------------------------------------------------------

    for entity in resolution.resolved_entities:

        embedded_pool.setdefault(
            normalize(entity.canonical_name),
            entity.canonical_name,
        )

        for mention in entity.mentions:
            embedded_pool.setdefault(
                normalize(mention.text),
                mention.text,
            )

    # --------------------------------------------------------
    # Unresolved mentions
    # --------------------------------------------------------

    for mention in resolution.unresolved_mentions:
        embedded_pool.setdefault(
            normalize(mention.text),
            mention.text,
        )

    return exact_pool, embedded_pool


# ============================================================
# Match detail
# ============================================================


def match_target(
    gold_target: str,
    exact_pool: dict[str, str],
    embedded_pool: dict[str, str],
) -> tuple[str, str | None, str | None]:
    """
    Return:

        (
            match_tier,
            matched_text,
            matched_source,
        )

    matched_source:
        "selected_target"
        "embedded_or_resolved"
        "substring"
        None
    """

    gold_norm = normalize(gold_target)

    # --------------------------------------------------------
    # 1. EXACT
    # --------------------------------------------------------

    if gold_norm in exact_pool:

        return (
            "EXACT",
            exact_pool[gold_norm],
            "selected_target",
        )

    # --------------------------------------------------------
    # 2. RESOLVED_OR_EMBEDDED
    # --------------------------------------------------------

    if gold_norm in embedded_pool:

        return (
            "RESOLVED_OR_EMBEDDED",
            embedded_pool[gold_norm],
            "embedded_or_resolved",
        )

    # --------------------------------------------------------
    # 3. SUBSTRING
    # --------------------------------------------------------

    all_candidates: list[
        tuple[str, str]
    ] = []

    for candidate_norm, original in exact_pool.items():
        all_candidates.append(
            (candidate_norm, original)
        )

    for candidate_norm, original in embedded_pool.items():
        all_candidates.append(
            (candidate_norm, original)
        )

    for candidate_norm, original in all_candidates:

        if not candidate_norm:
            continue

        if (
            gold_norm in candidate_norm
            or candidate_norm in gold_norm
        ):

            return (
                "SUBSTRING",
                original,
                "substring",
            )

    # --------------------------------------------------------
    # 4. NOT_FOUND
    # --------------------------------------------------------

    return (
        "NOT_FOUND",
        None,
        None,
    )


# ============================================================
# JSON helpers
# ============================================================


def compact_json(value) -> str:
    """Serialize a value compactly for CSV storage."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def target_texts(targets) -> list[str]:
    """Return selected target texts."""

    return [
        target.target
        for target in targets
        if target.target
    ]


def selected_target_details(targets) -> list[dict]:
    """
    Return compact TargetSelector output suitable for CSV.
    """

    results = []

    for target in targets:

        results.append(
            {
                "target": target.target,
                "role": target.role,
                "event": target.event,
                "confidence": target.confidence,
                "embedded_entities": list(
                    target.embedded_entities
                ),
            }
        )

    return results


def resolution_details(resolution) -> tuple[
    list[str],
    list[str],
    list[dict],
]:
    """
    Return:

        canonical_names
        unresolved_mentions
        resolved_entity_details
    """

    canonical_names = []
    unresolved_mentions = []
    resolved_details = []

    for entity in resolution.resolved_entities:

        canonical_names.append(
            entity.canonical_name
        )

        resolved_details.append(
            {
                "canonical_id": entity.canonical_id,
                "canonical_name": entity.canonical_name,
                "entity_type": entity.entity_type,
                "mentions": [
                    mention.text
                    for mention in entity.mentions
                ],
            }
        )

    for mention in resolution.unresolved_mentions:
        unresolved_mentions.append(
            mention.text
        )

    return (
        canonical_names,
        unresolved_mentions,
        resolved_details,
    )


# ============================================================
# Evaluate one model
# ============================================================


def evaluate_model(
    model_name: str,
    sentences: list[str],
    gold: dict[str, list[str]],
    batch_size: int,
) -> tuple[
    dict[str, int],
    list[dict[str, object]],
    list[tuple[str, str]],
    float,
]:
    """
    Evaluate one spaCy model.

    Returns:

        tier_counts
        prediction_rows
        not_found_rows
        elapsed_seconds
    """

    print()
    print("=" * 70)
    print(f"EVALUATING MODEL: {model_name}")
    print("=" * 70)

    start_time = time.perf_counter()

    extractor = SemanticExtractor(
        model_name=model_name
    )

    graph_builder = SemanticGraphBuilder()
    selector = TargetSelector()
    resolver = EntityResolver()

    tier_counts: dict[str, int] = defaultdict(int)

    prediction_rows: list[dict[str, object]] = []
    not_found_rows: list[tuple[str, str]] = []

    total_sentences = len(sentences)

    for sentence_index, (
        sentence,
        representation,
    ) in enumerate(
        iter_representations(
            sentences,
            extractor,
            batch_size,
        ),
        start=1,
    ):

        if (
            sentence_index == 1
            or sentence_index % 50 == 0
            or sentence_index == total_sentences
        ):
            print(
                f"  ...{sentence_index}/"
                f"{total_sentences} sentences processed"
            )

        # ----------------------------------------------------
        # Semantic pipeline
        # ----------------------------------------------------

        graph = graph_builder.build(
            representation
        )

        targets = selector.select_all(
            graph
        )

        resolution = resolver.resolve(
            [graph],
            [targets],
        )

        (
            exact_pool,
            embedded_pool,
        ) = build_candidate_pool(
            graph,
            targets,
            resolution,
        )

        (
            canonical_names,
            unresolved_mentions,
            resolved_entity_details,
        ) = resolution_details(
            resolution
        )

        selected_details = (
            selected_target_details(targets)
        )

        selected_target_texts = target_texts(
            targets
        )

        # ----------------------------------------------------
        # Gold targets
        # ----------------------------------------------------

        for gold_target in gold[sentence]:

            (
                tier,
                matched_text,
                matched_source,
            ) = match_target(
                gold_target,
                exact_pool,
                embedded_pool,
            )

            tier_counts[tier] += 1

            if tier == "NOT_FOUND":
                not_found_rows.append(
                    (
                        sentence,
                        gold_target,
                    )
                )

            prediction_rows.append(
                {
                    "model": model_name,
                    "sentence_index": sentence_index - 1,
                    "sentence": sentence,
                    "gold_target": gold_target,
                    "match_tier": tier,
                    "matched_text": matched_text or "",
                    "matched_source": matched_source or "",
                    "selected_targets": compact_json(
                        selected_target_texts
                    ),
                    "selected_target_details": compact_json(
                        selected_details
                    ),
                    "resolved_entities": compact_json(
                        canonical_names
                    ),
                    "resolved_entity_details": compact_json(
                        resolved_entity_details
                    ),
                    "unresolved_mentions": compact_json(
                        unresolved_mentions
                    ),
                }
            )

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    return (
        tier_counts,
        prediction_rows,
        not_found_rows,
        elapsed_seconds,
    )


# ============================================================
# Write prediction CSV
# ============================================================


def write_predictions_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write per-gold-target prediction rows."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "model",
        "sentence_index",
        "sentence",
        "gold_target",
        "match_tier",
        "matched_text",
        "matched_source",
        "selected_targets",
        "selected_target_details",
        "resolved_entities",
        "resolved_entity_details",
        "unresolved_mentions",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Write model comparison CSV
# ============================================================


def write_comparison_csv(
    path: Path,
    summaries: list[dict[str, object]],
) -> None:
    """Write aggregate comparison metrics."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "model",
        "sentences_evaluated",
        "gold_targets",
        "found_targets",
        "overall_recall",
        "exact_count",
        "exact_pct",
        "resolved_or_embedded_count",
        "resolved_or_embedded_pct",
        "substring_count",
        "substring_pct",
        "not_found_count",
        "not_found_pct",
        "elapsed_seconds",
        "targets_per_second",
        "recall_delta_vs_first_model",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(summaries)


# ============================================================
# Write NOT_FOUND TSV
# ============================================================


def write_not_found_tsv(
    path: Path,
    rows: list[tuple[str, str]],
) -> None:
    """Write NOT_FOUND rows as target + sentence."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        for sentence, target in rows:
            f.write(
                f"{target}\t{sentence}\n"
            )


# ============================================================
# Formatting
# ============================================================


def percentage(
    count: int,
    total: int,
) -> float:
    """Return count/total as percentage."""

    if total == 0:
        return 0.0

    return count / total * 100.0


def build_summary(
    model_name: str,
    sentences_evaluated: int,
    total_gold_targets: int,
    tier_counts: dict[str, int],
    elapsed_seconds: float,
    first_model_recall: float | None,
) -> dict[str, object]:
    """Create one comparison-summary row."""

    not_found = tier_counts.get(
        "NOT_FOUND",
        0,
    )

    found = (
        total_gold_targets - not_found
    )

    recall = (
        found / total_gold_targets
        if total_gold_targets
        else 0.0
    )

    targets_per_second = (
        total_gold_targets / elapsed_seconds
        if elapsed_seconds > 0
        else 0.0
    )

    if first_model_recall is None:
        recall_delta = ""
    else:
        recall_delta = (
            recall - first_model_recall
        )

    return {
        "model": model_name,
        "sentences_evaluated": sentences_evaluated,
        "gold_targets": total_gold_targets,
        "found_targets": found,
        "overall_recall": round(
            recall,
            6,
        ),
        "exact_count": tier_counts.get(
            "EXACT",
            0,
        ),
        "exact_pct": round(
            percentage(
                tier_counts.get(
                    "EXACT",
                    0,
                ),
                total_gold_targets,
            ),
            3,
        ),
        "resolved_or_embedded_count": tier_counts.get(
            "RESOLVED_OR_EMBEDDED",
            0,
        ),
        "resolved_or_embedded_pct": round(
            percentage(
                tier_counts.get(
                    "RESOLVED_OR_EMBEDDED",
                    0,
                ),
                total_gold_targets,
            ),
            3,
        ),
        "substring_count": tier_counts.get(
            "SUBSTRING",
            0,
        ),
        "substring_pct": round(
            percentage(
                tier_counts.get(
                    "SUBSTRING",
                    0,
                ),
                total_gold_targets,
            ),
            3,
        ),
        "not_found_count": not_found,
        "not_found_pct": round(
            percentage(
                not_found,
                total_gold_targets,
            ),
            3,
        ),
        "elapsed_seconds": round(
            elapsed_seconds,
            3,
        ),
        "targets_per_second": round(
            targets_per_second,
            3,
        ),
        "recall_delta_vs_first_model": (
            round(
                recall_delta,
                6,
            )
            if recall_delta != ""
            else ""
        ),
    }


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate TargetSelector + EntityResolver "
            "with batched spaCy inference."
        )
    )

    parser.add_argument(
        "csv_path",
        type=Path,
        help="Gold target CSV.",
    )

    model_group = parser.add_mutually_exclusive_group()

    model_group.add_argument(
        "--model",
        default="en_core_web_sm",
        help=(
            "Single spaCy model to evaluate. "
            "Default: en_core_web_sm"
        ),
    )

    model_group.add_argument(
        "--compare-models",
        nargs="+",
        help=(
            "Evaluate multiple spaCy models in the same run. "
            "Example: --compare-models "
            "en_core_web_sm en_core_web_trf"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Limit to the first N UNIQUE sentences."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help=(
            "spaCy nlp.pipe() batch size. "
            "Default: 16"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/target_selector_evaluation"
        ),
        help=(
            "Directory for CSV/TSV outputs. "
            "Default: outputs/target_selector_evaluation"
        ),
    )

    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=None,
        help=(
            "Override per-target predictions CSV path."
        ),
    )

    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=None,
        help=(
            "Override model-comparison CSV path."
        ),
    )

    parser.add_argument(
        "--dump-not-found",
        type=Path,
        default=None,
        help=(
            "Write all NOT_FOUND cases to this TSV path."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================


def main() -> None:

    args = parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError(
            "--limit must be greater than zero"
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be greater than zero"
        )

    csv_path: Path = args.csv_path

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Gold CSV does not exist: {csv_path}"
        )

    print(
        f"Loading gold data from {csv_path}..."
    )

    gold = load_gold_data(
        csv_path
    )

    sentences = list(
        gold.keys()
    )

    if args.limit is not None:
        sentences = sentences[: args.limit]

        print(
            f"Limiting to first "
            f"{args.limit} unique sentences."
        )

    total_gold_targets = sum(
        len(gold[sentence])
        for sentence in sentences
    )

    print(
        f"Sentences to evaluate: "
        f"{len(sentences)}"
    )

    print(
        f"Total gold targets: "
        f"{total_gold_targets}"
    )

    print(
        f"Batch size: "
        f"{args.batch_size}"
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    if args.compare_models:
        models = list(
            dict.fromkeys(
                args.compare_models
            )
        )
    else:
        models = [args.model]

    print()
    print(
        "Models:"
    )

    for model in models:
        print(
            f"  - {model}"
        )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    output_dir: Path = args.output_dir
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = (
        args.predictions_csv
        or output_dir
        / "target_selector_predictions.csv"
    )

    comparison_path = (
        args.comparison_csv
        or output_dir
        / "target_selector_model_comparison.csv"
    )

    # --------------------------------------------------------
    # Evaluate each model
    # --------------------------------------------------------

    all_prediction_rows: list[
        dict[str, object]
    ] = []

    summaries: list[
        dict[str, object]
    ] = []

    not_found_by_model: dict[
        str,
        list[tuple[str, str]],
    ] = {}

    first_model_recall: float | None = None

    for model_name in models:

        (
            tier_counts,
            prediction_rows,
            not_found_rows,
            elapsed_seconds,
        ) = evaluate_model(
            model_name=model_name,
            sentences=sentences,
            gold=gold,
            batch_size=args.batch_size,
        )

        all_prediction_rows.extend(
            prediction_rows
        )

        not_found_by_model[
            model_name
        ] = not_found_rows

        summary = build_summary(
            model_name=model_name,
            sentences_evaluated=len(
                sentences
            ),
            total_gold_targets=(
                total_gold_targets
            ),
            tier_counts=tier_counts,
            elapsed_seconds=(
                elapsed_seconds
            ),
            first_model_recall=(
                first_model_recall
            ),
        )

        summaries.append(
            summary
        )

        if first_model_recall is None:
            first_model_recall = (
                total_gold_targets
                - tier_counts.get(
                    "NOT_FOUND",
                    0,
                )
            ) / total_gold_targets if (
                total_gold_targets
            ) else 0.0

    # --------------------------------------------------------
    # Write outputs
    # --------------------------------------------------------

    write_predictions_csv(
        predictions_path,
        all_prediction_rows,
    )

    write_comparison_csv(
        comparison_path,
        summaries,
    )

    # --------------------------------------------------------
    # Optional NOT_FOUND dump
    # --------------------------------------------------------

    if args.dump_not_found is not None:

        combined_not_found = []

        for model_name in models:

            model_rows = (
                not_found_by_model[
                    model_name
                ]
            )

            for sentence, target in model_rows:
                combined_not_found.append(
                    (
                        f"{model_name}\t{target}",
                        sentence,
                    )
                )

        write_not_found_tsv(
            args.dump_not_found,
            combined_not_found,
        )

    # --------------------------------------------------------
    # Print final comparison
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("MODEL COMPARISON")
    print("=" * 78)

    header = (
        f"{'MODEL':28}"
        f"{'RECALL':>10}"
        f"{'EXACT':>10}"
        f"{'RES/EMB':>10}"
        f"{'SUBSTR':>10}"
        f"{'NOT_FOUND':>12}"
    )

    print(header)
    print("-" * len(header))

    for summary in summaries:

        model = str(
            summary["model"]
        )

        recall = float(
            summary["overall_recall"]
        )

        exact_pct = float(
            summary["exact_pct"]
        )

        resolved_pct = float(
            summary[
                "resolved_or_embedded_pct"
            ]
        )

        substring_pct = float(
            summary["substring_pct"]
        )

        not_found_pct = float(
            summary["not_found_pct"]
        )

        print(
            f"{model:28}"
            f"{recall:9.1%}"
            f"{exact_pct:9.1f}%"
            f"{resolved_pct:9.1f}%"
            f"{substring_pct:9.1f}%"
            f"{not_found_pct:11.1f}%"
        )

    print()
    print(
        f"Predictions CSV: "
        f"{predictions_path}"
    )

    print(
        f"Comparison CSV: "
        f"{comparison_path}"
    )

    if args.dump_not_found is not None:
        print(
            f"NOT_FOUND TSV: "
            f"{args.dump_not_found}"
        )

    print()
    print("=" * 78)
    print("EVALUATION COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nEvaluation interrupted."
        )
        sys.exit(130)
    except Exception as exc:
        print(
            f"\nERROR: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
"""
semantic/build_dataset.py
--------------------------

The unified target-extraction + sentiment pipeline. For each raw
sentence, runs the full semantic pipeline (extractor -> graph ->
target_selector -> entity_resolver) to find candidate targets, then
(optionally, via --predict) runs the trained sentiment model on
each one too -- giving one record per target with BOTH the gold
label (when a trustworthy match exists in --gold data) AND a live
model prediction (always, when --predict is used), plus whether
they agree.

This supersedes running build_dataset.py (gold-matching only) and
predict_pipeline.py (live prediction only) as two separate,
disconnected outputs -- one script, one combined record per target.
predict_pipeline.py still exists for quick single-sentence/ad-hoc
checks; this is the batch/corpus-scale tool.

See dataset_schema_v1.md for the base record format and rationale.

Usage:
    # Gold-matching only (original behavior, no --predict)
    PYTHONPATH=src python src/semantic/build_dataset.py <input.txt> <output.jsonl> --gold <gold.csv>

    # Gold-matching AND live model predictions, best model
    PYTHONPATH=src python src/semantic/build_dataset.py <input.txt> <output.jsonl> \\
        --gold <gold.csv> --model en_core_web_trf --batch-size 16 --predict

    If no arguments at all are given, runs on a small built-in
    sample sentence set and writes to dataset_sample.jsonl.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from semantic.semantic_extractor import SemanticExtractor, SemanticRepresentation
from semantic.semantic_graph import SemanticGraphBuilder
from semantic.target_selector import TargetSelector
from semantic.entity_resolver import EntityResolver
from semantic.predict_pipeline import locate_target_text


LABEL_TO_NUMERIC = {"Negative": 2.0, "Neutral": 4.0, "Positive": 6.0}
NUMERIC_TO_LABEL = {v: k for k, v in LABEL_TO_NUMERIC.items()}


def iter_representations(
    sentences: list[str],
    extractor: SemanticExtractor,
    batch_size: int,
) -> Iterable[tuple[str, SemanticRepresentation]]:
    """
    Parse sentences through spaCy using nlp.pipe() batching, while
    preserving SemanticExtractor.extract()'s existing logic.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    original_nlp = extractor.nlp

    docs = original_nlp.pipe(sentences, batch_size=batch_size)

    for sentence, doc in zip(sentences, docs):

        def prepared_nlp(_text: str, _doc=doc):
            return _doc

        extractor.nlp = prepared_nlp

        try:
            representation = extractor.extract(sentence)
        finally:
            extractor.nlp = original_nlp

        yield sentence, representation


SAMPLE_SENTENCES = [
    "Meta announced that it will invest billions of dollars in artificial intelligence.",
    "Tesla shares rose after the company reported stronger sales.",
    "Microsoft acquired Activision Blizzard.",
    "Google announced layoffs and Meta announced hiring.",
    "The European Central Bank raised rates. It cited persistent inflation pressures.",
]


def normalize(text: str) -> str:
    return text.strip().lower()


def load_gold_data(csv_path: Path) -> dict[str, list[tuple[str, float]]]:
    """
    Read a gold CSV (columns: sentence, target, polarity -- extra
    columns like target_from/target_to are ignored) and group
    (target, polarity) pairs by sentence.
    """

    gold: dict[str, list[tuple[str, float]]] = defaultdict(list)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence = row["sentence"].strip()
            target = row["target"].strip()
            polarity_raw = row.get("polarity", "").strip()
            if not sentence or not target:
                continue
            try:
                polarity = float(polarity_raw)
            except ValueError:
                polarity = None
            gold[sentence].append((target, polarity))

    return gold


def match_gold_label(
    target_text: str,
    embedded_entities: list[str],
    resolved_mentions: list[str],
    gold_targets: list[tuple[str, float]],
) -> tuple[str, float | None]:
    """
    Try to find a trustworthy gold label for this specific
    candidate target. See dataset_schema_v1.md for tier meanings.
    """

    target_norm = normalize(target_text)
    embedded_norm = {normalize(e) for e in embedded_entities}
    resolved_norm = {normalize(m) for m in resolved_mentions}

    for gold_target, polarity in gold_targets:
        gold_norm = normalize(gold_target)
        if gold_norm == target_norm:
            return "gold_exact", polarity

    for gold_target, polarity in gold_targets:
        gold_norm = normalize(gold_target)
        if gold_norm in embedded_norm or gold_norm in resolved_norm:
            return "gold_embedded_or_resolved", polarity

    return "needs_annotation", None


def build_records(
    sentence_id: str,
    sentence: str,
    representation: SemanticRepresentation,
    graph_builder: SemanticGraphBuilder,
    selector: TargetSelector,
    resolver: EntityResolver,
    gold_targets: list[tuple[str, float]] | None = None,
    predictor=None,
) -> list[dict]:
    """
    Run one sentence's ALREADY-EXTRACTED representation through the
    rest of the pipeline and return its dataset records.

    If `predictor` (a TransformerPredictor) is given, each record
    also gets a LIVE model prediction -- predicted_sentiment_label,
    predicted_sentiment_numeric, predicted_confidence -- and, when a
    trustworthy gold label also exists, sentiment_agreement (True/
    False). When no gold label exists, sentiment_agreement is None
    (nothing to compare against) rather than False, so "we don't
    know" is never confused with "the model was wrong."
    """

    graph = graph_builder.build(representation)
    targets = selector.select_all(graph)

    if not targets:
        return []

    resolution = resolver.resolve([graph], [targets])

    resolved_mentions_by_target: dict[str, list[str]] = {}
    resolved_type_by_target: dict[str, str | None] = {}

    for entity in resolution.resolved_entities:
        key = entity.canonical_name.strip().lower()
        resolved_mentions_by_target[key] = [m.text for m in entity.mentions]
        resolved_type_by_target[key] = entity.entity_type

    records: list[dict] = []
    seen_targets: set[str] = set()

    for t in targets:

        if t.target is None or t.confidence == "none":
            continue

        key = t.target.strip().lower()

        if key in seen_targets:
            continue
        seen_targets.add(key)

        resolved_mentions = resolved_mentions_by_target.get(key, [])

        if gold_targets:
            label_source, sentiment_label = match_gold_label(
                t.target, t.embedded_entities, resolved_mentions, gold_targets
            )
        else:
            label_source, sentiment_label = "needs_annotation", None

        record = {
            "sentence_id": sentence_id,
            "sentence": sentence,
            "target_text": t.target,
            "role": t.role,
            "event": t.event,
            "confidence": t.confidence,
            "embedded_entities": t.embedded_entities,
            "resolved_mentions": resolved_mentions,
            "resolved_entity_type": resolved_type_by_target.get(key),
            "label_source": label_source,
            "sentiment_label": sentiment_label,
        }

        if predictor is not None:

            located = locate_target_text(sentence, t.target)

            if located is None:
                record["predicted_sentiment_label"] = None
                record["predicted_sentiment_numeric"] = None
                record["predicted_confidence"] = None
                record["sentiment_agreement"] = None
                record["prediction_error"] = (
                    f"Target '{t.target}' could not be located in sentence."
                )
            else:
                try:
                    prediction = predictor.predict(sentence, located)
                except Exception as exc:
                    record["predicted_sentiment_label"] = None
                    record["predicted_sentiment_numeric"] = None
                    record["predicted_confidence"] = None
                    record["sentiment_agreement"] = None
                    record["prediction_error"] = str(exc)
                else:
                    pred_label = prediction["prediction"]
                    pred_numeric = LABEL_TO_NUMERIC.get(pred_label)

                    record["predicted_sentiment_label"] = pred_label
                    record["predicted_sentiment_numeric"] = pred_numeric
                    record["predicted_confidence"] = prediction["confidence"]
                    record["prediction_error"] = None

                    if sentiment_label is not None:
                        record["sentiment_agreement"] = (
                            pred_numeric == sentiment_label
                        )
                    else:
                        record["sentiment_agreement"] = None

        records.append(record)

    return records


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Generate target-selection (+ optional live sentiment prediction) dataset from raw sentences."
    )
    parser.add_argument(
        "input_path", type=Path, nargs="?", default=None,
        help="Text file, one raw sentence per line. Omit to run on the built-in sample.",
    )
    parser.add_argument(
        "output_path", type=Path, nargs="?", default=None,
        help="Where to write the JSONL dataset.",
    )
    parser.add_argument(
        "--gold", type=Path, default=None,
        help="Gold CSV (sentence,target,polarity) to auto-fill trustworthy sentiment labels from.",
    )
    parser.add_argument(
        "--model", default="en_core_web_sm",
        help="spaCy model to use (default: en_core_web_sm). Try en_core_web_trf for better accuracy.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="spaCy nlp.pipe() batch size for faster inference (default: 16).",
    )
    parser.add_argument(
        "--predict", action="store_true",
        help="Also run the trained sentiment model live on every target (loads transformer_predictor.TransformerPredictor).",
    )
    args = parser.parse_args()

    if args.input_path is not None and args.output_path is None:
        parser.error("output_path is required when input_path is given")

    if args.input_path is not None:
        sentences = [
            line.strip()
            for line in args.input_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        output_path = args.output_path
        print(f"Loaded {len(sentences)} sentences from {args.input_path}")
    else:
        sentences = SAMPLE_SENTENCES
        output_path = Path("dataset_sample.jsonl")
        print(
            f"No input file given -- running on {len(sentences)} "
            f"built-in sample sentences, writing to {output_path}"
        )

    print(f"Using spaCy model: {args.model}  (batch size: {args.batch_size})")

    gold_data: dict[str, list[tuple[str, float]]] = {}
    if args.gold is not None:
        print(f"Loading gold data from {args.gold}...")
        gold_data = load_gold_data(args.gold)
        print(f"Loaded gold data for {len(gold_data)} unique sentences")

    predictor = None
    if args.predict:
        print("Loading trained sentiment model (--predict)...")
        from transformer_predictor import TransformerPredictor
        predictor = TransformerPredictor()

    extractor = SemanticExtractor(model_name=args.model)
    graph_builder = SemanticGraphBuilder()
    selector = TargetSelector()
    resolver = EntityResolver()

    total_records = 0
    total_sentences_with_no_targets = 0
    label_source_counts: dict[str, int] = defaultdict(int)
    agreement_counts = {"agree": 0, "disagree": 0, "no_gold": 0}

    with open(output_path, "w", encoding="utf-8") as f:

        for index, (sentence, representation) in enumerate(
            iter_representations(sentences, extractor, args.batch_size),
            start=1,
        ):

            if index == 1 or index % 50 == 0 or index == len(sentences):
                print(f"  ...{index}/{len(sentences)} sentences processed")

            sentence_id = f"s{index:04d}"
            sentence_gold = gold_data.get(sentence.strip(), [])

            records = build_records(
                sentence_id, sentence, representation,
                graph_builder, selector, resolver,
                gold_targets=sentence_gold, predictor=predictor,
            )

            if not records:
                total_sentences_with_no_targets += 1

            for record in records:
                f.write(json.dumps(record, ensure_ascii=False))
                f.write("\n")
                total_records += 1
                label_source_counts[record["label_source"]] += 1

                if args.predict:
                    agreement = record.get("sentiment_agreement")
                    if agreement is True:
                        agreement_counts["agree"] += 1
                    elif agreement is False:
                        agreement_counts["disagree"] += 1
                    else:
                        agreement_counts["no_gold"] += 1

    print()
    print("=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"Sentences processed:        {len(sentences)}")
    print(f"Sentences with no targets:  {total_sentences_with_no_targets}")
    print(f"Total records written:      {total_records}")
    print(f"Output file:                {output_path}")

    if args.gold is not None:
        print()
        print("Label source breakdown:")
        for source in ("gold_exact", "gold_embedded_or_resolved", "needs_annotation"):
            count = label_source_counts.get(source, 0)
            pct = count / total_records if total_records else 0.0
            print(f"  {source:28} {count:5}  ({pct:.1%})")

    if args.predict:
        print()
        print("Live prediction vs gold agreement:")
        compared = agreement_counts["agree"] + agreement_counts["disagree"]
        acc = agreement_counts["agree"] / compared if compared else 0.0
        print(f"  Agree:    {agreement_counts['agree']:6}")
        print(f"  Disagree: {agreement_counts['disagree']:6}")
        print(f"  No gold to compare: {agreement_counts['no_gold']:6}")
        if compared:
            print(f"  Agreement rate (where comparable): {acc:.1%}")


if __name__ == "__main__":
    main()
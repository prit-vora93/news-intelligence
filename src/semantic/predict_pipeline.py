"""
semantic/predict_pipeline.py
------------------------------

The actual "News Intelligence" end-to-end pipeline: take a RAW
sentence (no gold data needed) and produce (target, sentiment)
predictions automatically.

    Sentence
        v
    SemanticExtractor -> SemanticGraphBuilder -> TargetSelector
        v
    Candidate targets (select_all())
        v
    EntityResolver (pronoun/generic-term resolution, for context)
        v
    TransformerPredictor (your trained models/newsmtc_distilbert_final)
        v
    (target, sentiment, confidence) for every candidate

This connects two previously-separate halves of the project: the
rule-based semantic pipeline (validated extensively this session,
~96%+ recall on gold data) and the already-trained sentiment
classifier (src/transformer_predictor.py), which until now had no
way to get targets except from pre-existing gold-labeled CSVs.

Usage:
    PYTHONPATH=src python src/semantic/predict_pipeline.py \\
        "Meta announced that it will invest billions of dollars in artificial intelligence."

    Or from a file, one sentence per line:
    PYTHONPATH=src python src/semantic/predict_pipeline.py --file sentences.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from semantic.semantic_extractor import SemanticExtractor
from semantic.semantic_graph import SemanticGraphBuilder
from semantic.target_selector import TargetSelector
from semantic.entity_resolver import EntityResolver

# transformer_predictor.py lives directly under src/, not under
# semantic/ -- this import works when run with PYTHONPATH=src.
from transformer_predictor import TransformerPredictor


def locate_target_text(sentence: str, target_text: str) -> str | None:
    """
    Find the REAL substring of `sentence` that corresponds to
    target_text, tolerating minor punctuation/whitespace
    differences between our extracted (cleaned) target text and
    the original sentence.

    Why this exists: target_selector.py deliberately strips
    list-separator commas from extracted text for readability
    (e.g. "most Americans, including wealthy ones" -> "most
    Americans including wealthy ones"). That's fine for display,
    but it means a literal `sentence.find(target_text)` can fail
    even though the target genuinely IS in the sentence --
    confirmed via real pipeline output, where a valid target got
    silently skipped for exactly this reason.

    Returns the actual matching substring FROM THE ORIGINAL
    SENTENCE (comma/spacing included, whatever it really is) so the
    model gets marked around real text, not our cleaned version --
    or None if no reasonable match can be found at all.
    """

    # 1. Try an exact match first (the common, simple case).
    if target_text in sentence:
        return target_text

    # 2. Tolerant match: treat runs of whitespace/commas in
    # target_text as flexible separators, so "most Americans
    # including wealthy ones" can still match "most Americans,
    # including wealthy ones" in the original sentence.
    tokens = target_text.split()

    if not tokens:
        return None

    pattern = r"[\s,]+".join(re.escape(t) for t in tokens)

    match = re.search(pattern, sentence)

    if match is None:
        return None

    return match.group(0)


def predict_for_sentence(
    sentence: str,
    extractor: SemanticExtractor,
    graph_builder: SemanticGraphBuilder,
    selector: TargetSelector,
    resolver: EntityResolver,
    predictor: TransformerPredictor,
) -> list[dict]:
    """
    Run the full pipeline on one sentence and return a prediction
    record per candidate target.

    Each record includes both the EXTRACTION metadata (role, event,
    confidence -- how sure we are this is a meaningful target) and
    the PREDICTION metadata (sentiment label, confidence -- how sure
    the trained model is about the sentiment). These are two
    different kinds of confidence and are kept separate rather than
    combined into one number.

    If a target's text can't be located in the sentence (should be
    rare given our spacing fixes, but not impossible for edge
    cases), that target is skipped with a note rather than crashing
    the whole batch -- consistent with this project's "leave
    unresolved/skip rather than guess wrong" principle throughout.
    """

    representation = extractor.extract(sentence)
    graph = graph_builder.build(representation)
    targets = selector.select_all(graph)

    if not targets:
        return []

    resolution = resolver.resolve([graph], [targets])

    resolved_mentions_by_target: dict[str, list[str]] = {}
    for entity in resolution.resolved_entities:
        key = entity.canonical_name.strip().lower()
        resolved_mentions_by_target[key] = [m.text for m in entity.mentions]

    records: list[dict] = []
    seen_targets: set[str] = set()

    for t in targets:

        if t.target is None or t.confidence == "none":
            continue

        key = t.target.strip().lower()
        if key in seen_targets:
            continue
        seen_targets.add(key)

        located_text = locate_target_text(sentence, t.target)

        if located_text is None:
            records.append(
                {
                    "target_text": t.target,
                    "role": t.role,
                    "extraction_confidence": t.confidence,
                    "embedded_entities": t.embedded_entities,
                    "resolved_mentions": resolved_mentions_by_target.get(key, []),
                    "sentiment": None,
                    "sentiment_confidence": None,
                    "probabilities": None,
                    "error": f"Target '{t.target}' could not be located in sentence, even fuzzily.",
                }
            )
            continue

        try:
            prediction = predictor.predict(sentence, located_text)
        except ValueError as exc:
            records.append(
                {
                    "target_text": t.target,
                    "role": t.role,
                    "extraction_confidence": t.confidence,
                    "embedded_entities": t.embedded_entities,
                    "resolved_mentions": resolved_mentions_by_target.get(key, []),
                    "sentiment": None,
                    "sentiment_confidence": None,
                    "probabilities": None,
                    "error": str(exc),
                }
            )
            continue

        records.append(
            {
                "target_text": t.target,
                "role": t.role,
                "extraction_confidence": t.confidence,
                "embedded_entities": t.embedded_entities,
                "resolved_mentions": resolved_mentions_by_target.get(key, []),
                "sentiment": prediction["prediction"],
                "sentiment_confidence": prediction["confidence"],
                "probabilities": prediction["probabilities"],
                "error": None,
            }
        )

    return records


def print_records(sentence: str, records: list[dict]) -> None:

    print()
    print("=" * 70)
    print(f"SENTENCE: {sentence}")
    print("=" * 70)

    if not records:
        print("  No targets found.")
        return

    for r in records:

        if r["error"]:
            print(f"  target={r['target_text']!r:30} -- SKIPPED ({r['error']})")
            continue

        print(
            f"  target={r['target_text']!r:30} "
            f"role={r['role']:8} "
            f"extraction_conf={r['extraction_confidence']:6} "
            f"-> {r['sentiment']:8} "
            f"(model confidence {r['sentiment_confidence']:.1%})"
        )

        if r["embedded_entities"]:
            print(f"      embedded_entities: {r['embedded_entities']}")
        if r["resolved_mentions"]:
            print(f"      resolved_mentions: {r['resolved_mentions']}")


def main() -> None:

    parser = argparse.ArgumentParser(
        description="End-to-end target extraction + sentiment prediction on raw sentences."
    )
    parser.add_argument(
        "sentence",
        nargs="?",
        default=None,
        help="A single sentence to analyze (wrap in quotes).",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Text file, one sentence per line, to analyze in batch.",
    )
    parser.add_argument(
        "--model",
        default="en_core_web_sm",
        help="spaCy model to use (default: en_core_web_sm).",
    )
    args = parser.parse_args()

    if args.sentence is None and args.file is None:
        parser.error("Provide either a sentence argument or --file")

    if args.sentence is not None:
        sentences = [args.sentence]
    else:
        sentences = [
            line.strip()
            for line in args.file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    print("Loading semantic pipeline...")
    extractor = SemanticExtractor(model_name=args.model)
    graph_builder = SemanticGraphBuilder()
    selector = TargetSelector()
    resolver = EntityResolver()

    print("Loading trained sentiment model...")
    predictor = TransformerPredictor()

    print(f"\nAnalyzing {len(sentences)} sentence(s)...")

    for sentence in sentences:
        records = predict_for_sentence(
            sentence, extractor, graph_builder, selector, resolver, predictor
        )
        print_records(sentence, records)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
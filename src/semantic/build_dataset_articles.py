"""
semantic/build_dataset_articles.py
-------------------------------------

Article-aware version of build_dataset.py: processes whole ARTICLES
(ordered sentence lists from the raw NewsMTSC dataset files, grouped
via article_loader.py), calling EntityResolver.resolve() ONCE PER
ARTICLE with all of that article's sentence graphs together -- so
cross-sentence resolution (see entity_resolver.py, "look one sentence
back") actually has multiple real sentences to work across.

This is a SEPARATE script from build_dataset.py, not a modification
to it, because the input data format is fundamentally different: raw
NewsMTSC JSONL (primary_gid-based, article-structured) vs. the flat
train_target.csv (individual sentences, no article grouping) used by
the rest of this project's validated pipeline. Keeping them separate
avoids destabilizing the extensively-validated flat-sentence
workflow.

Gold labels come directly from the raw dataset's own "targets" field
(mention + polarity) for each sentence -- no separate --gold CSV
needed here, unlike build_dataset.py.

Usage:
    PYTHONPATH=src python src/semantic/build_dataset_articles.py \\
        NewsMTSC/NewsSentiment/controller_data/datasets/NewsMTSC-dataset/train.jsonl \\
        dataset_articles.jsonl \\
        --model en_core_web_trf --predict
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from semantic.article_loader import load_articles
from semantic.semantic_extractor import SemanticExtractor
from semantic.semantic_graph import SemanticGraphBuilder
from semantic.target_selector import TargetSelector
from semantic.entity_resolver import EntityResolver
from semantic.predict_pipeline import locate_target_text


LABEL_TO_NUMERIC = {"Negative": 2.0, "Neutral": 4.0, "Positive": 6.0}
NUMERIC_TO_LABEL = {v: k for k, v in LABEL_TO_NUMERIC.items()}


def normalize(text: str) -> str:
    return text.strip().lower()


def match_gold_label(
    target_text: str,
    embedded_entities: list[str],
    resolved_mentions: list[str],
    gold_targets: list[tuple[str, float]],
) -> tuple[str, float | None]:
    """Same matching logic as build_dataset.py, reused here."""

    target_norm = normalize(target_text)
    embedded_norm = {normalize(e) for e in embedded_entities}
    resolved_norm = {normalize(m) for m in resolved_mentions}

    for gold_target, polarity in gold_targets:
        if normalize(gold_target) == target_norm:
            return "gold_exact", polarity

    for gold_target, polarity in gold_targets:
        gold_norm = normalize(gold_target)
        if gold_norm in embedded_norm or gold_norm in resolved_norm:
            return "gold_embedded_or_resolved", polarity

    return "needs_annotation", None


def process_article(
    article_id: str,
    sentences: list[tuple[int, str, list[tuple[str, float]]]],
    extractor: SemanticExtractor,
    graph_builder: SemanticGraphBuilder,
    selector: TargetSelector,
    predictor,
) -> list[dict]:
    """
    Process one article's sentences TOGETHER, in order, so
    cross-sentence resolution can actually link across them.
    """

    graphs = []
    target_lists = []
    sentence_texts = []
    original_indices = []
    gold_by_position = []

    for sentence_index, sentence_text, gold_targets in sentences:
        representation = extractor.extract(sentence_text)
        graph = graph_builder.build(representation)
        targets = selector.select_all(graph)

        graphs.append(graph)
        target_lists.append(targets)
        sentence_texts.append(sentence_text)
        original_indices.append(sentence_index)
        gold_by_position.append(gold_targets)

    resolver = EntityResolver()
    resolution = resolver.resolve(graphs, target_lists)

    resolved_mentions_by_target: dict[str, list[str]] = {}
    resolved_type_by_target: dict[str, str | None] = {}

    for entity in resolution.resolved_entities:
        key = entity.canonical_name.strip().lower()
        resolved_mentions_by_target[key] = [m.text for m in entity.mentions]
        resolved_type_by_target[key] = entity.entity_type

    records: list[dict] = []

    for position, targets in enumerate(target_lists):

        sentence_text = sentence_texts[position]
        original_sentence_index = original_indices[position]
        gold_targets = gold_by_position[position]

        seen_targets: set[str] = set()

        for t in targets:

            if t.target is None or t.confidence == "none":
                continue

            key = t.target.strip().lower()
            if key in seen_targets:
                continue
            seen_targets.add(key)

            resolved_mentions = resolved_mentions_by_target.get(key, [])

            label_source, sentiment_label = match_gold_label(
                t.target, t.embedded_entities, resolved_mentions, gold_targets
            )

            record = {
                "article_id": article_id,
                "sentence_index": original_sentence_index,
                "sentence": sentence_text,
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

                located = locate_target_text(sentence_text, t.target)

                if located is None:
                    record["predicted_sentiment_label"] = None
                    record["predicted_confidence"] = None
                    record["sentiment_agreement"] = None
                else:
                    try:
                        prediction = predictor.predict(sentence_text, located)
                    except Exception:
                        record["predicted_sentiment_label"] = None
                        record["predicted_confidence"] = None
                        record["sentiment_agreement"] = None
                    else:
                        pred_label = prediction["prediction"]
                        pred_numeric = LABEL_TO_NUMERIC.get(pred_label)

                        record["predicted_sentiment_label"] = pred_label
                        record["predicted_confidence"] = prediction["confidence"]

                        if sentiment_label is not None:
                            record["sentiment_agreement"] = (pred_numeric == sentiment_label)
                        else:
                            record["sentiment_agreement"] = None

            records.append(record)

    return records


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Article-aware dataset generation with cross-sentence entity resolution."
    )
    parser.add_argument("input_path", type=Path, help="Raw NewsMTSC JSONL file (e.g. train.jsonl).")
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--model", default="en_core_web_sm")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--limit-articles", type=int, default=None, help="Process only the first N articles (for quick testing).")
    args = parser.parse_args()

    print(f"Loading articles from {args.input_path}...")
    articles = load_articles(args.input_path)
    print(f"Loaded {len(articles)} articles")

    article_items = list(articles.items())
    if args.limit_articles is not None:
        article_items = article_items[: args.limit_articles]
        print(f"Limiting to first {args.limit_articles} articles")

    print(f"Using spaCy model: {args.model}")
    extractor = SemanticExtractor(model_name=args.model)
    graph_builder = SemanticGraphBuilder()
    selector = TargetSelector()

    predictor = None
    if args.predict:
        print("Loading trained sentiment model (--predict)...")
        from transformer_predictor import TransformerPredictor
        predictor = TransformerPredictor()

    total_records = 0
    total_articles_processed = 0
    label_source_counts: dict[str, int] = defaultdict(int)
    agreement_counts = {"agree": 0, "disagree": 0, "no_gold": 0}
    cross_sentence_entities_count = 0

    with open(args.output_path, "w", encoding="utf-8") as f:

        for i, (article_id, sentences) in enumerate(article_items, start=1):

            if i == 1 or i % 50 == 0 or i == len(article_items):
                print(f"  ...{i}/{len(article_items)} articles processed")

            records = process_article(
                article_id, sentences, extractor, graph_builder, selector, predictor
            )

            total_articles_processed += 1

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
    print("ARTICLE-AWARE DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"Articles processed:   {total_articles_processed}")
    print(f"Total records:        {total_records}")
    print(f"Output file:          {args.output_path}")
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

"""
semantic/live_ingest.py
-------------------------

The actual "live tool" piece: RSS feed(s) -> real article text ->
sentences -> our full target extraction + sentiment pipeline ->
JSONL output (appended, so repeated runs build a running history).

Pipeline:
    RSS feed(s) (feedparser)
        v
    Article URLs
        v
    trafilatura.fetch_url() + extract()  -- clean article text
        v
    spaCy sentence splitting
        v
    predict_pipeline.predict_for_sentence()  -- our existing,
        validated extraction + resolution + sentiment pipeline
        v
    JSONL file (appended)

Requires: feedparser, trafilatura (pip install feedparser trafilatura)

Usage:
    # Single feed
    PYTHONPATH=src python src/semantic/live_ingest.py \\
        "https://feeds.npr.org/1001/rss.xml" \\
        --max-articles 3 --model en_core_web_trf

    # Multiple feeds in one run
    PYTHONPATH=src python src/semantic/live_ingest.py \\
        "https://feeds.npr.org/1001/rss.xml" "https://feeds.bbci.co.uk/news/rss.xml" \\
        --max-articles 3 --model en_core_web_trf --output live_results.jsonl
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import trafilatura

from semantic.semantic_extractor import SemanticExtractor
from semantic.semantic_graph import SemanticGraphBuilder
from semantic.target_selector import TargetSelector
from semantic.entity_resolver import EntityResolver
from semantic.predict_pipeline import locate_target_text, print_records
from transformer_predictor import TransformerPredictor


def process_article(
    sentences: list[str],
    extractor: SemanticExtractor,
    graph_builder: SemanticGraphBuilder,
    selector: TargetSelector,
    predictor: TransformerPredictor,
) -> list[tuple[str, list[dict]]]:
    """
    Process a whole article's sentences TOGETHER, calling
    EntityResolver.resolve() ONCE for the entire article (not once
    per sentence) -- so cross-sentence resolution (see
    entity_resolver.py, "look one sentence back") actually has real
    sentences to work across.

    This mirrors process_article() in build_dataset_articles.py,
    adapted for live sentences with no gold data. Fixes a real gap
    found via actual live_ingest.py output: real articles have
    plenty of pronouns spread across many sentences (e.g. "Vance"
    mentioned repeatedly, then later just "he"), and single-sentence-
    only resolution (predict_pipeline.predict_for_sentence(), used
    per-sentence) was never getting the benefit of the cross-
    sentence capability this project already built and validated.

    Returns a list of (sentence, records) pairs, in order, matching
    predict_pipeline.print_records()'s expected shape.
    """

    graphs = []
    target_lists = []

    for sentence in sentences:
        representation = extractor.extract(sentence)
        graph = graph_builder.build(representation)
        targets = selector.select_all(graph)
        graphs.append(graph)
        target_lists.append(targets)

    resolver = EntityResolver()
    resolution = resolver.resolve(graphs, target_lists)

    resolved_mentions_by_target: dict[str, list[str]] = {}

    for entity in resolution.resolved_entities:
        key = entity.canonical_name.strip().lower()
        resolved_mentions_by_target[key] = [m.text for m in entity.mentions]

    results: list[tuple[str, list[dict]]] = []

    for sentence, targets in zip(sentences, target_lists):

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

            located = locate_target_text(sentence, t.target)

            if located is None:
                records.append(
                    {
                        "target_text": t.target,
                        "role": t.role,
                        "extraction_confidence": t.confidence,
                        "embedded_entities": t.embedded_entities,
                        "resolved_mentions": resolved_mentions,
                        "sentiment": None,
                        "sentiment_confidence": None,
                        "probabilities": None,
                        "error": f"Target '{t.target}' could not be located in sentence.",
                    }
                )
                continue

            try:
                prediction = predictor.predict(sentence, located)
            except Exception as exc:
                records.append(
                    {
                        "target_text": t.target,
                        "role": t.role,
                        "extraction_confidence": t.confidence,
                        "embedded_entities": t.embedded_entities,
                        "resolved_mentions": resolved_mentions,
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
                    "resolved_mentions": resolved_mentions,
                    "sentiment": prediction["prediction"],
                    "sentiment_confidence": prediction["confidence"],
                    "probabilities": prediction["probabilities"],
                    "error": None,
                }
            )

        results.append((sentence, records))

    return results


def get_article_urls(feed_url: str, max_articles: int) -> list[str]:
    """Parse an RSS feed and return up to max_articles entry URLs."""

    feed = feedparser.parse(feed_url)

    if feed.bozo:
        print(f"Warning: feed may be malformed ({feed.bozo_exception})")

    urls = [entry.link for entry in feed.entries[:max_articles] if hasattr(entry, "link")]

    return urls


def fetch_article_text(url: str) -> str | None:
    """Download and extract clean article text from a URL."""

    downloaded = trafilatura.fetch_url(url)

    if downloaded is None:
        return None

    return trafilatura.extract(downloaded)


def split_into_sentences(text: str, extractor: SemanticExtractor) -> list[str]:
    """
    Split article text into sentences using spaCy, treating newlines
    as HARD sentence boundaries first.

    Why: trafilatura separates a headline, dateline, and the article
    body only with newlines, not punctuation (e.g. "Sri Lanka court
    convicts 15...\\nCOLOMBO, Sri Lanka --\\nA Sri Lankan court found...").
    spaCy's sentence segmenter doesn't treat a bare newline as a
    boundary on its own, so without this, a headline and its
    following dateline get merged into one garbled "sentence" --
    confirmed via real live_ingest.py output on real RSS articles.
    Splitting on newlines first, then running spaCy's OWN sentence
    segmentation WITHIN each resulting line, keeps proper multi-
    sentence body paragraphs correctly split while guaranteeing no
    sentence ever spans a newline.

    Known remaining limitation, not fixed here: a standalone headline
    or dateline line (e.g. "COLOMBO, Sri Lanka --") still becomes its
    own low-value "sentence" rather than being filtered out entirely
    -- it just no longer corrupts the real content sentences next to
    it, which was the actual bug.
    """

    sentences: list[str] = []

    for line in text.split("\n"):

        line = line.strip()

        if not line:
            continue

        doc = extractor.nlp(line)

        sentences.extend(
            sent.text.strip() for sent in doc.sents if sent.text.strip()
        )

    return sentences


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Live pipeline: RSS feed(s) -> real articles -> target extraction + sentiment -> JSONL."
    )
    parser.add_argument("feed_urls", nargs="+", help="One or more RSS feed URLs.")
    parser.add_argument("--max-articles", type=int, default=3, help="How many articles PER FEED to process.")
    parser.add_argument("--max-sentences-per-article", type=int, default=15, help="Cap on sentences per article (long articles can be slow).")
    parser.add_argument("--model", default="en_core_web_sm")
    parser.add_argument(
        "--output", type=Path, default=Path("live_results.jsonl"),
        help="JSONL output file. APPENDED to (not overwritten), so repeated runs build a running history.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress the per-sentence console printout (still writes to --output).",
    )
    args = parser.parse_args()

    print("Loading semantic pipeline...")
    extractor = SemanticExtractor(model_name=args.model)
    graph_builder = SemanticGraphBuilder()
    selector = TargetSelector()

    print("Loading trained sentiment model...")
    predictor = TransformerPredictor()

    total_articles = 0
    total_articles_skipped = 0
    total_records = 0

    with open(args.output, "a", encoding="utf-8") as out_file:

        for feed_url in args.feed_urls:

            print()
            print(f"Fetching RSS feed: {feed_url}")
            urls = get_article_urls(feed_url, args.max_articles)
            print(f"Found {len(urls)} article(s) to process")

            for url in urls:

                if not args.quiet:
                    print()
                    print("#" * 70)
                    print(f"ARTICLE: {url}")
                    print("#" * 70)

                article_text = fetch_article_text(url)

                if not article_text:
                    if not args.quiet:
                        print("  Could not extract article text -- skipping.")
                    total_articles_skipped += 1
                    continue

                total_articles += 1

                sentences = split_into_sentences(article_text, extractor)
                sentences = sentences[: args.max_sentences_per_article]

                if not args.quiet:
                    print(f"  {len(sentences)} sentence(s) to analyze")

                fetched_at = datetime.now(timezone.utc).isoformat()

                article_results = process_article(
                    sentences, extractor, graph_builder, selector, predictor
                )

                for sentence, records in article_results:

                    if not args.quiet:
                        print_records(sentence, records)

                    for record in records:

                        out_record = {
                            "feed_url": feed_url,
                            "article_url": url,
                            "fetched_at": fetched_at,
                            "sentence": sentence,
                            **record,
                        }

                        out_file.write(json.dumps(out_record, ensure_ascii=False))
                        out_file.write("\n")
                        total_records += 1

                out_file.flush()

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Feeds processed:        {len(args.feed_urls)}")
    print(f"Articles processed:     {total_articles}")
    print(f"Articles skipped:       {total_articles_skipped}")
    print(f"Records written:        {total_records}")
    print(f"Output file (appended): {args.output}")


if __name__ == "__main__":
    main()
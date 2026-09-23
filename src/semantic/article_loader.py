"""
semantic/article_loader.py
----------------------------

Loads the RAW NewsMTSC dataset files (train.jsonl / devtest_*.jsonl --
not the flattened train_target.csv used everywhere else in this
project) and groups sentences back into their original articles,
in order, using the primary_gid field.

Why this exists: cross-sentence entity resolution (see
entity_resolver.py) needs multiple sentences from the SAME article,
in order, passed together in one resolve() call. train_target.csv
has no article-grouping information at all -- but the raw dataset
files do, encoded in primary_gid.

Confirmed primary_gid format (see entity_resolver_scope_v1.md
Section 10 for how this was verified against real data):

    {source}_{article_id_part1}_{article_id_part2}_{sentence_index}_{mention_text}_{char_from}_{char_to}

IMPORTANT CAVEAT: sentence indices per article are SPARSE, not
contiguous. Only sentences containing a qualifying coreference-
cluster mention were included in the original dataset creation, so
"one sentence back" in this data means "the nearest available prior
EXAMPLE sentence," not necessarily literal text-adjacency in the
real original article.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def parse_gid(gid: str) -> tuple[str, int, str, int, int] | None:
    """
    Parse a primary_gid into (article_id, sentence_index,
    mention_text, char_from, char_to).

    Returns None if the gid doesn't match the expected format
    (defensive -- don't crash a whole load over one malformed row).
    """

    parts = gid.split("_")

    if len(parts) < 6:
        return None

    article_id = "_".join(parts[0:3])

    try:
        sentence_index = int(parts[3])
        char_from = int(parts[-2])
        char_to = int(parts[-1])
    except ValueError:
        return None

    mention_text = "_".join(parts[4:-2])

    return article_id, sentence_index, mention_text, char_from, char_to


def load_articles(path: Path) -> dict[str, list[tuple[int, str, list[tuple[str, float]]]]]:
    """
    Load a raw NewsMTSC-format JSONL file and group sentences by
    article, IN ORDER.

    Returns:
        {
            article_id: [
                (sentence_index, sentence_text, [(target_mention, polarity), ...]),
                ...  # sorted by sentence_index
            ]
        }

    Multiple raw records sharing the same (article_id, sentence_index)
    -- e.g. two different targets in the same sentence, each with
    their own top-level record -- are merged into one sentence entry
    with a combined target list.
    """

    raw: dict[str, dict[int, dict]] = defaultdict(dict)

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            gid = record.get("primary_gid")
            if not gid:
                continue

            parsed = parse_gid(gid)
            if parsed is None:
                continue

            article_id, sentence_index, _, _, _ = parsed
            sentence_text = record.get("sentence_normalized", "")

            if not sentence_text:
                continue

            if sentence_index not in raw[article_id]:
                raw[article_id][sentence_index] = {
                    "text": sentence_text,
                    "targets": [],
                }

            for target in record.get("targets", []):
                mention = target.get("mention")
                polarity = target.get("polarity")
                if mention:
                    raw[article_id][sentence_index]["targets"].append(
                        (mention, polarity)
                    )

    articles: dict[str, list[tuple[int, str, list[tuple[str, float]]]]] = {}

    for article_id, sentences_by_index in raw.items():
        ordered = sorted(sentences_by_index.items())
        articles[article_id] = [
            (idx, data["text"], data["targets"])
            for idx, data in ordered
        ]

    return articles

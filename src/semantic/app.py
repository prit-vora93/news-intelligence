"""
semantic/app.py
------------------

Interactive frontend: paste a news article URL OR paste article
text directly, get back extracted targets and sentiment -- the
same validated pipeline used everywhere else this project (target
extraction, cross-sentence resolution, sentiment prediction), just
wrapped in a simple web UI instead of a terminal command.

Reuses process_article(), fetch_article_text(), and
split_into_sentences() from live_ingest.py directly -- no new
backend logic, just a UI layer on top of what's already validated.

Usage:
    pip install streamlit
    PYTHONPATH=src streamlit run src/semantic/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from semantic.semantic_extractor import SemanticExtractor
from semantic.semantic_graph import SemanticGraphBuilder
from semantic.target_selector import TargetSelector
from semantic.live_ingest import fetch_article_text, split_into_sentences, process_article
from semantic.analyze_entity_sentiment import canonical_entity, merge_aliases, derive_alias_mapping
from semantic.analyze_sentiment_trends import build_trend_stats, merge_aliases_per_day
from transformer_predictor import TransformerPredictor


KNOWN_OUTLET_NAMES = {
    "npr.org": "NPR",
    "bbci.co.uk": "BBC",
}


def outlet_display_name(feed_url: str) -> str:
    """
    Best-effort short display name for an outlet from its feed URL.
    Known outlets get a clean name; anything else falls back to a
    simple domain-based guess, which may look a little rough for
    unfamiliar feeds -- there's no reliable way to always get a
    perfect outlet name from a bare RSS URL alone.
    """

    from urllib.parse import urlparse

    netloc = urlparse(feed_url).netloc
    netloc = netloc.removeprefix("feeds.").removeprefix("www.")

    for domain, name in KNOWN_OUTLET_NAMES.items():
        if domain in netloc:
            return name

    return netloc.split(".")[0].upper()


@st.cache_resource
def load_pipeline(model_name: str):
    """
    Load the semantic pipeline + trained sentiment model ONCE and
    cache it across requests -- these are slow to load (spaCy
    transformer model, DistilBERT), so reloading on every single
    analysis would make the app painfully slow to use.
    """

    extractor = SemanticExtractor(model_name=model_name)
    graph_builder = SemanticGraphBuilder()
    selector = TargetSelector()
    predictor = TransformerPredictor()

    return extractor, graph_builder, selector, predictor


def summarize_by_entity(all_records: list[dict]) -> dict[str, dict]:
    """
    Aggregate this ONE article's records by entity, same
    canonicalization AND alias-merging used everywhere else this
    project (both reused from analyze_entity_sentiment.py) -- gives
    an "overall sentiment toward X in this article" summary, not
    just a flat per-sentence list.
    """

    summary: dict[str, dict] = {}

    for record in all_records:

        if record.get("sentiment") is None:
            continue

        entity = canonical_entity(record)
        if entity is None:
            continue

        key = entity.lower()

        if key not in summary:
            summary[key] = {
                "display_name": entity,
                "Positive": 0, "Neutral": 0, "Negative": 0, "total": 0,
                "confidences": [],
            }

        summary[key][record["sentiment"]] += 1
        summary[key]["total"] += 1
        summary[key]["confidences"].append(record.get("sentiment_confidence") or 0.0)

    return merge_aliases(summary)


st.set_page_config(page_title="News Intelligence", layout="wide")

st.title("News Intelligence")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "spaCy model",
        ["en_core_web_trf", "en_core_web_sm"],
        help="en_core_web_trf is slower but more accurate (recommended). en_core_web_sm is faster.",
    )

tab1, tab2, tab3 = st.tabs(["Analyze an Article", "Live Dashboard", "Outlet Comparison"])

with tab1:

    st.caption("Paste a news article URL or the article text directly to see extracted targets and sentiment.")

    input_mode = st.radio("Input type", ["Article URL", "Paste article text"], horizontal=True)

    article_text = None

    if input_mode == "Article URL":
        url = st.text_input("Article URL", placeholder="https://www.example.com/news/some-article")
        if url and st.button("Analyze", type="primary"):
            with st.spinner("Fetching and extracting article text..."):
                article_text = fetch_article_text(url)
            if article_text is None:
                st.error("Could not extract article text from that URL. It may be blocked, paywalled, or not a real article page.")
                article_text = None
    else:
        pasted = st.text_area("Article text", height=200, placeholder="Paste the full article text here...")
        if pasted and st.button("Analyze", type="primary"):
            article_text = pasted

    if article_text:

        with st.spinner(f"Loading pipeline ({model_name})... this can take a while the first time."):
            extractor, graph_builder, selector, predictor = load_pipeline(model_name)

        with st.spinner("Splitting into sentences and analyzing..."):
            sentences = split_into_sentences(article_text, extractor)
            results = process_article(sentences, extractor, graph_builder, selector, predictor)

        all_records = [record for _, records in results for record in records if record.get("sentiment") is not None]

        if not all_records:
            st.warning("No targets with sentiment were found in this article.")
        else:
            st.subheader("Overall sentiment by entity (this article)")

            entity_summary = summarize_by_entity(all_records)
            ranked = sorted(entity_summary.items(), key=lambda kv: -kv[1]["total"])

            cols = st.columns(min(4, len(ranked)) or 1)

            for i, (key, stats) in enumerate(ranked[:8]):
                with cols[i % len(cols)]:
                    total = stats["total"]
                    pos_pct = stats["Positive"] / total
                    neu_pct = stats["Neutral"] / total
                    neg_pct = stats["Negative"] / total
                    st.metric(stats["display_name"], f"{total} mention(s)")
                    st.progress(pos_pct, text=f"Positive {pos_pct:.0%}")
                    st.progress(neu_pct, text=f"Neutral {neu_pct:.0%}")
                    st.progress(neg_pct, text=f"Negative {neg_pct:.0%}")

            st.divider()
            st.subheader("Sentence-by-sentence breakdown")

            for sentence, records in results:

                real_records = [r for r in records if r.get("sentiment") is not None]
                if not real_records:
                    continue

                with st.expander(sentence[:100] + ("..." if len(sentence) > 100 else "")):
                    st.write(sentence)
                    for r in real_records:
                        sentiment_color = {"Positive": "green", "Neutral": "gray", "Negative": "red"}[r["sentiment"]]
                        st.markdown(
                            f"**{r['target_text']}** ({r['role']}) -> "
                            f":{sentiment_color}[{r['sentiment']}] "
                            f"({r['sentiment_confidence']:.0%} confidence)"
                        )
                        if r.get("resolved_mentions"):
                            st.caption(f"Resolved mentions: {', '.join(r['resolved_mentions'])}")

with tab2:

    st.caption(
        "The accumulated data your cron job has been collecting via live_ingest.py -- "
        "this grows on its own every time the scheduled job runs, no manual action needed."
    )

    default_path = "live_results.jsonl"
    dataset_path_str = st.text_input("Data file", value=default_path)
    dataset_path = Path(dataset_path_str)

    if not dataset_path.exists():
        st.info(f"No file found at '{dataset_path}' yet. Run live_ingest.py (or wait for the next cron run) to generate it.")
    else:

        records = []
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))

        all_records = [r for r in records if r.get("sentiment") is not None]
        article_urls = {r.get("article_url") for r in records if r.get("article_url")}
        dates = sorted({r.get("fetched_at", "")[:10] for r in records if r.get("fetched_at")})

        col1, col2, col3 = st.columns(3)
        col1.metric("Total target records", len(records))
        col2.metric("Articles processed", len(article_urls))
        col3.metric("Days of data", len(dates))

        if not all_records:
            st.warning("No records with sentiment predictions found in this file.")
        else:

            st.subheader("Entity leaderboard (all accumulated data)")

            min_mentions = st.slider("Minimum mentions to show", 1, 20, 3)

            entity_summary = summarize_by_entity(all_records)
            eligible = {k: v for k, v in entity_summary.items() if v["total"] >= min_mentions}
            ranked = sorted(eligible.items(), key=lambda kv: -kv[1]["total"])[:20]

            if not ranked:
                st.info("No entities meet the minimum mention threshold yet.")
            else:
                chart_data = pd.DataFrame(
                    {
                        stats["display_name"]: {
                            "Positive": stats["Positive"],
                            "Neutral": stats["Neutral"],
                            "Negative": stats["Negative"],
                        }
                        for _, stats in ranked
                    }
                ).T

                st.bar_chart(chart_data, color=["#2ecc71", "#95a5a6", "#e74c3c"])

                st.divider()
                st.subheader("Sentiment trend for one entity")

                entity_names = [stats["display_name"] for _, stats in ranked]
                selected_entity = st.selectbox("Entity", entity_names)

                per_day_stats = build_trend_stats(records)
                per_day_stats = merge_aliases_per_day(per_day_stats)

                selected_key = selected_entity.lower()

                if selected_key not in per_day_stats or len(per_day_stats[selected_key]) < 2:
                    st.info(
                        f"Only 1 day of data so far for '{selected_entity}' -- "
                        "check back after cron has run across more days to see a real trend."
                    )
                else:
                    days = per_day_stats[selected_key]
                    sorted_days = sorted(days.items())

                    counts_line = "  |  ".join(
                        f"{date}: {d['total']} mention(s)" for date, d in sorted_days
                    )
                    st.caption(counts_line)

                    thin_days = [date for date, d in sorted_days if d["total"] < 10]
                    if thin_days:
                        st.caption(
                            f":orange[Fewer than 10 mentions on {', '.join(thin_days)} -- "
                            "percentages for that day are less reliable and can swing a lot from just a few sentences.]"
                        )

                    trend_df = pd.DataFrame(
                        {
                            date: {
                                "Positive %": d["Positive"] / d["total"] * 100,
                                "Neutral %": d["Neutral"] / d["total"] * 100,
                                "Negative %": d["Negative"] / d["total"] * 100,
                            }
                            for date, d in sorted_days
                        }
                    ).T
                    st.line_chart(trend_df, color=["#2ecc71", "#95a5a6", "#e74c3c"])

with tab3:

    st.caption(
        "Compare how different outlets frame the same entity -- only entities covered by "
        "2 or more outlets can be compared. Uses the same accumulated data as the Live Dashboard."
    )

    outlet_data_path_str = st.text_input("Data file", value=default_path, key="outlet_data_path")
    outlet_data_path = Path(outlet_data_path_str)

    if not outlet_data_path.exists():
        st.info(f"No file found at '{outlet_data_path}' yet.")
    else:

        records = []
        with open(outlet_data_path, encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))

        all_records = [r for r in records if r.get("sentiment") is not None and r.get("feed_url")]

        if not all_records:
            st.warning("No records with both sentiment and a source feed found in this file.")
        else:

            entity_keys = {
                canonical_entity(r).lower()
                for r in all_records
                if canonical_entity(r) is not None
            }
            mapping = derive_alias_mapping(entity_keys)

            # Group by (merged entity key) -> (outlet -> sentiment counts)
            outlet_entity_stats: dict[str, dict[str, dict]] = {}
            display_names: dict[str, str] = {}

            for r in all_records:

                entity = canonical_entity(r)
                if entity is None:
                    continue

                raw_key = entity.lower()
                key = mapping.get(raw_key, raw_key)
                outlet = outlet_display_name(r["feed_url"])

                if key not in outlet_entity_stats:
                    outlet_entity_stats[key] = {}
                    display_names[key] = entity

                if raw_key == key:
                    display_names[key] = entity

                if outlet not in outlet_entity_stats[key]:
                    outlet_entity_stats[key][outlet] = {"Positive": 0, "Neutral": 0, "Negative": 0, "total": 0}

                outlet_entity_stats[key][outlet][r["sentiment"]] += 1
                outlet_entity_stats[key][outlet]["total"] += 1

            min_mentions_per_outlet = st.slider(
                "Minimum mentions per outlet to compare",
                1, 10, 3,
                help="Prevents a framing-gap claim from being based on just 1-2 mentions from one side.",
            )

            comparable = {
                key: outlets for key, outlets in outlet_entity_stats.items()
                if len(outlets) >= 2
                and all(o["total"] >= min_mentions_per_outlet for o in outlets.values())
            }

            if not comparable:
                st.info("No entities are covered by 2 or more outlets yet -- check back as more data accumulates.")
            else:

                ranked_keys = sorted(
                    comparable.keys(),
                    key=lambda k: -sum(o["total"] for o in comparable[k].values()),
                )

                entity_options = [display_names[k] for k in ranked_keys]
                selected_display = st.selectbox("Entity", entity_options)
                selected_key = ranked_keys[entity_options.index(selected_display)]

                outlets_for_entity = sorted(
                    comparable[selected_key].items(), key=lambda kv: -kv[1]["total"]
                )

                if len(outlets_for_entity) == 2:
                    (name_a, stats_a), (name_b, stats_b) = outlets_for_entity
                    neg_a = stats_a["Negative"] / stats_a["total"] * 100
                    neg_b = stats_b["Negative"] / stats_b["total"] * 100
                    gap = neg_a - neg_b

                    if abs(gap) < 5:
                        st.info(f"{name_a} and {name_b} frame {selected_display} similarly -- no meaningful gap in this data.")
                    elif gap > 0:
                        st.info(f"{name_a}'s coverage of {selected_display} is {gap:.0f} percentage points more negative than {name_b}'s.")
                    else:
                        st.info(f"{name_b}'s coverage of {selected_display} is {-gap:.0f} percentage points more negative than {name_a}'s.")

                cols = st.columns(len(outlets_for_entity))

                for i, (outlet_name, stats) in enumerate(outlets_for_entity):
                    with cols[i]:
                        total = stats["total"]
                        pos_pct = stats["Positive"] / total
                        neu_pct = stats["Neutral"] / total
                        neg_pct = stats["Negative"] / total
                        st.metric(outlet_name, f"{total} mention(s)")
                        st.progress(pos_pct, text=f"Positive {pos_pct:.0%}")
                        st.progress(neu_pct, text=f"Neutral {neu_pct:.0%}")
                        st.progress(neg_pct, text=f"Negative {neg_pct:.0%}")
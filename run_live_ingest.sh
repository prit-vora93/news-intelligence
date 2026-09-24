#!/bin/bash
# run_live_ingest.sh
# --------------------
# Wrapper for scheduling live_ingest.py via cron, so real multi-day
# data accumulates in live_results.jsonl without manual intervention.
#
# Setup:
#   1. chmod +x run_live_ingest.sh
#   2. Edit FEEDS below to whatever RSS feeds you want to track.
#   3. Add a cron entry, e.g. to run once every 6 hours:
#        crontab -e
#      then add this line (adjust the path to wherever this script lives):
#        0 */6 * * * /home/prit/news_intelligence/run_live_ingest.sh >> /home/prit/news_intelligence/cron.log 2>&1
#
# Why every 6 hours rather than once a day: RSS feeds publish
# continuously, and --max-articles caps how many NEW articles get
# pulled per run -- running more often means less chance of missing
# articles that get pushed off the feed's recent-items list between
# runs, especially for busier feeds.

set -e

cd "$(dirname "$0")"

source venv/bin/activate

FEEDS=(
    "https://feeds.npr.org/1001/rss.xml"
    "https://feeds.bbci.co.uk/news/rss.xml"
    "https://www.thehindu.com/feeder/default.rss"
)

echo "=== Run started: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

PYTHONPATH=src python src/semantic/live_ingest.py \
    "${FEEDS[@]}" \
    --max-articles 5 \
    --model en_core_web_trf \
    --output live_results.jsonl \
    --quiet

echo "=== Run finished: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

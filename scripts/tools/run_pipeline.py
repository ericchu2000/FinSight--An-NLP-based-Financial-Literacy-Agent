#!/usr/bin/env python3
# scripts/tools/run_pipeline.py
"""
One-command pipeline for FinSight demo.

Usage (from project root):
    python -m scripts.tools.run_pipeline --ticker 601857
"""

import os
import sys
import json
import glob
import logging
from datetime import datetime, timedelta
from typing import Optional

# Make project root importable
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.tools.data_analyzer import analyze_stock_data
from scripts.tools.news_crawler import get_stock_news
from scripts.tools.sentiment_analyzer import SentimentAnalyzer, save_analysis_result
from scripts.tools import insight_agent
import argparse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("FIN_SIGHT_PIPELINE")

# ✅ ensure cache is always inside scripts/tools/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))     # scripts/tools/
CACHE_BASE = os.path.join(BASE_DIR, "cache")              # scripts/tools/cache/


def find_latest_market_csv(symbol: str):
    """同目录找到最新市场csv文件"""
    pattern = os.path.join(CACHE_BASE, "stock_price_data", symbol, f"{symbol}_analysis_*.csv")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def find_latest_news_json(symbol: str, date: Optional[str] = None):
    """找到最新新闻数据的json"""
    if date:
        candidate = os.path.join(CACHE_BASE, "news", "stock_news", symbol, f"{symbol}_news_{date}.json")
        return candidate if os.path.exists(candidate) else None

    pattern = os.path.join(CACHE_BASE, "news", "stock_news", symbol, f"{symbol}_news_*.json")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def run_pipeline(symbol: str, days: int = 365, max_news: int = 30):
    log.info(f"🚀 Running FinSight pipeline for ticker: {symbol}")

    # -------------------- 1. Price & Indicators --------------------
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    log.info(f"▶ Step 1: generating market CSV ({start_date} → {end_date})")

    analyze_stock_data(symbol, start_date, end_date)

    market_csv = find_latest_market_csv(symbol)
    if not market_csv:
        log.error("❌ Market CSV not found after analyze_stock_data()")
        raise SystemExit(1)
    log.info(f"✅ Market CSV located: {market_csv}")

    # -------------------- 2. News Crawling --------------------
    log.info(f"▶ Step 2: crawling news (max={max_news})")
    news_list = get_stock_news(symbol, max_news=max_news, date=None)

    news_json = find_latest_news_json(symbol)
    if not news_json:
        log.warning("⚠ news_crawler did not produce json; writing fallback")
        news_dir = os.path.join(CACHE_BASE, "news", "stock_news", symbol)
        os.makedirs(news_dir, exist_ok=True)

        fname = f"{symbol}_news_{datetime.now().strftime('%Y-%m-%d')}.json"
        news_json = os.path.join(news_dir, fname)

        with open(news_json, "w", encoding="utf-8") as f:
            json.dump({"news": news_list}, f, ensure_ascii=False, indent=2)

    log.info(f"✅ News JSON saved: {news_json}")

    # -------------------- 3. Sentiment Analysis --------------------
    log.info("▶ Step 3: running sentiment analysis")

    with open(news_json, "r", encoding="utf-8") as f:
        raw_news = json.load(f)

    news_items = raw_news.get("news") if isinstance(raw_news, dict) else raw_news
    if not news_items:
        log.error("❌ No news items available for sentiment analysis")
        raise SystemExit(1)

    analyzer = SentimentAnalyzer(use_snownlp=False, add_finance_words=True)
    sentiment_result = analyzer.analyze_news_list(news_items)

    sentiment_result["stock_code"] = symbol
    sentiment_result["data_source"] = news_json

    sentiment_dir = os.path.join(CACHE_BASE, "sentiment_analysis")
    os.makedirs(sentiment_dir, exist_ok=True)

    sentiment_fname = f"{symbol}_news_{datetime.now().strftime('%Y-%m-%d')}_sentiment_analyses.json"
    sentiment_path = os.path.join(sentiment_dir, sentiment_fname)

    save_analysis_result(sentiment_result, sentiment_path)
    log.info(f"✅ Sentiment JSON saved: {sentiment_path}")

    # -------------------- 4. Insight Agent --------------------
    log.info("▶ Step 4: running insight agent")

    sentiment_obj = insight_agent.load_sentiment_json(sentiment_path)
    market_obj = insight_agent.load_market_csv(market_csv)

    insight_output = insight_agent.run_insight(sentiment_obj, market_obj)
    insight_agent.save_to_cache(insight_output)

    log.info("🎯 Pipeline complete — insight report generated.")


def main():
    parser = argparse.ArgumentParser(description="FinSight pipeline: data → news → sentiment → insight")
    parser.add_argument("--ticker", "-t", required=True, help="stock ticker (e.g., 601857)")
    parser.add_argument("--days", "-d", type=int, default=365, help="days of market data to fetch")
    parser.add_argument("--max-news", type=int, default=30, help="max news items to collect")
    args = parser.parse_args()

    run_pipeline(args.ticker, days=args.days, max_news=args.max_news)


if __name__ == "__main__":
    main()

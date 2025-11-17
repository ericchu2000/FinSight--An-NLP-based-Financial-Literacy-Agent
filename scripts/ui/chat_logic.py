#!/usr/bin/env python3
"""
Chat logic for FinSight Educational Bot.

This module contains:
- helper functions (ticker extraction, cache file lookup, summaries)
- simple session state (CURRENT_TICKER)
- the core chat_fn used by Gradio
"""

import os
import sys
import glob
import re
import json
import logging
from typing import List, Tuple, Optional

import pandas as pd

# ----------------- Import project modules -----------------

# Make project root importable
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.tools.run_pipeline import run_pipeline, CACHE_BASE  # existing pipeline

log = logging.getLogger("FIN_SIGHT_GRADIO")
logging.basicConfig(level=logging.INFO)

# ----------------- Simple session-like state (per process) -----------------
# For a single-user / demo setting this is fine. Later you can move to a proper state object.
CURRENT_TICKER: Optional[str] = None

# ----------------- Description used in Gradio UI -----------------

DESCRIPTION_MD = """
# 🎓 FinSight Educational Bot (Gradio Demo)

Type a message containing a **6-digit stock code** (e.g. `600028`, `601857`, `301201`)
to run the full pipeline for that stock.

After the first analysis, you can ask follow-up questions, for example:

- `Show me the sentiment details`
- `What indicators did you use?`
- `Explain MACD in this context`
"""

# ----------------- Helpers -----------------

def extract_ticker_from_message(message: str) -> Optional[str]:
    """
    Very simple heuristic: find a 6-digit number as ticker (e.g., 600028, 601857, 301201).
    You can replace this later with a better parser.
    """
    if not message:
        return None
    match = re.search(r"\b\d{6}\b", message)
    return match.group(0) if match else None


def find_latest_education_report(ticker: str) -> Optional[str]:
    """
    Find the latest education markdown file for a given ticker.
    Looks under CACHE_BASE/education_reports/<ticker>_*.md
    """
    edu_dir = os.path.join(CACHE_BASE, "education_reports")
    pattern = os.path.join(edu_dir, f"{ticker}_*.md")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def find_latest_sentiment_json(ticker: str) -> Optional[str]:
    """
    Find the latest sentiment analysis JSON for a given ticker.
    Looks under CACHE_BASE/sentiment_analysis/<ticker>_news_*_sentiment_analyses.json
    """
    sent_dir = os.path.join(CACHE_BASE, "sentiment_analysis")
    pattern = os.path.join(sent_dir, f"{ticker}_news_*_sentiment_analyses.json")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def find_latest_market_csv(ticker: str) -> Optional[str]:
    """
    Find the latest market CSV for a given ticker.
    Looks under CACHE_BASE/stock_price_data/<ticker>/<ticker>_analysis_*.csv
    """
    mkt_dir = os.path.join(CACHE_BASE, "stock_price_data", ticker)
    pattern = os.path.join(mkt_dir, f"{ticker}_analysis_*.csv")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def summarize_sentiment(path: str) -> str:
    """
    Load a sentiment JSON file and return a human-readable summary string.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return f"⚠ I found a sentiment file at `{path}`, but could not read it: `{e}`"

    overall = data.get("overall_sentiment", {})
    label = overall.get("label", "unknown")
    score = overall.get("score", "N/A")

    dist = data.get("sentiment_distribution", {})
    pos = dist.get("positive", 0)
    neu = dist.get("neutral", 0)
    neg = dist.get("negative", 0)

    detailed = data.get("detailed_analyses") or data.get("news") or []
    samples = detailed[:3] if isinstance(detailed, list) else []

    lines = [
        "### 📰 Sentiment Overview",
        "",
        f"- Overall label: **{label}**",
        f"- Score: **{score}**",
        f"- Distribution: {pos} positive · {neu} neutral · {neg} negative",
        "",
    ]

    if samples:
        lines.append("**Sample news items:**")
        for item in samples:
            title = item.get("title") if isinstance(item, dict) else str(item)
            s = item.get("final_sentiment", {}).get("label") if isinstance(item, dict) else None
            if s:
                lines.append(f"- ({s}) {title}")
            else:
                lines.append(f"- {title}")
    else:
        lines.append("_No detailed news items available in this sentiment file._")

    lines.append("")
    lines.append(f"_Source: `{os.path.basename(path)}`_")
    return "\n".join(lines)


def summarize_market_snapshot(path: str) -> str:
    """
    Load the latest row from a market CSV and return a human-readable summary.
    """
    try:
        df = pd.read_csv(path)
        if df.empty:
            return f"⚠ Market CSV at `{path}` is empty."
        last = df.tail(1).iloc[0]
    except Exception as e:
        return f"⚠ I found a market CSV at `{path}`, but could not read it: `{e}`"

    def get(col, default="N/A"):
        return last[col] if col in df.columns else default

    close = get("close")
    ma5 = get("ma5")
    ma10 = get("ma10")
    ma20 = get("ma20")
    rsi = get("rsi")
    macd = get("macd")
    signal = get("singal_line") if "singal_line" in df.columns else get("signal_line")
    hist = get("macd_hist")

    lines = [
        "### 📈 Technical Snapshot (latest bar)",
        "",
        f"- Close: **{close}**",
        f"- MA5: **{ma5}**, MA10: **{ma10}**, MA20: **{ma20}**",
        f"- RSI: **{rsi}**",
        f"- MACD: **{macd}**, Signal: **{signal}**, Hist: **{hist}**",
        "",
    ]

    # Quick MA interpretation
    try:
        ma_comment = ""
        if all(isinstance(x, (int, float)) for x in [close, ma5, ma10, ma20]):
            if close > ma5 and close > ma10 and close > ma20:
                ma_comment = "Price is above all key moving averages → bullish trend bias."
            elif close < ma5 and close < ma10 and close < ma20:
                ma_comment = "Price is below all key moving averages → bearish trend bias."
            else:
                ma_comment = "Price is mixed relative to moving averages → possible consolidation."
            lines.append(f"- **MA interpretation:** {ma_comment}")
    except Exception:
        pass

    lines.append("")
    lines.append(f"_Source: `{os.path.basename(path)}`_")
    return "\n".join(lines)


# ----------------- Core chat logic -----------------

def chat_fn(message: str, history: List[Tuple[str, str]]) -> str:
    """
    Gradio ChatInterface handler.

    - message: latest user message
    - history: list of (user, bot) tuples (previous turns) [managed by Gradio]

    Behavior:
    - If the user mentions a new ticker → run full pipeline and show education.
    - Else → interpret the question as a follow-up about CURRENT_TICKER.
    """
    global CURRENT_TICKER

    msg = message or ""
    msg_lower = msg.lower()

    # Detect ticker in message
    ticker = extract_ticker_from_message(msg)

    # ----- CASE 1: New ticker or change of ticker → run full pipeline -----
    if ticker and ticker != CURRENT_TICKER:
        CURRENT_TICKER = ticker
        log.info(f"[NEW TICKER] Starting full pipeline for: {ticker}")

        intro = (
            f"Got it! I’ll analyze **{ticker}** using the full pipeline:\n"
            "- market data + technical indicators\n"
            "- news sentiment\n"
            "- LLM insight\n"
            "- educational explanation\n\n"
            "_Running analysis..._\n"
        )

        try:
            run_pipeline(ticker)
        except Exception as e:
            err_msg = (
                f"{intro}\n"
                f"❌ Sorry, something went wrong while running the pipeline for **{ticker}**:\n"
                f"`{e}`\n\n"
                "Please try another ticker or check the backend logs."
            )
            log.exception(f"Pipeline error for {ticker}: {e}")
            return err_msg

        edu_path = find_latest_education_report(ticker)
        if not edu_path:
            msg = (
                f"{intro}\n"
                f"✅ The analysis for **{ticker}** finished, but I couldn’t find an education report.\n\n"
                "Please check that the education agent is integrated and writing to:\n"
                f"`{os.path.join(CACHE_BASE, 'education_reports')}`"
            )
            return msg

        try:
            with open(edu_path, "r", encoding="utf-8") as f:
                edu_markdown = f.read()
        except Exception as e:
            msg = (
                f"{intro}\n"
                f"✅ I found an education report at `{edu_path}` but couldn’t read it:\n"
                f"`{e}`"
            )
            return msg

        reply = (
            f"{intro}\n"
            f"✅ Here is the educational explanation for **{ticker}** "
            f"(from `{os.path.basename(edu_path)}`):\n\n"
            f"{edu_markdown}\n\n"
            "---\n"
            "You can now ask follow-up questions like:\n"
            "- `Show me the sentiment details`\n"
            "- `What indicators did you use?`\n"
            "- `Explain MACD in this context`\n"
        )
        return reply

    # ----- CASE 2: No ticker in message or same ticker → follow-up about CURRENT_TICKER -----
    if CURRENT_TICKER is None:
        return (
            "I don’t have any stock analysis in this session yet.\n\n"
            "Please start by asking about a ticker, e.g.:\n"
            "`What do you think of 600028?`"
        )

    # Very simple rule-based intent detection

    # Sentiment drilldown
    if any(word in msg_lower for word in ["sentiment", "news", "headline"]):
        sent_path = find_latest_sentiment_json(CURRENT_TICKER)
        if not sent_path:
            return (
                f"🔍 I looked for sentiment analysis for **{CURRENT_TICKER}** but "
                "couldn’t find a cached file.\n\n"
                "Try running a full analysis again by asking something like:\n"
                f"`What do you think of {CURRENT_TICKER}?`"
            )
        return summarize_sentiment(sent_path)

    # Technical drilldown
    if any(word in msg_lower for word in ["indicator", "technical", "ma", "rsi", "macd", "bollinger"]):
        mkt_path = find_latest_market_csv(CURRENT_TICKER)
        if not mkt_path:
            return (
                f"🔍 I looked for technical indicator data for **{CURRENT_TICKER}** but "
                "couldn’t find a cached CSV.\n\n"
                "Try running a full analysis again by asking something like:\n"
                f"`What do you think of {CURRENT_TICKER}?`"
            )
        return summarize_market_snapshot(mkt_path)

    # Re-show education report
    if any(word in msg_lower for word in ["explain", "education", "learning", "teach"]):
        edu_path = find_latest_education_report(CURRENT_TICKER)
        if not edu_path:
            return (
                f"🔍 I couldn’t find an education report for **{CURRENT_TICKER}**.\n\n"
                "Try running a full analysis again by asking something like:\n"
                f"`What do you think of {CURRENT_TICKER}?`"
            )
        try:
            with open(edu_path, "r", encoding="utf-8") as f:
                edu_markdown = f.read()
        except Exception as e:
            return (
                f"⚠ I found an education report at `{edu_path}`, but couldn’t read it:\n"
                f"`{e}`"
            )

        return (
            f"Here is the latest educational explanation for **{CURRENT_TICKER}** "
            f"(from `{os.path.basename(edu_path)}`):\n\n"
            f"{edu_markdown}"
        )

    # Fallback: Unknown follow-up
    return (
        f"I’m currently focused on **{CURRENT_TICKER}**.\n\n"
        "You can ask follow-up questions like:\n"
        "- `Show me the sentiment details`\n"
        "- `What indicators did you use?`\n"
        "- `Explain MACD in this context`\n"
        "Or start a new analysis by asking about another 6-digit ticker."
    )

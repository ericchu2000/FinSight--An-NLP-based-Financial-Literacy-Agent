#!/usr/bin/env python3
"""
Chat logic for FinSight Educational Bot.

Supports:
- New ticker questions → run full pipeline + education report
    e.g. "I think 600028 will go up, what do you think?"
- Follow-up questions on the same ticker:
    - "Show me the sentiment details"
    - "What indicators did you use?"
    - "What does MACD mean in this context?"
- General finance education questions via OpenAI:
    - "What is MACD?"
    - "What does diversification mean?"
    - "What is the difference between investing and trading?"

The goal is:
- Ground answers in your cached data when relevant
- Let OpenAI phrase explanations in natural, tutor-style English
- Pick up the user's own prediction ("up/down/flat") when they mention it,
  and inject it into the education text for demo purposes.
"""

import os
import sys
import glob
import re
import json
import logging
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
import requests

# ----------------- Import project modules -----------------

# Make project root importable
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.tools.run_pipeline import run_pipeline, CACHE_BASE  # existing pipeline

# LLM config (OpenAI)
try:
    from scripts.tools.llm_config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL
except ImportError:
    # Fallback to template if user hasn't created llm_config.py yet
    from scripts.tools.llm_config_template import (
        OPENAI_API_KEY,
        OPENAI_MODEL,
        OPENAI_BASE_URL,
    )

log = logging.getLogger("FIN_SIGHT_GRADIO")
logging.basicConfig(level=logging.INFO)

# ----------------- Simple session-like state (per process) -----------------
# For a single-user / demo setting this is fine. Later you can move to a proper state object.
CURRENT_TICKER: Optional[str] = None
USER_PREDICTIONS: Dict[str, str] = {}  # ticker -> "up" | "down" | "flat"

# ----------------- Description used in Gradio UI -----------------

DESCRIPTION_MD = """
# 🎓 FinSight Educational Bot (Gradio Demo)

Type a message containing a **6-digit stock code** (e.g. `600028`, `601857`, `301201`)
to run the full pipeline for that stock.

You can also tell me your own view, for example:

- `I think 600028 will go up, what do you think?`
- `I feel 300014 might drop, can you analyze it?`

"""

# ----------------- Finance keyword hints (for fallback tutor mode) -----------------

FINANCE_KEYWORDS = [
    "stock",
    "stocks",
    "market",
    "etf",
    "index",
    "indexes",
    "indices",
    "bond",
    "bonds",
    "portfolio",
    "diversification",
    "risk",
    "volatility",
    "pe ratio",
    "p/e",
    "eps",
    "dividend",
    "valuation",
    "interest rate",
    "inflation",
    "macd",
    "rsi",
    "moving average",
    "ma",
    "bollinger",
    "stop loss",
    "trend",
    "technical indicator",
]

INDICATOR_KEYWORDS = ["macd", "rsi", "ma", "moving average", "bollinger", "indicator"]


# ----------------- LLM helpers -----------------

def _has_real_openai_key() -> bool:
    """
    Heuristic to detect if OPENAI_API_KEY has been set to a real value.
    """
    if not OPENAI_API_KEY:
        return False
    placeholder_fragments = ["paste", "your-api-key", "your_api_key", "example"]
    key_lower = OPENAI_API_KEY.lower()
    return not any(fragment in key_lower for fragment in placeholder_fragments)


def call_openai_chat(
    prompt: str,
    system: str = (
        "You are a friendly financial education tutor. Explain things clearly, "
        "avoid giving direct investment advice, and emphasize that content is educational only."
    ),
    temperature: float = 0.3,
) -> str:
    """
    Minimal helper to call OpenAI Chat Completions API.

    Uses OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL from llm_config.
    Returns the assistant's content text, or a fallback error message.
    """
    if not _has_real_openai_key():
        return (
            "⚠ I’m configured without a real OpenAI API key, so I can’t generate a "
            "natural-language explanation right now. Please set OPENAI_API_KEY in "
            "`scripts/tools/llm_config.py`."
        )

    url = OPENAI_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    body: Dict[str, Any] = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content.strip()
    except Exception as e:
        log.error(f"OpenAI chat call failed: {e}")
        return (
            "⚠ I tried to generate a natural-language explanation using the LLM, "
            f"but there was an error: `{e}`"
        )


# ----------------- Cache file helpers -----------------

def extract_ticker_from_message(message: str) -> Optional[str]:
    """
    Very simple heuristic: find a 6-digit number as ticker (e.g., 600028, 601857, 301201).
    You can replace this later with a better parser.
    """
    if not message:
        return None
    match = re.search(r"\b\d{6}\b", message)
    return match.group(0) if match else None


def parse_user_prediction(message: str) -> Optional[str]:
    """
    Try to infer the user's prediction from the message.

    Examples we want to catch:
    - "I think 600028 will go up"
    - "I feel 300014 might drop"
    - "I think it will be flat / sideways"
    Returns: "up", "down", "flat", or None.
    """
    if not message:
        return None

    msg_lower = message.lower()

    up_words = [
        "go up",
        "will go up",
        "rise",
        "rising",
        "going up",
        "uptrend",
        "bullish",
        "涨",
        "上涨",
    ]
    down_words = [
        "go down",
        "will go down",
        "drop",
        "fall",
        "falling",
        "going down",
        "downtrend",
        "bearish",
        "跌",
        "下跌",
    ]
    flat_words = [
        "flat",
        "sideways",
        "range bound",
        "range-bound",
        "consolidating",
        "震荡",
    ]

    for w in up_words:
        if w in msg_lower:
            return "up"
    for w in down_words:
        if w in msg_lower:
            return "down"
    for w in flat_words:
        if w in msg_lower:
            return "flat"

    return None


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


def find_latest_insight_json(ticker: str) -> Optional[str]:
    """
    Find the latest insight report JSON for a given ticker.
    Looks under CACHE_BASE/insight_reports/<ticker>_*.json
    """
    ins_dir = os.path.join(CACHE_BASE, "insight_reports")
    pattern = os.path.join(ins_dir, f"{ticker}_*.json")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


# ----------------- Local summarizers (and context builders for LLM) -----------------

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


def load_market_snapshot(path: str) -> Optional[Dict[str, Any]]:
    """
    Load the latest row from a market CSV and return a dict of key values.
    Used to build context for indicator explanations.
    """
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        last = df.tail(1).iloc[0]
    except Exception as e:
        log.error(f"Error reading market CSV {path}: {e}")
        return None

    def get(col):
        return last[col] if col in df.columns else None

    snapshot: Dict[str, Any] = {
        "date": get("date"),
        "close": get("close"),
        "ma5": get("ma5"),
        "ma10": get("ma10"),
        "ma20": get("ma20"),
        "rsi": get("rsi"),
        "macd": get("macd"),
        "signal_line": get("singal_line") if "singal_line" in df.columns else get("signal_line"),
        "macd_hist": get("macd_hist"),
    }
    return snapshot


def summarize_market_snapshot(path: str) -> str:
    """
    Human-readable technical summary (numbers + quick MA interpretation).
    Used for requests like "show me the indicators".
    """
    snapshot = load_market_snapshot(path)
    if snapshot is None:
        return f"⚠ I found a market CSV at `{path}`, but could not read it or it was empty."

    close = snapshot.get("close", "N/A")
    ma5 = snapshot.get("ma5", "N/A")
    ma10 = snapshot.get("ma10", "N/A")
    ma20 = snapshot.get("ma20", "N/A")
    rsi = snapshot.get("rsi", "N/A")
    macd = snapshot.get("macd", "N/A")
    signal = snapshot.get("signal_line", "N/A")
    hist = snapshot.get("macd_hist", "N/A")

    lines = [
        "### 📈 Technical Snapshot (latest bar)",
        "",
        f"- Close: **{close}**",
        f"- MA5: **{ma5}**, MA10: **{ma10}**, MA20: **{ma20}**",
        f"- RSI: **{rsi}**",
        f"- MACD: **{macd}**, Signal: **{signal}**, Hist: **{hist}**",
        "",
    ]

    # Quick MA interpretation (best-effort)
    try:
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


def build_indicator_context_for_llm(
    ticker: str,
    market_csv_path: Optional[str],
    insight_json_path: Optional[str],
) -> str:
    """
    Build a compact text context describing the indicator situation for a ticker
    using cached market CSV + insight report.

    This is passed into OpenAI when user asks things like:
    - "What does MACD mean in this context for 600028?"
    - "How should I interpret RSI here?"
    """
    lines: List[str] = []
    lines.append(f"Ticker: {ticker}")

    # From insight JSON (direction, confidence, factors)
    if insight_json_path and os.path.exists(insight_json_path):
        try:
            with open(insight_json_path, "r", encoding="utf-8") as f:
                ins = json.load(f)
            model_pred = ins.get("model_prediction", {})
            direction = model_pred.get("direction", "unknown")
            confidence = model_pred.get("confidence", "N/A")
            factors = ins.get("factors", [])
            indicators_info = ins.get("indicators", [])

            lines.append(f"Model direction: {direction}")
            lines.append(f"Model confidence: {confidence}")

            if factors and isinstance(factors, list):
                lines.append("Factors considered:")
                for fct in factors[:5]:
                    name = fct.get("name")
                    impact = fct.get("impact")
                    value = fct.get("value")
                    lines.append(f"- {name}: {value} (impact: {impact})")

            if indicators_info and isinstance(indicators_info, list):
                lines.append("Indicator readings (qualitative):")
                for ind in indicators_info[:5]:
                    name = ind.get("indicator")
                    reading = ind.get("reading")
                    lines.append(f"- {name}: {reading}")
        except Exception as e:
            log.error(f"Failed to load insight JSON {insight_json_path}: {e}")

    # From market CSV (exact numbers)
    if market_csv_path and os.path.exists(market_csv_path):
        snap = load_market_snapshot(market_csv_path)
        if snap:
            lines.append("Latest market snapshot (numbers):")
            for k in ["date", "close", "ma5", "ma10", "ma20", "rsi", "macd", "signal_line", "macd_hist"]:
                v = snap.get(k)
                lines.append(f"- {k}: {v}")
    else:
        lines.append("No market CSV snapshot available.")

    return "\n".join(lines)


def postprocess_education_markdown(
    edu_markdown: str,
    ticker: str,
    user_prediction: Optional[str],
) -> str:
    """
    For demo purposes, patch the education markdown:

    - If it contains "Your prediction is not specified", replace that with the actual user prediction.
    - Remove any mention of "unknown result" from the AI.
    - Remove any sentence about "actual outcome" (we don't need it for the demo).
    - Add a short note about the model prediction direction (from insight JSON),
      ignoring the top-level `result` field in JSON completely.
    """
    result = edu_markdown

    # Load latest insight JSON to get direction (and optionally confidence)
    direction = "unknown"
    confidence = None
    insight_path = find_latest_insight_json(ticker)
    if insight_path and os.path.exists(insight_path):
        try:
            with open(insight_path, "r", encoding="utf-8") as f:
                ins = json.load(f)
            model_pred = ins.get("model_prediction", {})
            direction = model_pred.get("direction", "unknown")
            confidence = model_pred.get("confidence")
        except Exception as e:
            log.error(f"Failed to load insight JSON for postprocess: {e}")

    # 1) Patch "Your prediction is not specified"
    if user_prediction:
        user_line = f"Your prediction: **{user_prediction}**."
    else:
        user_line = "Your prediction: (not provided in your question)."

    if "Your prediction is not specified" in result:
        result = result.replace("Your prediction is not specified", user_line)

    # 2) Remove any explicit mention of "actual outcome"
    if "The actual outcome isn't provided" in result:
        result = result.replace("The actual outcome isn't provided", "")

    lines = result.splitlines()
    filtered_lines: List[str] = []
    for line in lines:
        lower = line.lower()
        # Drop lines that mention the "actual outcome"
        if "actual outcome" in lower:
            continue
        # Drop lines that mention an "unknown result" from the AI prediction
        if "unknown" in lower and "result" in lower and "prediction" in lower:
            continue
        filtered_lines.append(line)
    result = "\n".join(filtered_lines)

    # 3) Append a short note about the model prediction direction,
    #    completely ignoring the JSON-level `result` field.
    model_note_parts: List[str] = []
    if direction and direction != "unknown":
        if confidence is not None:
            model_note_parts.append(
                f"Model prediction direction: **{direction}** (confidence ~{confidence:.2f})."
            )
        else:
            model_note_parts.append(
                f"Model prediction direction: **{direction}**."
            )

    extra_lines: List[str] = []
    if user_prediction or model_note_parts:
        extra_lines.append("")
        extra_lines.append("---")
        extra_lines.append("_Demo note:_")
        extra_lines.append(f"- {user_line}")
        if model_note_parts:
            for ln in model_note_parts:
                extra_lines.append(f"- {ln}")

    if extra_lines:
        result = result.rstrip() + "\n" + "\n".join(extra_lines)

    return result


# ----------------- Intent helpers -----------------

def is_finance_related(msg_lower: str) -> bool:
    return any(kw in msg_lower for kw in FINANCE_KEYWORDS)


def is_indicator_question(msg_lower: str) -> bool:
    return any(kw in msg_lower for kw in INDICATOR_KEYWORDS)


def is_explainer_style_question(msg_lower: str) -> bool:
    # e.g. "what does macd mean here", "explain rsi", "how should I interpret ..."
    return any(word in msg_lower for word in ["what is", "what does", "mean", "explain", "interpret", "how should i"])


# ----------------- Core chat logic -----------------

def chat_fn(message: str, history: List[Tuple[str, str]]) -> str:
    """
    Gradio ChatInterface handler.

    - message: latest user message
    - history: list of (user, bot) tuples (previous turns) [managed by Gradio]

    Behavior:
    - If the user mentions a new ticker → run full pipeline and show education.
      Also try to pick up their own prediction (up/down/flat).
    - Else → interpret the question as:
        * follow-up about CURRENT_TICKER (sentiment / indicators / education)
        * OR general finance education question (via OpenAI).
    """
    global CURRENT_TICKER, USER_PREDICTIONS

    msg = message or ""
    msg_lower = msg.lower().strip()

    # Detect ticker in message
    ticker = extract_ticker_from_message(msg)
    # Detect user prediction in message (if any)
    user_pred = parse_user_prediction(msg)

    # ----- CASE 1: New ticker or change of ticker → run full pipeline -----
    if ticker and ticker != CURRENT_TICKER:
        CURRENT_TICKER = ticker

        if user_pred:
            USER_PREDICTIONS[CURRENT_TICKER] = user_pred
        else:
            USER_PREDICTIONS.pop(CURRENT_TICKER, None)

        log.info(f"[NEW TICKER] Starting full pipeline for: {ticker}, user_pred={user_pred}")

        intro_lines = [
            f"Got it! I’ll analyze **{ticker}** using the full pipeline:",
            "- market data + technical indicators",
            "- news sentiment",
            "- LLM insight",
            "- educational explanation",
            "",
        ]
        if user_pred:
            intro_lines.append(
                f"_I’ll also record your view that the stock may go **{user_pred}** for this demo._"
            )
        intro_lines.append("_Running analysis..._")
        intro = "\n".join(intro_lines)

        try:
            run_pipeline(ticker)
        except SystemExit:
            log.exception(f"Pipeline exited for {ticker} due to missing data.")
            CURRENT_TICKER = None
            USER_PREDICTIONS.pop(ticker, None)
            err_msg = (
                f"{intro}\n\n"
                f"❌ I tried to analyze **{ticker}**, but no usable market data was available, "
                "so I couldn’t generate indicators or run the full pipeline.\n\n"
                "This often happens if the ticker is not supported by the data source, "
                "is very new, delisted, or has no price history in the selected date range.\n\n"
                "Please try another 6-digit stock code."
            )
            return err_msg
        except Exception as e:
            err_msg = (
                f"{intro}\n\n"
                f"❌ Sorry, something went wrong while running the pipeline for **{ticker}**:\n"
                f"`{e}`\n\n"
                "Please try another ticker or check the backend logs."
            )
            log.exception(f"Pipeline error for {ticker}: {e}")
            return err_msg

        edu_path = find_latest_education_report(ticker)
        if not edu_path:
            msg_out = (
                f"{intro}\n\n"
                f"✅ The analysis for **{ticker}** finished, but I couldn’t find an education report.\n\n"
                "Please check that the education agent is integrated and writing to:\n"
                f"`{os.path.join(CACHE_BASE, 'education_reports')}`"
            )
            return msg_out

        try:
            with open(edu_path, "r", encoding="utf-8") as f:
                edu_markdown_raw = f.read()
        except Exception as e:
            msg_out = (
                f"{intro}\n\n"
                f"✅ I found an education report at `{edu_path}` but couldn’t read it:\n"
                f"`{e}`"
            )
            return msg_out

        # ----- Build main trend statement (model + user) -----
        model_direction = "unknown"
        model_conf = None
        insight_path = find_latest_insight_json(ticker)
        if insight_path and os.path.exists(insight_path):
            try:
                with open(insight_path, "r", encoding="utf-8") as f:
                    ins = json.load(f)
                mp = ins.get("model_prediction", {})
                model_direction = mp.get("direction", "unknown")
                model_conf = mp.get("confidence")
            except Exception as e:
                log.error(f"Failed to load insight JSON for main trend statement: {e}")

        if model_direction and model_direction != "unknown":
            if model_conf is not None:
                model_line = (
                    f"- **Model prediction:** trend is **{model_direction}** "
                    f"(confidence ~{model_conf:.2f})."
                )
            else:
                model_line = f"- **Model prediction:** trend is **{model_direction}**."
        else:
            model_line = "- **Model prediction:** trend signal is currently inconclusive."

        user_view = USER_PREDICTIONS.get(CURRENT_TICKER)
        if user_view:
            user_line = f"- **Your view:** you think it will go **{user_view}**."
        else:
            user_line = "- **Your view:** not specified in your question."

        main_statement = (
            f"### Trend summary for {ticker}\n"
            f"{model_line}\n"
            f"{user_line}\n"
        )

        # Patch the education markdown with user prediction + model direction note
        final_edu_md = postprocess_education_markdown(
            edu_markdown_raw,
            ticker=CURRENT_TICKER,
            user_prediction=USER_PREDICTIONS.get(CURRENT_TICKER),
        )

        reply = (
            f"{intro}\n\n"
            f"{main_statement}\n"
            f"✅ Here is the educational explanation for **{ticker}** "
            f"(from `{os.path.basename(edu_path)}`):\n\n"
            f"{final_edu_md}\n\n"
            "---\n"
            "You can now ask follow-up questions like:\n"
            "- `Show me the sentiment details`\n"
            "- `What indicators did you use?`\n"
            "- `What does MACD mean in this context?`\n"
        )
        return reply

    # ----- CASE 2: No ticker in message or same ticker → follow-up or general question -----

    if CURRENT_TICKER is None:
        if is_finance_related(msg_lower):
            prompt = (
                "The user asked the following finance-related question (no specific data context):\n"
                f"{message}\n\n"
                "Please answer as a financial education tutor for a beginner. "
                "Explain clearly, avoid giving direct buy/sell advice, and add a one-line reminder "
                "that this is educational only."
            )
            return call_openai_chat(prompt)
        return (
            "I don’t have any stock analysis in this session yet.\n\n"
            "You can start by asking about a ticker, for example:\n"
            "- `What do you think of 600028?`\n"
            "- `Can you analyze 601857 for me?`\n"
            "Or ask a general finance question like `What is diversification?`"
        )

    # From here on, CURRENT_TICKER is set, so we interpret as follow-ups or generic finance.

    if user_pred:
        USER_PREDICTIONS[CURRENT_TICKER] = user_pred

    # --- Sentiment drilldown ---
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

    # --- Indicator / technical questions ---
    if is_indicator_question(msg_lower):
        mkt_path = find_latest_market_csv(CURRENT_TICKER)
        if not mkt_path:
            return (
                f"🔍 I looked for technical indicator data for **{CURRENT_TICKER}** but "
                "couldn’t find a cached CSV.\n\n"
                "Try running a full analysis again by asking something like:\n"
                f"`What do you think of {CURRENT_TICKER}?`"
            )

        if is_explainer_style_question(msg_lower):
            insight_path = find_latest_insight_json(CURRENT_TICKER)
            context = build_indicator_context_for_llm(
                ticker=CURRENT_TICKER,
                market_csv_path=mkt_path,
                insight_json_path=insight_path,
            )

            prompt = (
                f"The user asked: {message}\n\n"
                "Here is the current analysis snapshot for this stock:\n"
                "----------------------------------------\n"
                f"{context}\n"
                "----------------------------------------\n\n"
                "Please answer ONLY their question, explaining the relevant indicators "
                "in this specific context.\n\n"
                "Requirements:\n"
                "- Start with 1–2 sentences explaining the indicator(s) at a high level.\n"
                "- Then explain what the current values imply for this stock's trend or momentum.\n"
                "- Ground your explanation strictly in the numbers provided; do not invent data.\n"
                "- Do NOT give explicit investment advice (no 'you should buy/sell').\n"
                "- End with a short reminder that this is for learning, not a recommendation."
            )
            return call_openai_chat(prompt)

        return summarize_market_snapshot(mkt_path)

    # --- Re-show education report ---
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
                edu_markdown_raw = f.read()
        except Exception as e:
            return (
                f"⚠ I found an education report at `{edu_path}`, but couldn’t read it:\n"
                f"`{e}`"
            )

        final_edu_md = postprocess_education_markdown(
            edu_markdown_raw,
            ticker=CURRENT_TICKER,
            user_prediction=USER_PREDICTIONS.get(CURRENT_TICKER),
        )

        return (
            f"Here is the latest educational explanation for **{CURRENT_TICKER}** "
            f"(from `{os.path.basename(edu_path)}`):\n\n"
            f"{final_edu_md}"
        )

    # --- General finance tutor fallback ---
    if is_finance_related(msg_lower):
        prompt = (
            "The user asked the following finance-related question:\n"
            f"{message}\n\n"
            f"The current conversation is focused on ticker: {CURRENT_TICKER}.\n"
            "If it makes sense, you may use this ticker as an example, but you don't have "
            "access to live market data beyond what the user or system has given you.\n\n"
            "Please answer as a financial education tutor for a beginner. "
            "Explain clearly, use simple examples, avoid giving direct buy/sell advice, "
            "and add a one-line reminder that this is educational only."
        )
        return call_openai_chat(prompt)

    # --- Fallback: not clearly finance-related ---
    return (
        f"I’m currently focused on **{CURRENT_TICKER}** as your financial tutor.\n\n"
        "You can ask follow-up questions like:\n"
        "- `Show me the sentiment details`\n"
        "- `What indicators did you use?`\n"
        "- `What does MACD mean in this context?`\n"
        "Or ask a general finance question like:\n"
        "- `What is diversification?`\n"
        "- `What is the difference between investing and trading?`"
    )

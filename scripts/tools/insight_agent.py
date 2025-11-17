# insight_agent.py
"""
FinSight — Insight agent CLI

Usage (from project root):
  python -m scripts.tools.insight_agent --ticker 601857
  python -m scripts.tools.insight_agent --sentiment-path cache/sentiment_analysis/601857_news_2025-11-09_sentiment_analyses.json --market-path cache/stock_price_data/601857/601857_analysis_20251109.csv
"""

import os
import json
import uuid
import logging
import glob
import argparse
from datetime import datetime
from typing import Optional, Dict, Any, List

import requests
import pandas as pd
from pydantic import BaseModel

# --------------------------------------------------------------------------------
# CONFIG (edit paths / model info as needed)
# --------------------------------------------------------------------------------
try:
    from .llm_config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL
except ImportError:
    # fallback for fresh clones / public repo (safe template)
    from .llm_config_template import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("INSIGHT_AGENT")


# --------------------------------------------------------------------------------
# Data models
# --------------------------------------------------------------------------------
class NewsItem(BaseModel):
    title: str
    final_sentiment: Dict[str, Any]


class SentimentSummary(BaseModel):
    analysis_date: str
    stock_code: str
    sentiment_distribution: Dict[str, int]
    overall_sentiment: Dict[str, Any]
    time_series_sentiment: Optional[Dict[str, float]] = None
    detailed_analyses: List[NewsItem]


class MarketSnapshot(BaseModel):
    date: str
    close: Optional[float] = None
    rsi: Optional[float] = None
    ma5: Optional[float] = None
    ma10: Optional[float] = None
    ma20: Optional[float] = None
    macd: Optional[float] = None
    signal_line: Optional[float] = None
    macd_hist: Optional[float] = None
    volume_ratio: Optional[float] = None
    atr: Optional[float] = None
    historical_volatility: Optional[float] = None


class InsightOutput(BaseModel):
    request_id: str
    ticker: str

    # Educational fields controlled by your app, not by the LLM:
    user_prediction: Optional[str] = None   # "up" / "down" / "flat" / None
    result: Optional[str] = "unknown"       # true realized direction later: "up" / "down" / "flat" / "unknown"

    # LLM-generated content:
    model_prediction: Dict[str, Any]
    explanation_short: str
    teaching_note: str
    supporting_facts: List[str]
    factors: List[Dict[str, Any]] = []
    indicators: List[Dict[str, Any]] = []

    disclaimer: str
    generated_at: str



# --------------------------------------------------------------------------------
# Helpers: find latest cached files for a ticker
# --------------------------------------------------------------------------------
def find_latest_sentiment_for_ticker(base_dir: str, ticker: str) -> Optional[str]:
    pattern = os.path.join(base_dir, "cache", "sentiment_analysis", f"{ticker}_news_*_sentiment_analyses.json")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def find_latest_market_csv_for_ticker(base_dir: str, ticker: str) -> Optional[str]:
    pattern = os.path.join(base_dir, "cache", "stock_price_data", ticker, f"{ticker}_analysis_*.csv")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


# --------------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------------
def load_sentiment_json(path: str) -> SentimentSummary:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Ensure stock_code exists (fallback if missing)
    if "stock_code" not in raw:
        raw["stock_code"] = raw.get("stock_code", "unknown")

    # Clean up time_series_sentiment so Pydantic doesn't choke on None
    ts = raw.get("time_series_sentiment")
    if isinstance(ts, dict):
        # Keep only numeric values; drop None or weird types
        cleaned = {
            k: float(v)
            for k, v in ts.items()
            if isinstance(v, (int, float))
        }
        # If nothing valid remains, set it to None
        raw["time_series_sentiment"] = cleaned if cleaned else None
    else:
        raw["time_series_sentiment"] = None

    return SentimentSummary(**raw)


def load_market_csv(path: str) -> MarketSnapshot:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Market CSV is empty")
    last = df.tail(1).iloc[0]

    def safe(col):
        return float(last[col]) if col in df.columns and pd.notna(last[col]) else None

    return MarketSnapshot(
        date=str(last.get("date"))[:10],
        close=safe("close"),
        rsi=safe("rsi"),
        ma5=safe("ma5"),
        ma10=safe("ma10"),
        ma20=safe("ma20"),
        macd=safe("macd"),
        signal_line=safe("singal_line") or safe("signal_line"),
        macd_hist=safe("macd_hist"),
        volume_ratio=safe("volume_ratio"),
        atr=safe("atr"),
        historical_volatility=safe("historical_volatility"),
    )


# --------------------------------------------------------------------------------
# Call OpenRouter (LLM)
# --------------------------------------------------------------------------------
def call_llm(prompt: str) -> Optional[Dict[str, Any]]:
    """
    Call OpenAI's chat completion API using the shared config in llm_config.py.
    Returns a Python dict parsed from the JSON the model outputs.
    """
    if "paste-your-openai-key-here" in OPENAI_API_KEY:
        log.error("No API key set in OPENAI_API_KEY (see llm_config.py).")
        return None

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "model": OPENAI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.2,
    }

    try:
        r = requests.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            headers=headers,
            json=body,
            timeout=25,
        )
    except Exception as e:
        log.error(f"LLM request failed: {e}")
        return None

    log.info(f"LLM HTTP status: {r.status_code}")

    try:
        raw = r.json()
        text = raw["choices"][0]["message"]["content"]
    except Exception:
        log.error("Invalid LLM response; raw text saved to logs.")
        log.debug(r.text)
        return None

    # strip code fences if present
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        log.error("Failed to parse JSON returned by LLM. Returning None and logging content.")
        log.debug(text)
        return None


# --------------------------------------------------------------------------------
# Core insight logic
# --------------------------------------------------------------------------------
def _compute_simple_confidence(sentiment: SentimentSummary, market: MarketSnapshot) -> float:
    """
    Small heuristic fallback for confidence when LLM doesn't provide one.
    Combines sentiment confidence (if provided) and availability of technical indicators.
    """
    conf_parts = []
    s_conf = sentiment.overall_sentiment.get("confidence")
    if isinstance(s_conf, (int, float)):
        conf_parts.append(float(s_conf))
    # reward presence of key indicators
    indicators = 0
    for v in (market.ma5, market.ma10, market.ma20, market.rsi, market.macd):
        if v is not None:
            indicators += 1
    indicators_ratio = min(indicators / 5.0, 1.0)
    conf_parts.append(indicators_ratio * 0.6)  # weight technical presence
    # average and clamp
    avg = sum(conf_parts) / max(len(conf_parts), 1)
    return round(max(0.0, min(1.0, avg)), 2)


def run_insight(
    sentiment: SentimentSummary,
    market: MarketSnapshot,
    user_prediction: Optional[str] = None,
    result: Optional[str] = "unknown",
) -> InsightOutput:
    """
    user_prediction:
        - "up" / "down" / "flat" if the user made a guess in chat
        - None if the user didn't give any prediction
    result:
        - actual realized direction ("up" / "down" / "flat") to be filled later
        - at prediction time we typically use "unknown"
    """

    prompt = f"""
You are an investment insight agent. Analyze sentiment + technical indicators and return ONLY valid JSON
(no markdown, no code fences, no extra commentary).

Priority of signals when forming a view:
1. Moving-average (MA) trend
2. RSI
3. MACD
4. Sentiment

Conflict rule:
- If key indicators conflict (e.g., MA suggests up but RSI is overbought or MACD is weakening), set direction = "flat".

Use English for all text fields.

INPUT:
SentimentLabel={sentiment.overall_sentiment.get("label")}
SentimentScore={sentiment.overall_sentiment.get("score")}
Trend7d={sentiment.time_series_sentiment.get("recent_7d") if sentiment.time_series_sentiment else None}
Trend30d={sentiment.time_series_sentiment.get("recent_30d") if sentiment.time_series_sentiment else None}
Distribution={sentiment.sentiment_distribution}

Market:
Date={market.date}
Close={market.close}
MA5={market.ma5}, MA10={market.ma10}, MA20={market.ma20}
RSI={market.rsi}, MACD={market.macd}, Signal={market.signal_line}, Hist={market.macd_hist}
VolRatio={market.volume_ratio}, ATR={market.atr}, Volatility={market.historical_volatility}

Return JSON EXACTLY in this structure (no extra top-level keys):

{{
  "model_prediction": {{
    "direction": "up" | "down" | "flat",
    "confidence": 0.0,
    "reason": "one-sentence reason"
  }},
  "explanation_short": "one-sentence explanation",
  "teaching_note": "how to interpret these indicators",
  "supporting_facts": [
    "short bullet fact about price / MA / volume / etc.",
    "another short bullet fact"
  ],
  "factors": [
    {{
      "name": "MA",
      "value": "Price above MA5, MA10, and MA20",
      "impact": "positive"  // one of "positive", "negative", or "neutral"
    }},
    {{
      "name": "MACD",
      "value": "MACD above signal with positive histogram",
      "impact": "positive"
    }}
  ],
  "indicators": [
    {{
      "indicator": "MACD",
      "reading": "MACD above signal, bullish momentum"
    }},
    {{
      "indicator": "RSI",
      "reading": "RSI in neutral zone"
    }},
    {{
      "indicator": "MA",
      "reading": "price above all key moving averages"
    }}
  ],
  "disclaimer": "Educational only. Not financial advice."
}}
"""

    llm_out = call_llm(prompt) or {}

    # ---------------------------
    # Normalize returned structure
    # ---------------------------
    raw_model_pred = llm_out.get("model_prediction")
    model_prediction = raw_model_pred if isinstance(raw_model_pred, dict) else {}

    direction = model_prediction.get("direction", "unknown")
    confidence = model_prediction.get("confidence")

    # If LLM didn't supply a numeric confidence, compute fallback
    if not isinstance(confidence, (int, float)):
        confidence = _compute_simple_confidence(sentiment, market)

    model_prediction_normalized = {
        "direction": direction,
        "confidence": round(float(confidence), 2),
        "reason": model_prediction.get("reason", str(llm_out.get("reason", ""))) if isinstance(model_prediction.get("reason", ""), str) else ""
    }

    explanation_short = llm_out.get("explanation_short") or llm_out.get("explanation", "No explanation produced.")
    teaching_note = llm_out.get("teaching_note", "When multiple signals are present, prioritize MA trend, then RSI, then MACD, then sentiment.")

    supporting_facts_raw = llm_out.get("supporting_facts")
    if isinstance(supporting_facts_raw, list):
        supporting_facts = [str(x) for x in supporting_facts_raw]
    else:
        supporting_facts = [str(supporting_facts_raw or "No supporting facts.")]

    factors_raw = llm_out.get("factors")
    if isinstance(factors_raw, list):
        factors = [x for x in factors_raw if isinstance(x, dict)]
    else:
        factors = []

    indicators_raw = llm_out.get("indicators")
    if isinstance(indicators_raw, list):
        indicators = [x for x in indicators_raw if isinstance(x, dict)]
    else:
        indicators = []

    disclaimer = llm_out.get("disclaimer", "Educational only. Not financial advice.")

    return InsightOutput(
        request_id=f"req-{uuid.uuid4().hex[:8]}",
        ticker=sentiment.stock_code,
        user_prediction=user_prediction,   # <- from your app / chat
        result=result,                     # <- "unknown" now; fill later when you know the truth
        model_prediction=model_prediction_normalized,
        explanation_short=explanation_short,
        teaching_note=teaching_note,
        supporting_facts=supporting_facts,
        factors=factors,
        indicators=indicators,
        disclaimer=disclaimer,
        generated_at=datetime.now().isoformat(),
    )


# --------------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------------
def save_to_cache(result: InsightOutput, base_dir: Optional[str] = None) -> str:
    base = base_dir if base_dir else os.path.dirname(__file__)
    folder = os.path.join(base, "cache", "insight_reports")
    os.makedirs(folder, exist_ok=True)
    fpath = os.path.join(folder, f"{result.ticker}_{result.request_id}.json")
    with open(fpath, "w", encoding="utf-8") as f:
        json_str = result.model_dump_json(indent=2)
        f.write(json_str)
    log.info(f"Saved insight to: {fpath}")
    return fpath


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------
def _resolve_files_from_ticker(base_dir: str, ticker: str):
    sent = find_latest_sentiment_for_ticker(base_dir, ticker)
    market = find_latest_market_csv_for_ticker(base_dir, ticker)
    return sent, market


def main():
    parser = argparse.ArgumentParser(description="Run insight agent (pass --ticker or explicit paths)")
    parser.add_argument("--ticker", "-t", help="ticker (e.g. 601857). will auto-find latest CSV/JSON in cache/")
    parser.add_argument("--sentiment-path", help="explicit sentiment JSON path")
    parser.add_argument("--market-path", help="explicit market CSV path")
    parser.add_argument("--base-dir", default=os.path.dirname(os.path.abspath(__file__)), help="base directory to resolve cache/ paths (default: script dir)")
    args = parser.parse_args()

    base_dir = args.base_dir

    sentiment_path = args.sentiment_path
    market_path = args.market_path

    if args.ticker:
        s, m = _resolve_files_from_ticker(base_dir, args.ticker)
        if not sentiment_path and s:
            sentiment_path = s
        if not market_path and m:
            market_path = m

    if not sentiment_path or not market_path:
        log.error("Missing input files. Provide --ticker or both --sentiment-path and --market-path.")
        parser.print_help()
        raise SystemExit(2)

    # load
    try:
        sentiment = load_sentiment_json(os.path.join(base_dir, sentiment_path) if not os.path.isabs(sentiment_path) else sentiment_path)
    except Exception as e:
        log.error(f"Failed to load sentiment JSON: {e}")
        raise

    try:
        market = load_market_csv(os.path.join(base_dir, market_path) if not os.path.isabs(market_path) else market_path)
    except Exception as e:
        log.error(f"Failed to load market CSV: {e}")
        raise

    # run
    out = run_insight(sentiment, market)

    # print + save
    print("\n=== Insight Result ===")
    print(out.model_dump_json(indent=2, ensure_ascii=False))
    save_to_cache(out, base_dir=base_dir)


if __name__ == "__main__":
    main()

# insight_agent.py

import os
import json
import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

import requests
import pandas as pd
from pydantic import BaseModel


# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "OPENROUTER_API_KEY": "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",
    "OPENROUTER_MODEL": "minimax/minimax-m2:free",
    "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",

    "SENTIMENT_SAMPLE_PATH": "cache/sentiment_analysis/000300_news_2025-11-07_sentiment_analyses.json",
    "MARKET_SAMPLE_CSV": "cache/stock_price_data/600519/600519_analysis_20251107.csv",
}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("INSIGHT_AGENT")


# =============================================================================
# MODELS
# =============================================================================
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
    model_prediction: Dict[str, Any]  # includes direction + confidence + reason
    explanation_short: str
    teaching_note: str
    supporting_facts: List[str]
    disclaimer: str
    generated_at: str


# =============================================================================
# LOADING FUNCTIONS
# =============================================================================
def load_sentiment_json(path: str) -> SentimentSummary:
    with open(path, "r", encoding="utf-8") as f:
        return SentimentSummary(**json.load(f))


def load_market_csv(path: str) -> MarketSnapshot:
    df = pd.read_csv(path)
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
        signal_line=safe("signal_line"),
        macd_hist=safe("macd_hist"),
        volume_ratio=safe("volume_ratio"),
        atr=safe("atr"),
        historical_volatility=safe("historical_volatility"),
    )


# =============================================================================
# OPENROUTER LLM CALL
# =============================================================================
def call_openrouter(prompt: str) -> Optional[Dict[str, Any]]:
    headers = {
        "Authorization": f"Bearer {CONFIG['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
    }

    req = {
        "model": CONFIG["OPENROUTER_MODEL"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }

    r = requests.post(
        CONFIG["OPENROUTER_BASE_URL"] + "/chat/completions",
        headers=headers,
        json=req,
        timeout=20
    )

    log.info(f"LLM HTTP Response: {r.status_code}")

    try:
        text = r.json()["choices"][0]["message"]["content"]
    except Exception:
        log.error(f"❌ Invalid response from LLM:\n{r.text}")
        return None

    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        log.error("❌ JSON parse failed:\n" + text)
        return None


# =============================================================================
# INSIGHT GENERATION
# =============================================================================
def run_insight(sentiment: SentimentSummary, market: MarketSnapshot) -> InsightOutput:
    prompt = f"""
You are an investment insight agent. Analyze sentiment + technical indicators and respond in JSON only.

Rules:
- MA trend > RSI > MACD > Sentiment
- If signals conflict: direction = "flat"
- Include confidence (0 to 1)

INPUT:
SentimentLabel={sentiment.overall_sentiment.get("label")}
Score={sentiment.overall_sentiment.get("score")}
Trend7d={sentiment.time_series_sentiment.get("recent_7d")}
Trend30d={sentiment.time_series_sentiment.get("recent_30d")}
Distribution={sentiment.sentiment_distribution}

Market:
Close={market.close}, MA5={market.ma5}, MA10={market.ma10}, MA20={market.ma20}
RSI={market.rsi}, MACD={market.macd}, Signal={market.signal_line}, Hist={market.macd_hist}
VolRatio={market.volume_ratio}, ATR={market.atr}, Volatility={market.historical_volatility}

Return JSON ONLY:

{{
 "model_prediction": {{
     "direction": "up/down/flat",
     "confidence": 0.0,
     "reason": "short explanation"
 }},
 "explanation_short": "what is happening",
 "teaching_note": "how to interpret this",
 "supporting_facts": ["fact1", "fact2"],
 "disclaimer": "Educational only. Not financial advice."
}}
"""

    llm = call_openrouter(prompt) or {}

    llm.setdefault("model_prediction", {"direction": "unknown", "confidence": 0.0})
    llm.setdefault("explanation_short", "No explanation produced.")
    llm.setdefault("teaching_note", "Sentiment gives early signals; indicators confirm.")
    llm.setdefault("supporting_facts", ["No supporting facts generated."])
    llm.setdefault("disclaimer", "Educational only. Not financial advice.")

    return InsightOutput(
        request_id=f"req-{uuid.uuid4().hex[:8]}",
        ticker=sentiment.stock_code,
        model_prediction=llm["model_prediction"],
        explanation_short=llm["explanation_short"],
        teaching_note=llm["teaching_note"],
        supporting_facts=llm["supporting_facts"],
        disclaimer=llm["disclaimer"],
        generated_at=datetime.now().isoformat(),
    )


# =============================================================================
# SAVE RESULT
# =============================================================================
def save_to_cache(result: InsightOutput):
    folder = os.path.join(os.path.dirname(__file__), "cache", "insight_reports")
    os.makedirs(folder, exist_ok=True)

    fpath = os.path.join(folder, f"{result.ticker}_{result.request_id}.json")
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2, ensure_ascii=False))

    print(f"✅ Saved: {fpath}")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    sentiment = load_sentiment_json(os.path.join(base, CONFIG["SENTIMENT_SAMPLE_PATH"]))
    market = load_market_csv(os.path.join(base, CONFIG["MARKET_SAMPLE_CSV"]))

    result = run_insight(sentiment, market)

    print("\n=== Insight Result ===")
    print(result.model_dump_json(indent=2, ensure_ascii=False))

    save_to_cache(result)

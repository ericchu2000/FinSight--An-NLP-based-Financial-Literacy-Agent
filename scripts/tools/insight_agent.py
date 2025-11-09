# insight_agent.py
"""
Concise Investment Advisory / Insight Agent (FastAPI).
- Reads sentiment JSON + market CSV by default (paths configured in CONFIG).
- Calls OpenRouter (placeholders) to get LLM-generated explanation.
- Provides a single /explain endpoint expecting in-memory payloads (preferred),
  and a small CLI mode that loads from files for quick local testing.

Usage (dev):
  pip install fastapi uvicorn pydantic pandas requests
  uvicorn insight_agent:app --reload --port 8100

Configuration: edit values in CONFIG below (do NOT hardcode real secrets into source).
"""

from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
import json
import os
import uuid
import logging
import requests
import pandas as pd

# -----------------------
# CONFIG (edit these)
# -----------------------
CONFIG = {
    # OpenRouter placeholders (replace with env var usage in production)
    "OPENROUTER_API_KEY": "paste-your-api-key-here",
    "OPENROUTER_MODEL": "paste-your-model-here",
    "OPENROUTER_BASE_URL": "https://openrouter.ai/v1",

    # Local sample file paths (used by __main__ convenience runner only)
    "SENTIMENT_SAMPLE_PATH": "cache/news/stock_news/000300/000300_news_2025-11-07_sentiment_analyses.json",
    "MARKET_SAMPLE_CSV": "cache/stock_price_data/600519/600519_analysis_20251107.csv",
}

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("insight_agent")

# -----------------------
# Pydantic models
# -----------------------
class MarketFeatureSnapshot(BaseModel):
    date: str
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None
    rsi: Optional[float] = None
    ma5: Optional[float] = None
    daily_return: Optional[float] = None

class NewsItem(BaseModel):
    news_id: Optional[str] = None
    title: str
    content: Optional[str] = None
    publish_time: Optional[str] = None
    source: Optional[str] = None
    final_sentiment: Optional[Dict[str, Any]] = None

class SentimentSummary(BaseModel):
    analysis_date: str
    total_news_count: int
    overall_sentiment: Dict[str, Any]
    sentiment_distribution: Dict[str, int]
    time_series_sentiment: Optional[Dict[str, float]] = None
    detailed_analyses: List[NewsItem]
    stock_code: str

class UserPrediction(BaseModel):
    user_id: Optional[str] = None
    ticker: str
    query_date: str
    predicted_direction: Optional[str] = None
    predicted_confidence: Optional[float] = None

class InsightRequest(BaseModel):
    request_id: Optional[str] = None
    user_prediction: Optional[UserPrediction] = None
    sentiment: SentimentSummary
    market_snapshot: MarketFeatureSnapshot
    config: Optional[Dict[str, Any]] = Field(default_factory=lambda: {"verbosity": "short", "teaching_level": "beginner"})

class SupportingFact(BaseModel):
    type: str
    id: Optional[str] = None
    text: str

class InsightOutput(BaseModel):
    request_id: str
    ticker: str
    analysis_date: str
    model_prediction: Dict[str, Any]
    explanation_short: str
    teaching_note: str
    supporting_facts: List[SupportingFact]
    confidence_meta: Dict[str, Any]
    disclaimer: str
    meta: Optional[Dict[str, Any]] = None

# -----------------------
# App
# -----------------------
app = FastAPI(title="FinSight Insight Agent", version="0.3")

# -----------------------
# File loaders (simple adapters)
# -----------------------
def load_sentiment_from_file(path: str) -> SentimentSummary:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # ensure detailed items are modeled correctly
    if "detailed_analyses" in raw:
        raw["detailed_analyses"] = [
            {**item, "news_id": item.get("news_id") or f"news_{i}"} for i, item in enumerate(raw["detailed_analyses"])
        ]
    return SentimentSummary(**raw)

def load_market_from_csv(path: str) -> MarketFeatureSnapshot:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["date"], low_memory=False)
    if df.empty:
        raise ValueError("Market CSV is empty: " + path)
    last = df.sort_values("date").iloc[-1].to_dict()
    return MarketFeatureSnapshot(
        date=str(last.get("date"))[:10],
        open=last.get("open"),
        high=last.get("high"),
        low=last.get("low"),
        close=last.get("close"),
        volume=last.get("volume"),
        rsi=last.get("rsi"),
        ma5=last.get("ma5"),
        daily_return=last.get("daily_return")
    )

# -----------------------
# Helpers
# -----------------------
def pick_top_supporting_news(detailed_list: List[Dict[str, Any]], n: int = 2) -> List[SupportingFact]:
    items = []
    for i, d in enumerate(detailed_list):
        try:
            score = abs(float(d.get("final_sentiment", {}).get("score", 0) or 0))
        except Exception:
            score = 0.0
        publish = d.get("publish_time") or ""
        items.append((i, score, publish))
    items_sorted = sorted(items, key=lambda t: (t[1], t[2]), reverse=True)
    chosen = []
    for idx, _, _ in items_sorted[:n]:
        d = detailed_list[idx]
        text = (d.get("title") or d.get("content") or "")[:120]
        news_id = d.get("news_id") or f"news_{idx}"
        chosen.append(SupportingFact(type="news", id=news_id, text=text))
    return chosen

def simple_rule_predict(sentiment_summary: Dict[str, Any], market_snapshot: MarketFeatureSnapshot) -> Dict[str, Any]:
    label = sentiment_summary.get("overall_sentiment", {}).get("label", "neutral")
    s_score = float(sentiment_summary.get("overall_sentiment", {}).get("score", 0.5) or 0.5)
    rsi = market_snapshot.rsi or 50.0
    direction = "flat"
    conf = 0.5
    if label == "positive" and rsi > 52:
        direction = "up"
        conf = min(0.92, 0.4 + s_score * 0.5 + (rsi - 50) / 50)
    elif label == "negative" and rsi < 48:
        direction = "down"
        conf = min(0.92, 0.4 + (1 - s_score) * 0.5 + (50 - rsi) / 50)
    else:
        direction = "flat"
        conf = max(0.2, 0.4 + (abs(s_score - 0.5)) * 0.6)
    return {"direction": direction, "confidence": round(float(conf), 2)}

# -----------------------
# Prompt builder & OpenRouter caller
# -----------------------
PROMPT_TEMPLATE = """System:
You are a concise, beginner-friendly financial educator. Return ONLY a single JSON object (no extra text).
Required keys: model_prediction, explanation_short, teaching_note, supporting_facts, disclaimer.

Context:
Ticker: {ticker}
Date: {date}
Sentiment: pos={pos}, neu={neu}, neg={neg}, overall_label={overall_label}, overall_score={overall_score}
Market: RSI={rsi}, MA5={ma5}, daily_return={daily_return}

Top news:
{top_news}

Task:
Return JSON only.
"""

def build_prompt(req: InsightRequest) -> str:
    ds = req.sentiment
    m = req.market_snapshot
    top_news = "\n".join([f"- {n.title}" for n in ds.detailed_analyses[:4]]) or "- none"
    prompt = PROMPT_TEMPLATE.format(
        ticker=ds.stock_code,
        date=ds.analysis_date,
        pos=ds.sentiment_distribution.get("positive", 0),
        neu=ds.sentiment_distribution.get("neutral", 0),
        neg=ds.sentiment_distribution.get("negative", 0),
        overall_label=ds.overall_sentiment.get("label", "neutral"),
        overall_score=ds.overall_sentiment.get("score", 0.5),
        rsi=m.rsi or "N/A",
        ma5=m.ma5 or "N/A",
        daily_return=m.daily_return or "N/A",
        top_news=top_news
    )
    return prompt

def call_openrouter(prompt: str) -> Dict[str, Any]:
    api_key = CONFIG["OPENROUTER_API_KEY"]
    model = CONFIG["OPENROUTER_MODEL"]
    base = CONFIG["OPENROUTER_BASE_URL"].rstrip("/")
    if api_key.startswith("paste-your-api-key"):
        logger.warning("OPENROUTER_API_KEY placeholder in CONFIG; call will fail until replaced.")
    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 450
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        if r.status_code != 200:
            logger.error("OpenRouter error %s: %s", r.status_code, r.text[:400])
            raise RuntimeError("OpenRouter API error")
        j = r.json()
        raw = j.get("choices", [{}])[0].get("message", {}).get("content", "")
        raw = raw.strip()
        if raw.startswith("```"):
            try:
                raw = raw.split("```", 2)[-1].rsplit("```", 1)[0].strip()
            except Exception:
                raw = raw.replace("```", "").strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # salvage substring
            try:
                start = raw.index("{"); end = raw.rindex("}") + 1
                parsed = json.loads(raw[start:end])
            except Exception:
                logger.error("Failed to parse LLM output. Head: %s", raw[:400])
                raise
        # normalize confidence
        if "model_prediction" in parsed:
            try:
                conf = float(parsed["model_prediction"].get("confidence", 0) or 0)
                parsed["model_prediction"]["confidence"] = max(0.0, min(1.0, conf))
            except Exception:
                parsed["model_prediction"]["confidence"] = 0.2
        else:
            raise ValueError("LLM JSON missing model_prediction")
        return parsed
    except Exception as e:
        logger.exception("OpenRouter call failed: %s", e)
        return {
            "model_prediction": {"direction": "flat", "confidence": 0.2},
            "explanation_short": "Model unavailable; defaulting to neutral.",
            "teaching_note": "No LLM output available.",
            "supporting_facts": [],
            "disclaimer": "Educational only — not financial advice."
        }

# -----------------------
# Core explain logic
# -----------------------
def explain_market(req: InsightRequest) -> InsightOutput:
    req_id = req.request_id or f"req-{uuid.uuid4().hex[:8]}"
    ds = req.sentiment
    m = req.market_snapshot

    # supporting facts from sentiment
    detailed = [n.dict() if isinstance(n, NewsItem) else n for n in ds.detailed_analyses]
    support_from_sent = pick_top_supporting_news(detailed, n=2)

    # call LLM
    prompt = build_prompt(req)
    llm_out = call_openrouter(prompt)

    model_pred = llm_out.get("model_prediction") or simple_rule_predict(ds.dict(), m)

    # parse LLM supporting facts (if present)
    llm_support_raw = llm_out.get("supporting_facts") or []
    support_list: List[SupportingFact] = []
    if isinstance(llm_support_raw, list) and llm_support_raw:
        for s in llm_support_raw[:3]:
            if isinstance(s, dict):
                support_list.append(SupportingFact(type=s.get("type", "news"), id=s.get("id"), text=(s.get("text") or "")[:120]))
            else:
                support_list.append(SupportingFact(type="news", id=None, text=str(s)[:120]))
    if not support_list:
        support_list = support_from_sent

    explanation_short = llm_out.get("explanation_short") or ""
    teaching_note = llm_out.get("teaching_note") or ""
    disclaimer = llm_out.get("disclaimer") or "Educational only — not financial advice."

    confidence_meta = {
        "signal_sources": ["sentiment", "market_features"],
        "confidence_reason": f"Derived from sentiment ({ds.overall_sentiment.get('label')}) and RSI={m.rsi}"
    }

    return InsightOutput(
        request_id=req_id,
        ticker=ds.stock_code,
        analysis_date=ds.analysis_date,
        model_prediction=model_pred,
        explanation_short=explanation_short,
        teaching_note=teaching_note,
        supporting_facts=support_list,
        confidence_meta=confidence_meta,
        disclaimer=disclaimer,
        meta={"model_version": "insight-v0.3", "generated_at": datetime.now().isoformat()}
    )

# -----------------------
# API endpoint
# -----------------------
@app.post("/explain", response_model=InsightOutput)
def explain_endpoint(req: InsightRequest):
    try:
        out = explain_market(req)
        return out
    except Exception as e:
        logger.exception("explain_endpoint error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------
# CLI/dev helper
# -----------------------
if __name__ == "__main__":
    try:
        s = load_sentiment_from_file(CONFIG["SENTIMENT_SAMPLE_PATH"])
        m = load_market_from_csv(CONFIG["MARKET_SAMPLE_CSV"])
    except Exception as err:
        logger.error("Sample load failed: %s", err)
        print("Edit CONFIG paths at top of file to point to your sample files.")
        raise SystemExit(1)

    req = InsightRequest(request_id=None, sentiment=s, market_snapshot=m)
    out = explain_market(req)
    print(out.json(indent=2, ensure_ascii=False))

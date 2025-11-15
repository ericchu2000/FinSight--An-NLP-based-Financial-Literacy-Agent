#!/usr/bin/env python3
"""
Simple Gradio front-end for FinSight educational bot.

Run from project root:
    python -m scripts.ui.gradio_app
Then open the URL Gradio prints (usually http://127.0.0.1:7860).
"""

import os
import sys
import glob
import re
import logging
from typing import List, Tuple

import gradio as gr

# ----------------- Import project modules -----------------

# Make project root importable
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.tools.run_pipeline import run_pipeline, CACHE_BASE  # you already have this


log = logging.getLogger("FIN_SIGHT_GRADIO")
logging.basicConfig(level=logging.INFO)


# ----------------- Helpers -----------------

def extract_ticker_from_message(message: str) -> str | None:
    """
    Very simple heuristic: find a 6-digit number as ticker (e.g., 600028, 601857, 301201).
    You can replace this later with a better parser.
    """
    if not message:
        return None
    match = re.search(r"\b\d{6}\b", message)
    return match.group(0) if match else None


def find_latest_education_report(ticker: str) -> str | None:
    """
    Find the latest education markdown file for a given ticker.
    Looks under CACHE_BASE/education_reports/<ticker>_*.md
    """
    edu_dir = os.path.join(CACHE_BASE, "education_reports")
    pattern = os.path.join(edu_dir, f"{ticker}_*.md")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


# ----------------- Chat logic -----------------

def chat_fn(message: str, history: List[Tuple[str, str]]):
    """
    Gradio ChatInterface handler.

    - message: latest user message
    - history: list of (user, bot) tuples (previous turns) [managed by Gradio]

    IMPORTANT:
    For gr.ChatInterface, this function must return a *single message*
    (string / markdown / ChatMessage), NOT (message, history).
    """

    # 1. Try to extract ticker
    ticker = extract_ticker_from_message(message)

    if not ticker:
        reply = (
            "I didn’t detect a 6-digit stock code in your message.\n\n"
            "Please ask something like: `What do you think of 600028?` "
            "or `Can you analyze 601857 for me?`"
        )
        return reply

    # 2. Acknowledge + run pipeline
    log.info(f"Received request for ticker: {ticker}")

    intro = (
        f"Got it! I’ll analyze **{ticker}** using the full pipeline:\n"
        "- market data + technical indicators\n"
        "- news sentiment\n"
        "- LLM insight\n"
        "- educational explanation\n\n"
        "_Running analysis..._\n"
    )

    # 3. Run the pipeline (this will create insight + education reports in cache)
    try:
        # You can pass days / max_news here if you want, e.g. days=365, max_news=30
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

    # 4. Load the latest education report
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

    # 5. Build final assistant message (intro + educational content)
    reply = (
        f"{intro}\n"
        f"✅ Here is the educational explanation for **{ticker}** "
        f"(from `{os.path.basename(edu_path)}`):\n\n"
        f"{edu_markdown}"
    )

    # For ChatInterface: just return the assistant's reply (no history tuple!)
    return reply


# ----------------- Gradio app -----------------

def main():
    description_md = """
# 🎓 FinSight Educational Bot (Gradio Demo)

Type a message containing a **6-digit stock code** (e.g. `600028`, `601857`, `301201`).

Examples:
- `What do you think of 600028?`
- `Can you analyze 601857 for me?`
"""

    chat = gr.ChatInterface(
        fn=chat_fn,
        title="FinSight Educational Bot",
        description=description_md,
    )

    chat.launch()


if __name__ == "__main__":
    main()

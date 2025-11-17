#!/usr/bin/env python3
"""
Gradio entry point for FinSight Educational Bot.

Run from project root:
    python -m scripts.ui.gradio_app
"""

import gradio as gr

from scripts.ui.chat_logic import chat_fn, DESCRIPTION_MD


def main():
    chat = gr.ChatInterface(
        fn=chat_fn,
        title="FinSight Educational Bot",
        description=DESCRIPTION_MD,
    )
    chat.launch()


if __name__ == "__main__":
    main()

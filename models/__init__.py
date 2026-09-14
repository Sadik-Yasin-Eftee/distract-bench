"""
models package

WHAT: Unified client interface over the three model families under study.

WHY: The core research question compares reasoning models with an explicit
     token budget (Gemini 2.0/2.5 Flash Thinking, DeepSeek-R1) against a
     non-reasoning baseline (Llama 3.3 70B via Groq). A shared
     query(prompt, token_budget) -> response interface lets the experiment
     loop in experiments/run_scaling_curves.py treat all three identically.

RESEARCH STEP: Week 2, Steps 9-12.
"""

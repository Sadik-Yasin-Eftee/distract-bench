"""
experiments package

WHAT: Prompt construction, correctness checking, and the main scaling-curve
      experiment loop.

WHY: This is the core data-collection layer for the inverse scaling
     experiment: for every (instance, condition, token budget, model)
     combination it builds the prompt, queries the model, checks
     correctness, and saves the result incrementally.

RESEARCH STEP: Week 2-3, Steps 13-16.
"""

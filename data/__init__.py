"""
data package

WHAT: Dataset loading, normalization, and sampling for DISTRACT-BENCH.

WHY: Groups all code that turns raw HuggingFace datasets (GSM8K, TriviaQA,
     CommonsenseQA) into the 600-instance normalized benchmark used by every
     later stage of the pipeline (distractor generation, model evaluation).

RESEARCH STEP: Week 1, Steps 3-5.
"""

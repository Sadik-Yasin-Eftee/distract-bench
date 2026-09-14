"""
test_run_scaling_curves.py

WHAT: Unit tests for experiments/run_scaling_curves.py's
      _build_model_clients(), which decides which models to run based on
      which API keys are configured.

WHY: This project's 5-week timeline means API keys for the three
     providers are typically provisioned one at a time, not all at once
     (e.g. only GOOGLE_API_KEY available while DeepSeek/Groq accounts are
     still being set up). _build_model_clients() must skip any model
     whose provider key is absent rather than crashing the whole
     experiment — this is exactly the kind of behavior that is easy to
     silently break in a later refactor (e.g. by moving the key check
     after client construction instead of before), so it has its own
     regression test.

RESEARCH STEP: Ongoing — run via `pytest tests/test_run_scaling_curves.py`.

DEPENDS ON: experiments/run_scaling_curves.py.
"""

from unittest.mock import patch, MagicMock

import config
import experiments.run_scaling_curves as rsc

_FAKE_SPECS = {
    "gemini_primary": {"provider": "gemini", "model_id": "gemini-3.5-flash"},
    "gemini_secondary": {"provider": "gemini", "model_id": "gemini-3.6-flash"},
    "deepseek_reasoning": {"provider": "deepseek", "model_id": "deepseek-flash"},
    "llama_baseline": {"provider": "groq", "model_id_candidates": ("llama-3.3-70b-versatile",)},
}


def test_builds_only_gemini_clients_when_only_google_key_set():
    with patch.object(config, "MODEL_SPECS", _FAKE_SPECS), \
         patch.object(config, "GOOGLE_API_KEY", "fake-google-key"), \
         patch.object(config, "DEEPSEEK_API_KEY", ""), \
         patch.object(config, "GROQ_API_KEY", ""), \
         patch("experiments.run_scaling_curves.GeminiClient", return_value=MagicMock()), \
         patch("experiments.run_scaling_curves.DeepSeekClient") as mock_deepseek, \
         patch("experiments.run_scaling_curves.GroqClient") as mock_groq:
        clients = rsc._build_model_clients()

    assert set(clients.keys()) == {"gemini_primary", "gemini_secondary"}
    mock_deepseek.assert_not_called()
    mock_groq.assert_not_called()


def test_builds_all_clients_when_all_keys_set():
    with patch.object(config, "MODEL_SPECS", _FAKE_SPECS), \
         patch.object(config, "GOOGLE_API_KEY", "fake-google-key"), \
         patch.object(config, "DEEPSEEK_API_KEY", "fake-deepseek-key"), \
         patch.object(config, "GROQ_API_KEY", "fake-groq-key"), \
         patch("experiments.run_scaling_curves.GeminiClient", return_value=MagicMock()), \
         patch("experiments.run_scaling_curves.DeepSeekClient", return_value=MagicMock()), \
         patch("experiments.run_scaling_curves.GroqClient", return_value=MagicMock()):
        clients = rsc._build_model_clients()

    assert set(clients.keys()) == set(_FAKE_SPECS.keys())


def test_raises_clear_error_when_no_keys_set():
    with patch.object(config, "MODEL_SPECS", _FAKE_SPECS), \
         patch.object(config, "GOOGLE_API_KEY", ""), \
         patch.object(config, "DEEPSEEK_API_KEY", ""), \
         patch.object(config, "GROQ_API_KEY", ""):
        try:
            rsc._build_model_clients()
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "GOOGLE_API_KEY" in str(e)

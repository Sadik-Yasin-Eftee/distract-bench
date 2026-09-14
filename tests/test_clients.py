"""
test_clients.py

WHAT: Unit tests for models/gemini_client.py, models/deepseek_client.py,
      and models/groq_client.py that mock the underlying SDK objects.

WHY: The engineering requirements call for testing model clients before
     each week's experiments, but the clients themselves make real,
     quota-consuming network calls. Mocking each provider's SDK client
     lets this project verify the parts that are actually this project's
     own code and therefore actually at risk of a bug — request
     construction (right model id, right budget-to-parameter mapping)
     and response parsing (pulling answer/reasoning text and metadata out
     of each SDK's response shape) — without spending API quota or
     requiring network access / API keys to run in CI or before keys are
     provisioned.

RESEARCH STEP: Ongoing — run via `pytest tests/test_clients.py` before
               each week's experiment scripts that call live models.

DEPENDS ON: models/gemini_client.py, models/deepseek_client.py,
            models/groq_client.py. Uses unittest.mock to replace each
            provider's SDK client class.

DESIGN DECISIONS:
- Mocks are constructed to match the minimal real response shape each
  client's `_query_once` actually reads (verified against each SDK's
  installed version during development of the corresponding client file)
  rather than a full realistic payload — keeping each mock's shape
  exactly as narrow as what the code under test touches makes a test
  failure point directly at the accessor that broke, rather than at an
  incidental mismatch elsewhere in an over-specified fake payload.
"""

from unittest.mock import MagicMock, patch

import config

# Tests set API keys directly on the config module (rather than requiring
# a real .env) since the module-under-test's __init__ methods check
# `config.GOOGLE_API_KEY` etc. at call time via the *module* attribute —
# patching config.<KEY> here is sufficient without touching os.environ.


# ────────────────────────────────────────────────────────────────────────
# GEMINI
# ────────────────────────────────────────────────────────────────────────

def test_gemini_missing_api_key_raises():
    from models.gemini_client import GeminiClient

    with patch.object(config, "GOOGLE_API_KEY", None):
        try:
            GeminiClient("gemini-3.5-flash")
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "GOOGLE_API_KEY" in str(e)


def _mock_gemini_model(name):
    m = MagicMock()
    m.name = f"models/{name}"
    return m


def test_gemini_rejects_unavailable_model_id():
    from models.gemini_client import GeminiClient

    fake_client = MagicMock()
    fake_client.models.list.return_value = [_mock_gemini_model("gemini-3.5-flash")]

    with patch.object(config, "GOOGLE_API_KEY", "fake-key"), \
         patch("models.gemini_client.genai.Client", return_value=fake_client):
        try:
            GeminiClient("gemini-2.0-flash-thinking-exp")  # known-dead model id
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "not available" in str(e)


def test_gemini_query_parses_response():
    from models.gemini_client import GeminiClient

    fake_client = MagicMock()
    fake_client.models.list.return_value = [_mock_gemini_model("gemini-3.5-flash")]

    thought_part = MagicMock(thought=True, text="Let me think... 2+2=4")
    answer_part = MagicMock(thought=False, text="4")
    fake_response = MagicMock()
    fake_response.text = "4"
    fake_response.candidates = [MagicMock(content=MagicMock(parts=[thought_part, answer_part]))]
    fake_response.usage_metadata.model_dump.return_value = {"total_token_count": 42}
    fake_client.models.generate_content.return_value = fake_response

    with patch.object(config, "GOOGLE_API_KEY", "fake-key"), \
         patch("models.gemini_client.genai.Client", return_value=fake_client):
        client = GeminiClient("gemini-3.5-flash")
        result = client.query("What is 2+2?", token_budget=256)

    assert result.answer_text == "4"
    assert result.reasoning_text == "Let me think... 2+2=4"
    assert result.token_budget == 256
    assert result.raw_metadata["usage_metadata"]["total_token_count"] == 42

    # Confirm the thinking budget was actually forwarded to the API call,
    # not silently dropped — this is the whole point of this client.
    _, kwargs = fake_client.models.generate_content.call_args
    assert kwargs["config"].thinking_config.thinking_budget == 256


# ────────────────────────────────────────────────────────────────────────
# DEEPSEEK
# ────────────────────────────────────────────────────────────────────────

def test_deepseek_missing_api_key_raises():
    from models.deepseek_client import DeepSeekClient

    with patch.object(config, "DEEPSEEK_API_KEY", None):
        try:
            DeepSeekClient("deepseek-flash")
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "DEEPSEEK_API_KEY" in str(e)


def test_deepseek_query_parses_reasoning_content():
    from models.deepseek_client import DeepSeekClient

    fake_client = MagicMock()
    fake_client.models.list.return_value.data = [MagicMock(id="deepseek-flash")]

    fake_message = MagicMock()
    fake_message.content = "4"
    fake_message.reasoning_content = "2 + 2 is a basic addition, equals 4."
    fake_choice = MagicMock(message=fake_message, finish_reason="stop")
    fake_response = MagicMock(choices=[fake_choice])
    fake_response.usage.model_dump.return_value = {"total_tokens": 30}
    fake_client.chat.completions.create.return_value = fake_response

    with patch.object(config, "DEEPSEEK_API_KEY", "fake-key"), \
         patch("models.deepseek_client.OpenAI", return_value=fake_client):
        client = DeepSeekClient("deepseek-flash")
        result = client.query("What is 2+2?", token_budget=512)

    assert result.answer_text == "4"
    assert result.reasoning_text == "2 + 2 is a basic addition, equals 4."

    _, kwargs = fake_client.chat.completions.create.call_args
    # max_tokens is this project's operationalization of "token budget"
    # for DeepSeek — see deepseek_client.py's DESIGN DECISIONS.
    assert kwargs["max_tokens"] == 512
    assert kwargs["reasoning_effort"] == "high"
    assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}


# ────────────────────────────────────────────────────────────────────────
# GROQ
# ────────────────────────────────────────────────────────────────────────

def test_groq_missing_api_key_raises():
    from models.groq_client import GroqClient

    with patch.object(config, "GROQ_API_KEY", None):
        try:
            GroqClient(("llama-3.3-70b-versatile",))
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "GROQ_API_KEY" in str(e)


def test_groq_resolves_first_available_candidate():
    from models.groq_client import GroqClient

    fake_client = MagicMock()
    # First candidate is decommissioned; second is live — client should
    # skip past the dead one rather than failing outright.
    fake_client.models.list.return_value.data = [MagicMock(id="llama-3.1-8b-instant")]

    with patch.object(config, "GROQ_API_KEY", "fake-key"), \
         patch("models.groq_client.Groq", return_value=fake_client):
        client = GroqClient(("llama-3.3-70b-versatile", "llama-3.1-8b-instant"))

    assert client.model_id == "llama-3.1-8b-instant"


def test_groq_raises_when_no_candidate_available():
    from models.groq_client import GroqClient

    fake_client = MagicMock()
    fake_client.models.list.return_value.data = [MagicMock(id="some-other-model")]

    with patch.object(config, "GROQ_API_KEY", "fake-key"), \
         patch("models.groq_client.Groq", return_value=fake_client):
        try:
            GroqClient(("llama-3.3-70b-versatile", "llama-3.1-8b-instant"))
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "None of the candidate" in str(e)


def test_groq_query_ignores_token_budget_as_generation_param():
    from models.groq_client import GroqClient

    fake_client = MagicMock()
    fake_client.models.list.return_value.data = [MagicMock(id="llama-3.1-8b-instant")]

    fake_choice = MagicMock(message=MagicMock(content="4"), finish_reason="stop")
    fake_response = MagicMock(choices=[fake_choice])
    fake_response.usage.model_dump.return_value = {"total_tokens": 10}
    fake_client.chat.completions.create.return_value = fake_response

    with patch.object(config, "GROQ_API_KEY", "fake-key"), \
         patch("models.groq_client.Groq", return_value=fake_client):
        client = GroqClient(("llama-3.1-8b-instant",))
        result = client.query("What is 2+2?", token_budget=8192)

    assert result.answer_text == "4"
    assert result.reasoning_text is None
    # token_budget is recorded on the result for traceability...
    assert result.token_budget == 8192
    # ...but must NOT have been forwarded as a generation-limiting
    # parameter, since this baseline has no thinking phase to budget.
    _, kwargs = fake_client.chat.completions.create.call_args
    assert "max_tokens" not in kwargs
    assert "thinking_budget" not in kwargs

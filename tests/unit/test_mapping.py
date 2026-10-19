"""Unit tests for pure request mapping functions across providers."""

from unified_llm.core.schemas import ChatRequest, GenerationParams, Message
from unified_llm.providers._openai_style import to_openai_payload
from unified_llm.providers.gemini_adapter import to_gemini_payload
from unified_llm.providers.ollama_adapter import to_ollama_payload


def test_openai_payload_mapping():
    req = ChatRequest(
        provider="openai",
        model="gpt-4o",
        messages=[
            Message(role="system", content="System instruction"),
            Message(role="user", content="Hello assistant"),
        ],
        params=GenerationParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
            stop=["STOP"],
            seed=42,
            presence_penalty=0.1,
            frequency_penalty=0.2,
            response_format="json",
        ),
        provider_options={"logit_bias": {"50256": -100}},
    )

    payload = to_openai_payload(req)

    assert payload["model"] == "gpt-4o"
    assert len(payload["messages"]) == 2
    assert payload["messages"][0] == {"role": "system", "content": "System instruction"}
    assert payload["messages"][1] == {"role": "user", "content": "Hello assistant"}
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.9
    assert payload["max_tokens"] == 256
    assert payload["stop"] == ["STOP"]
    assert payload["seed"] == 42
    assert payload["presence_penalty"] == 0.1
    assert payload["frequency_penalty"] == 0.2
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["logit_bias"] == {"50256": -100}


def test_gemini_payload_mapping():
    req = ChatRequest(
        provider="gemini",
        model="gemini-1.5-flash",
        messages=[
            Message(role="system", content="System part 1"),
            Message(role="system", content="System part 2"),
            Message(role="user", content="User prompt"),
            Message(role="assistant", content="Assistant reply"),
        ],
        params=GenerationParams(
            temperature=0.3,
            max_tokens=500,
            top_k=20,
            stop=["\n\n"],
            response_format="json",
        ),
    )

    contents, config, warnings = to_gemini_payload(req)

    # System instruction concatenated
    assert config.system_instruction == "System part 1\n\nSystem part 2"
    assert len(warnings) == 1
    assert warnings[0].code == "multiple_system_prompts_joined"

    # Contents should only have user and assistant (mapped to 'model')
    assert len(contents) == 2
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "User prompt"
    assert contents[1].role == "model"
    assert contents[1].parts[0].text == "Assistant reply"

    assert config.temperature == 0.3
    assert config.max_output_tokens == 500
    assert config.top_k == 20
    assert config.stop_sequences == ["\n\n"]
    assert config.response_mime_type == "application/json"


def test_ollama_payload_mapping():
    req = ChatRequest(
        provider="ollama",
        model="llama3.2",
        messages=[
            Message(role="system", content="System prompt"),
            Message(role="user", content="Hello Ollama"),
        ],
        params=GenerationParams(
            temperature=0.8,
            max_tokens=150,
            top_k=40,
            seed=123,
            response_format="json",
        ),
        provider_options={"mirostat": 2},
    )

    payload = to_ollama_payload(req, keep_alive="5m")

    assert payload["model"] == "llama3.2"
    assert payload["stream"] is False
    assert payload["keep_alive"] == "5m"
    assert payload["format"] == "json"
    assert payload["mirostat"] == 2

    opts = payload["options"]
    assert opts["temperature"] == 0.8
    assert opts["num_predict"] == 150
    assert opts["top_k"] == 40
    assert opts["seed"] == 123

# Unified LLM Integration Layer (ULIL) / LLMBridge

A high-performance, standardized middleware service providing a single unified API for LLM providers (OpenAI, Groq, Google Gemini, Ollama, and extensible plugins).

## Features

- **Unified API Contract**: Canonical request and response schemas across all providers.
- **Provider Adapters**: Independent adapters for OpenAI, Groq, Google Gemini, and Ollama.
- **Robust Retry & Timeout Policy**: Centralized Tenacity retries with exponential backoff, jitter, and total request deadline enforcement.
- **Strict & Lenient Validation**: Safe pre-flight capability checks (`strict_params=false` lenient warnings or `strict_params=true` fast validation failure).
- **Error Taxonomy**: Consistent HTTP status codes and structured error envelope for all client and upstream failures.
- **Observability**: Structlog structured logging (JSON or console), automatic secret redaction, request correlation IDs (`X-Request-ID`), and Prometheus metrics (`/metrics`).
- **Security**: Inbound bearer token authentication (`ULIL_API_KEYS`), payload size limiting, and sanitized error messages.
- **Docker-ready**: Multi-stage, non-root Docker build.

## Quick Start

### 1. Install Dependencies
```bash
pip install -e .
```

### 2. Configure Environment
Copy `.env.example` to `.env` and set your credentials:
```bash
cp .env.example .env
```

### 3. Run Locally
```bash
uvicorn unified_llm.main:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

### 4. API Endpoints
- `POST /v1/messages` - Canonical message execution for dialogues, agent loops, and tool calling
- `GET /v1/providers` - Inspect configured providers and capabilities
- `GET /healthz` - Liveness health check
- `GET /readyz` - Readiness probe (`?deep=true` for live upstream checks)
- `GET /metrics` - Prometheus metrics

---

## Messages API Specification (`POST /v1/messages`)

The `/v1/messages` endpoint provides a unified, provider-agnostic interface for multi-turn dialogues, tool calling, and autonomous agent loops across all upstream providers (OpenAI, Groq, Google Gemini, Ollama).

### HTTP Headers

| Header | Type | Required | Description |
|---|---|---|---|
| `Content-Type` | String | **Yes** | Must be `application/json` |
| `Authorization` | String | Conditional | `Bearer <API_KEY>` (required when `ULIL_API_KEYS` is configured) |
| `X-Request-ID` | String | Optional | Request correlation ID (auto-generated if omitted, returned in response) |
| `X-Trace-ID` | String | Optional | Distributed tracing identifier |
| `X-Caller-ID` | String | Optional | Caller, service, or tenant identifier for tracking |

---

### Request Body Schema

A request payload represents a canonical `MessageRequest`:

```json
{
  "provider": "openai",
  "model": "gpt-4o",
  "messages": [
    {
      "role": "system",
      "content": "You are an autonomous assistant with access to tools."
    },
    {
      "role": "user",
      "content": "What is the current status of order #9821?"
    }
  ],
  "params": {
    "temperature": 0.2,
    "max_tokens": 1024,
    "top_p": 1.0,
    "stop": ["Observation:"],
    "response_format": "text"
  },
  "options": {
    "timeout_s": 30.0,
    "max_retries": 2,
    "strict_params": false,
    "include_raw": false
  },
  "metadata": {
    "session_id": "sess_12345",
    "agent_id": "order_lookup_v1"
  }
}
```

### Field Descriptions

#### Top-Level Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `provider` | `string` | **Yes** | Target backend adapter: `"openai"`, `"groq"`, `"gemini"`, or `"ollama"`. |
| `model` | `string` | **Yes** | Model identifier (e.g. `"gpt-4o"`, `"llama-3.3-70b-versatile"`, `"gemini-1.5-pro"`). |
| `messages` | `Message[]` | **Yes** | Ordered array of message objects (minimum 1 item). |
| `params` | `GenerationParams` | No | Model generation parameters (defaults applied if omitted). |
| `options` | `RequestOptions` | No | Middleware and resilience execution options. |
| `provider_options`| `object` | No | Provider-specific pass-through arguments. |
| `metadata` | `object` | No | Key-value string pairs attached to structured logs and telemetry. |

#### Message Object (`messages[]`)

| Field | Type | Required | Description |
|---|---|---|---|
| `role` | `string` | **Yes** | Message role: `"system"`, `"user"`, or `"assistant"`. |
| `content` | `string` \| `array` | **Yes** | Message text or structured multimodal/tool parts. |

#### Generation Parameters (`params`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `temperature` | `float` | Provider default | Sampling temperature (`>= 0.0`). |
| `top_p` | `float` | Provider default | Nucleus sampling probability (`0.0 < top_p <= 1.0`). |
| `top_k` | `int` | Provider default | Top-k token sampling (`>= 1`). |
| `max_tokens` | `int` | Provider default | Maximum tokens to generate (`>= 1`). |
| `stop` | `string[]` | `null` | Up to 16 stop sequences. |
| `seed` | `int` | `null` | Deterministic random seed (when supported by provider). |
| `presence_penalty` | `float` | `null` | Penalty for new tokens based on presence in text. |
| `frequency_penalty`| `float` | `null` | Penalty for new tokens based on existing frequency. |
| `response_format` | `string` | `null` | Structured output enforcement: `"text"` or `"json"`. |

#### Execution Options (`options`)

| Option | Type | Default | Description |
|---|---|---|---|
| `timeout_s` | `float` | Provider default | Request timeout in seconds (`<= 600.0`). Governed by service ceiling. |
| `max_retries` | `int` | Policy default | Max retries for transient upstream failures (429, 502, 503, 504) (`0` to `5`). |
| `strict_params` | `bool` | `false` | **Lenient mode** (`false`): Drops unsupported parameters and returns warnings. **Strict mode** (`true`): Fails fast with `400 Bad Request` if a parameter is not supported by the provider's declared capabilities. |
| `include_raw` | `bool` | `false` | When `true`, includes the unmodified upstream provider response in `raw`. |

---

### Agent Loops & Tool Execution Pattern

In an autonomous agent architecture (ReAct or tool-use loop), feed the history of messages, agent thoughts, and tool observations into `messages` on each iteration:

```json
{
  "provider": "groq",
  "model": "llama-3.3-70b-versatile",
  "messages": [
    {
      "role": "system",
      "content": "You are a database agent. To run queries, respond with: TOOL_CALL: query_db ARGS: {\"sql\": \"...\"}"
    },
    {
      "role": "user",
      "content": "How many active users signed up this week?"
    },
    {
      "role": "assistant",
      "content": "TOOL_CALL: query_db ARGS: {\"sql\": \"SELECT COUNT(*) FROM users WHERE created_at >= NOW() - INTERVAL '7 days'\"}"
    },
    {
      "role": "user",
      "content": "TOOL_RESULT: 1,482 active users"
    }
  ],
  "params": {
    "temperature": 0.0
  }
}
```

---

### Example cURL Request

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-key-1" \
  -d '{
    "provider": "openai",
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "Explain vector embeddings in one sentence."}
    ],
    "params": {
      "temperature": 0.7,
      "max_tokens": 150
    },
    "options": {
      "strict_params": false
    }
  }'
```

---

### Canonical Response Example

```json
{
  "id": "resp_01j7x8a9b2c3d4e5",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "model_served": "gpt-4o-mini-2024-07-18",
  "message": {
    "role": "assistant",
    "content": "Vector embeddings are numerical representations of text or data in a multi-dimensional space where semantic similarity is captured by geometric proximity."
  },
  "finish_reason": "stop",
  "usage": {
    "input_tokens": 14,
    "output_tokens": 26,
    "total_tokens": 40
  },
  "warnings": [],
  "meta": {
    "latency_ms": 385,
    "attempts": 1
  },
  "raw": null
}
```

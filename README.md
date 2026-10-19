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
- **Docker-ready**: Multi-stage, non-root Docker build and docker-compose configurations.

## Quick Start

### 1. Install Dependencies
```bash
pip install -e ".[dev]"
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
- `POST /v1/chat` - Canonical chat completion
- `GET /v1/providers` - Inspect configured providers and capabilities
- `GET /healthz` - Liveness health check
- `GET /readyz` - Readiness probe (`?deep=true` for live upstream checks)
- `GET /metrics` - Prometheus metrics

## Running Tests
```bash
pytest
```

# Unified LLM Integration Layer — Architecture

**Status**: Proposed (v1 design)  
**Audience**: Backend engineers, AI-agent developers, DevOps  
**Stack**: Python 3.12+, FastAPI, Pydantic v2, Uvicorn, HTTPX, Tenacity, Structlog, Pytest, Docker  

## Table of Contents
- A. Project Overview and Scope
- B. High-Level Architecture
- C. Layered Component Design
- D. Public API Contract
- E. Provider Adapter Abstraction
- F. Request Validation and Parameter Mapping
- G. Response Normalization
- H. Error Handling
- I. Timeouts and Bounded Retries
- J. Configuration and Credential Management
- K. Concurrency, Lifecycle, and Streaming
- L. Observability
- M. Security Considerations
- N. Project Structure
- O. Docker Deployment
- P. Testing Strategy
- Q. Extensibility: Adding a New Provider
- R. Alternative: LiteLLM
- S. Architecture Decision Records
- T. Risks, Open Questions, and Roadmap

---

## A. Project Overview and Scope
### A.1 Problem
AI agents and applications that talk to more than one LLM provider end up carrying a lot of integration baggage:
- Different SDKs, async models, and client versioning.
- Request shapes differ (`messages` vs `contents`, system role vs separate fields, `max_tokens` vs `max_output_tokens` vs `num_predict`).
- Response shapes, finish reasons, and token usage differ.
- Authentication differs (API keys, base URLs, no auth for local Ollama).
- Errors differ in type, status code, and retry semantics.

### A.2 Solution
Unified LLM Integration Layer (ULIL) is a containerized middleware service exposing one stable HTTP API. The caller states which provider and model to use, plus messages and optional generation parameters. ULIL:
1. Validates the request against a canonical schema.
2. Selects the adapter for the requested provider.
3. Maps canonical parameters to the provider's native format.
4. Executes the call through the official provider SDK with a bounded timeout and bounded retries.
5. Normalizes the response (content, finish reason, usage).
6. Converts any failure into a standardized error envelope.

### A.3 Architectural Goals
- **G1**: Standardized request/response interface (Canonical Pydantic models).
- **G2**: Independent provider adapters (`ProviderAdapter` contract, one module per provider).
- **G3**: Centralized configuration and credentials (`pydantic-settings` + optional YAML).
- **G4**: Validation and parameter mapping (Pre-flight validation against adapter capabilities).
- **G5**: Consistent response normalization (Per-adapter normalizers into one response model).
- **G6**: Standardized errors (Error taxonomy with a single wire format).
- **G7**: Configurable timeouts, bounded retries (Service-level deadline and Tenacity policy).
- **G8**: Extensibility without core changes (Registry and entry points).
- **G9**: Docker-based deployment (Multi-stage image plus Compose).
- **G10**: Testability and maintainability (Layered design, shared adapter contract tests).

---

## B. High-Level Architecture & Principles
- **Thin core, fat edges**: The core knows nothing about OpenAI, Gemini, etc. Everything provider-specific lives in an adapter.
- **Adapters translate; the service governs**: Adapters do mapping, SDK calls, normalization, and error translation. The service layer owns validation, retries, deadlines, and logging.
- **Explicit over magic**: Unsupported parameters are reported (as warnings or errors), never silently dropped.
- **One retry layer**: SDK-internal retries are disabled so total attempts and latency are predictable and observable.
- **Dependency direction**: `api` -> `core` -> `providers`. The core depends on an abstract interface, never on concrete adapters.

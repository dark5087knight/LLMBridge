# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --prefix=/install .

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN useradd --create-home --uid 10001 ulil

COPY --from=builder /install /usr/local

USER ulil
WORKDIR /app

COPY config/ /app/config/

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "unified_llm.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]

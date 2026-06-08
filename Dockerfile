FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.7.22 /uv /uvx /bin/

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src

RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "chatbot_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]

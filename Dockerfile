# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Pin tool versions in images: "latest" means a different image every build.
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /bin/uv

WORKDIR /app

# Layer 1: dependencies. Only rebuilt when pyproject.toml / uv.lock change,
# so code-only changes rebuild in seconds instead of minutes.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# Layer 2: application code (changes often, so it goes last).
COPY app ./app

# Don't run as root inside the container.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

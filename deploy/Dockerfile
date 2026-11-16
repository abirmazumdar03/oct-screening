
# Build from the repository root:
# docker build -f deploy/Dockerfile -t oct-screen .
# docker build -f deploy/Dockerfile --build-arg WITH_HEATMAP=0 -t oct-screen:lite .

FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ARG WITH_HEATMAP=1

ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# 1. Install dependencies only (cached layer).
COPY pyproject.toml uv.lock .python-version ./

RUN G="--group app"; \
    if [ "$WITH_HEATMAP" = "1" ]; then G="$G --group app-heatmap"; fi; \
    uv sync --frozen --no-dev --no-install-project $G

# 2. Copy application code and configuration.
COPY octscreen ./octscreen
COPY configs ./configs

RUN G="--group app"; \
    if [ "$WITH_HEATMAP" = "1" ]; then G="$G --group app-heatmap"; fi; \
    uv sync --frozen --no-dev $G

ENV PATH="/app/.venv/bin:$PATH" \
    OCT_HOST=0.0.0.0 \
    OCT_ENABLE_HEATMAP=${WITH_HEATMAP} \
    OCT_MODEL_DIR=/app/model_package \
    GRADIO_ANALYTICS_ENABLED=False

EXPOSE 7860

CMD ["python", "-m", "octscreen.app"]

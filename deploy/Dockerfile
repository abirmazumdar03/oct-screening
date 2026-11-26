# deploy/Dockerfile: build from the REPO ROOT:
#   docker build -f deploy/Dockerfile -t oct-screen .                                   (with heatmaps)
#   docker build -f deploy/Dockerfile --build-arg WITH_HEATMAP=0 -t oct-screen:lite .   (no torch, for 512 MB hosts)
FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
# If a host can't pull ghcr.io, replace the line above with:  RUN pip install uv
ARG WITH_HEATMAP=1
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# 1) dependencies only (cached layer)
COPY pyproject.toml uv.lock .python-version ./
RUN G="--group app"; if [ "$WITH_HEATMAP" = "1" ]; then G="$G --group app-heatmap"; fi; \
    uv sync --frozen --no-dev --no-install-project $G

# 2) our code
COPY octscreen ./octscreen
COPY configs ./configs
RUN G="--group app"; if [ "$WITH_HEATMAP" = "1" ]; then G="$G --group app-heatmap"; fi; \
    uv sync --frozen --no-dev $G

ENV PATH="/app/.venv/bin:$PATH" \
    OCT_HOST=0.0.0.0 \
    OCT_ENABLE_HEATMAP=${WITH_HEATMAP} \
    OCT_MODEL_DIR=/app/model_package \
    GRADIO_ANALYTICS_ENABLED=False \
    PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["python", "-m", "octscreen.app"]

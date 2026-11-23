FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_NO_SYNC=1 \
    UV_PYTHON_DOWNLOADS=0 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/program \
    PYTHONUNBUFFERED=1

WORKDIR /opt/program

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --group ml --no-install-project

COPY --chmod=755 serve.sh /usr/local/bin/serve
COPY solar_predictor solar_predictor

FROM python:3.12.8-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

COPY .python-version pyproject.toml uv.lock .env ./

RUN uv sync

COPY src src
COPY assets assets

CMD ["uv", "run", "src/main.py"]

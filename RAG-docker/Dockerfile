FROM python:3.12-slim

ARG CUDA_VISIBLE_DEVICES="0"
ARG CUDA_DEVICE_ORDER="PCI_BUS_ID"
ARG GRADIO_SERVER_PORT=7860

ENV PIP_DEFAULT_TIMEOUT=100 \
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER} \
    GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}

EXPOSE ${GRADIO_SERVER_PORT}

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create true

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-dev

COPY src src
COPY assets assets

CMD ["poetry", "run", "python", "src/main.py"]

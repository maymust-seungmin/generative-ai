FROM python:3.12-slim

ENV PIP_DEFAULT_TIMEOUT=100
ARG CUDA_VISIBLE_DEVICES="0"
ENV CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
ARG CUDA_DEVICE_ORDER="PCI_BUS_ID"
ENV CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER}
ARG GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}
EXPOSE ${GRADIO_SERVER_PORT}

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

# 3. Poetry 설치 및 환경설정
RUN pip install --no-cache-dir poetry && poetry config virtualenvs.create true

# 4. pyproject.toml 및 lock 파일 복사 + 종속성 설치
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-dev

COPY src app
COPY assets app

# Populate local environment variables
COPY .env .
# RUN source .env && printenv # To debug, uncomment this line to see env vars in the build logs
# ENV $(cat .env)

CMD ["python", "/app/src/main.py"]


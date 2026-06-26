FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --index-url ${TORCH_INDEX_URL} torch==2.8.0 torchvision==0.23.0 && \
    python -m pip install -e ".[dev,notebook]" && \
    python -m pip install jupyterlab

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=maxima"]
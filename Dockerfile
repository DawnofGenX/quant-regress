FROM python:3.12-slim

# Install torch CPU-only wheels: ~200 MB instead of ~2.5 GB with CUDA.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      --index-url "${TORCH_INDEX}" \
      torch>=2.0

RUN pip install --no-cache-dir transformers>=4.40 numpy>=1.24

WORKDIR /action
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

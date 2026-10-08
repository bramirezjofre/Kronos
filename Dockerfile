FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface

WORKDIR /app
COPY requirements.txt ./
# Use the CPU wheel explicitly; the default PyPI torch package pulls large CUDA
# runtimes into this CPU-oriented image.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.0.0" \
    && grep -v '^torch' requirements.txt > /tmp/requirements-no-torch.txt \
    && pip install --no-cache-dir -r /tmp/requirements-no-torch.txt
COPY . .

# Safe, offline-by-default validation; it does not download checkpoints.
ENTRYPOINT ["python"]
CMD ["scripts/kronos_service.py"]

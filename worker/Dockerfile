# GPU worker — build context = folder "cluster/" (this package root):
#   docker build -f worker/Dockerfile -t tls-classify-worker .
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MANAGER_URL=http://manager:5357 \
    WORKER_MODE=gpu \
    CLASSIFY_SCRIPT=/opt/tls/pipeline/GPU_0_0_1/classify_tls_gpu_0_0_1.py \
    PYTHONPATH=/opt/tls/pipeline \
    PYTHON_BIN=python3 \
    RF_REFINER=/opt/tls/pipeline/models/etalon_rf_scene_20260909.pkl \
    RF_CONTEXT_MODE=scene \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /opt/tls

COPY pipeline/GPU_0_0_1/requirements-gpu-core.txt /tmp/requirements-gpu-core.txt
COPY pipeline/GPU_0_0_1/constraints-tested.txt /tmp/constraints-tested.txt
COPY worker/requirements.txt /tmp/requirements-worker.txt
RUN pip3 install --no-cache-dir --upgrade pip \
 && pip3 install --no-cache-dir -c /tmp/constraints-tested.txt -r /tmp/requirements-gpu-core.txt \
 && pip3 install --no-cache-dir -c /tmp/constraints-tested.txt -r /tmp/requirements-worker.txt \
 && pip3 install --no-cache-dir -c /tmp/constraints-tested.txt --extra-index-url https://pypi.nvidia.com \
      "cuml-cu12==26.2.0" "cupy-cuda12x==14.1.1"

COPY pipeline/ /opt/tls/pipeline/
COPY worker/worker_agent.py /opt/tls/worker/worker_agent.py
COPY worker/las_result.py /opt/tls/worker/las_result.py

RUN mkdir -p /mnt/storage

WORKDIR /opt/tls/worker
CMD ["python3", "worker_agent.py"]

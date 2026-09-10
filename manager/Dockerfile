# Manager: build React UI, then serve FastAPI + static
# Build context = folder "cluster/" (this package root):
#   docker build -f manager/Dockerfile -t tls-classify-manager .
FROM node:20-bookworm AS ui
WORKDIR /ui
COPY manager/frontend/package.json ./
RUN npm install
COPY manager/frontend/ ./
RUN npm run build && test -f /static/index.html

FROM python:3.11-slim-bookworm
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATABASE_URL=sqlite:////data/cluster.db \
    STORAGE_ROOTS=storage=/mnt/storage \
    MANAGER_HOST=0.0.0.0 \
    MANAGER_PORT=5357

COPY manager/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY manager/app ./app
COPY manager/run.py ./
COPY --from=ui /static ./static

RUN mkdir -p /data /mnt/storage

EXPOSE 5357
CMD ["python", "run.py"]

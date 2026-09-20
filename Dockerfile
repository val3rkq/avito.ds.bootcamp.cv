# CPU-only образ
FROM python:3.12

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY weights ./weights
COPY notebooks ./notebooks
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Данные (test/, sample_submission.csv, data/, outputs/) монтируются томами
ENTRYPOINT ["/entrypoint.sh"]

# CPU-only образ
FROM python:3.12

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY weights ./weights
COPY notebooks ./notebooks

# Данные (test/images, sample_submission.csv) монтируются томами
# По умолчанию: бейзлайн -> /app/outputs/submission.csv
CMD ["sh", "-c", "python -m src.baseline_paddle --images test/images \
     $( [ -f sample_submission.csv ] && echo --sample-submission sample_submission.csv ) \
     --out outputs/submission.csv --raw-out outputs/raw_v1_paddle.csv"]

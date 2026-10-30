FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --create-home app \
    && mkdir -p /app/data /home/app/.cache/huggingface \
    && chown -R app:app /app /home/app/.cache

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
RUN python -m pip install --timeout 120 --retries 10 -r requirements.txt

COPY --chown=app:app src ./src

USER app

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]

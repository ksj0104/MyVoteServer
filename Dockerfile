FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv/streaming

COPY requirements.txt requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && groupadd --gid 10001 streaming \
    && useradd --uid 10001 --gid streaming --no-create-home --shell /usr/sbin/nologin streaming

COPY --chown=10001:10001 app/ ./app/

USER 10001:10001
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PAISAWISE_DATA_DIR=/data PORT=8000
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend backend
COPY frontend frontend
COPY data data
COPY model model
# Retrain from the supplied dataset against this image's scikit-learn, so the pickle always matches
RUN PAISAWISE_NO_AUTOLOAD=1 python backend/train_model.py
RUN useradd -m app && mkdir -p /data && chown app /data
USER app
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT','8000'), timeout=3)"
# ONE worker: the model and the correction memory live in this process. Threads handle concurrency.
CMD ["sh", "-c", "exec gunicorn -w 1 --threads 8 --timeout 120 -b 0.0.0.0:${PORT} --chdir backend app:app"]

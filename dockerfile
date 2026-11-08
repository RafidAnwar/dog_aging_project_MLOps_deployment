FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir .

COPY asset ./asset
COPY scripts ./scripts
COPY app.py .
COPY README.md .
COPY data/final.csv ./data/final.csv
COPY artifacts/cslb_model.joblib ./artifacts/cslb_model.joblib

ENV PYTHONPATH=/app/src
ENV DAPDATAPATH=/app/data/final.csv
ENV CSLBMODELPATH=/app/artifacts/cslbmodel.joblib

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
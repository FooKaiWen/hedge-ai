FROM python:3.11-slim-bookworm

WORKDIR /app

COPY webapp/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY webapp/app.py /app/webapp/app.py
COPY data/palm_oil_enriched_data.csv /app/data/palm_oil_enriched_data.csv
COPY data/decision_history.json /app/data/decision_history.json

WORKDIR /app/webapp
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]

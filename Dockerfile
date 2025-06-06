FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY src/requirements.txt /opt/ml/code/requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/code/requirements.txt

ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/ml/code:${PATH}"

COPY src/train.py /opt/ml/code/train.py
COPY src/inference.py /opt/ml/code/inference.py

WORKDIR /opt/ml/code

ENTRYPOINT ["python", "train.py"]
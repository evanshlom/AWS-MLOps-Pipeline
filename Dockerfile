FROM python:3.9-slim

WORKDIR /opt/ml

COPY src/requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./code/
ENV PYTHONPATH=/opt/ml/code

EXPOSE 8080

ENTRYPOINT ["python", "code/inference.py"]
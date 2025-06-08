FROM python:3.9-slim

WORKDIR /app

# Install dependencies for testing
RUN pip install boto3 numpy

# Copy test script from scripts folder
COPY scripts/test_endpoint.py .

# Set default command
CMD ["python", "local/test_endpoint.py"]
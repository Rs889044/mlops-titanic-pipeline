# Use a Python version that matches your local training environment
FROM python:3.13-slim

# Install the latest available LTS Java version (OpenJDK 21)
RUN apt-get update && \
    apt-get install -y openjdk-21-jre-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Update JAVA_HOME to point to Java 21

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-arm64
ENV PATH=$PATH:$JAVA_HOME/bin

# Proceed with the Python application setup as before
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
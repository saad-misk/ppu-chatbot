FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps
COPY nlp_engine/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Copy source
COPY . .

EXPOSE 8001
CMD ["uvicorn", "nlp_engine.nlp_server:app", "--host", "0.0.0.0", "--port", "8001"]

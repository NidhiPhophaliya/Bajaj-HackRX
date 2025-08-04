FROM python:3.10-slim-bullseye

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y git && apt-get upgrade -y && apt-get clean

# Install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source
COPY . .


ENV PORT=7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]

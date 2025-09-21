FROM python:3.10-slim


# Install system dependencies
RUN apt-get update && apt-get install -y \
    git wget curl libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install torch torchvision fastapi uvicorn pillow python-multipart

# Set working directory
WORKDIR /app
COPY . /app

# Expose port
EXPOSE 7860

# Run FastAPI with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

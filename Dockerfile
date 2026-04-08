# Start from an official Python image (slim = smaller size)
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy and install dependencies first
# (Docker caches this layer — only re-runs if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY api/ ./api/
COPY mlruns/ ./mlruns/

# Expose the port the server runs on
EXPOSE 8000

# Start the FastAPI server when the container runs
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
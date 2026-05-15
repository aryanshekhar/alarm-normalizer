FROM python:3.12-slim

WORKDIR /app

# Install torch CPU-only in an isolated layer so the ~700 MB wheel is cached
# independently of requirements.txt changes.
# To use CUDA instead, replace the index URL with e.g.:
#   https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir \
        torch \
        --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (pip skips torch — already satisfied above)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and bundled model weights
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

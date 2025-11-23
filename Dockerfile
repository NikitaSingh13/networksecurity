# -----------------------
# 1. Base Image
# -----------------------
FROM python:3.10-slim

# -----------------------
# 2. Create working directory
# -----------------------
WORKDIR /app

# -----------------------
# 3. Copy requirements first
# -----------------------
COPY requirements.txt .

RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------
# 4. Copy entire project
# -----------------------
COPY . .

# -----------------------
# 5. Expose port for uvicorn
# -----------------------
EXPOSE 8000

# -----------------------
# 6. Run the FastAPI app
# IMPORTANT: app.py is in ROOT → so entry = "app:app"
# -----------------------
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

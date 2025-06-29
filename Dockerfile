# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project code
COPY . .

# Default command for the container (you can override this in docker-compose)
CMD ["streamlit", "run", "frontend/feedback_app.py"]

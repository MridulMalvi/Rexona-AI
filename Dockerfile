FROM python:3.9-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "frontend.py", "--server.address=0.0.0.0"]

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PySheaf directly from the repository
RUN git clone https://github.com/kb1dds/pysheaf.git /tmp/pysheaf \
    && cp -r /tmp/pysheaf/*.py /usr/local/lib/python3.10/site-packages/ \
    && rm -rf /tmp/pysheaf

# Copy the application code
COPY . .

# Create directory for visualizations and ensure proper permissions
RUN mkdir -p visualizations && chmod 777 visualizations

# Expose the port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
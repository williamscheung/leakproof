FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd -m appuser && chown -R appuser /app
USER appuser
COPY . .
EXPOSE 5000
CMD ["python", "api_service.py"]
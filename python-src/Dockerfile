FROM python:3.15.2-alpine3.24
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt && python -m pip uninstall -y pip
COPY ./app /app/app
COPY ./data /app/data
CMD ["gunicorn", "--chdir", "/app/app", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "main:create_app()"]

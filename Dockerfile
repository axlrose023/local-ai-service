FROM python:3.12-slim

WORKDIR /app

# Системні залежності
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python залежності
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код додатку
COPY . .

# Entrypoint скрипт
RUN chmod +x /app/entrypoint.sh

# Порт Chainlit
EXPOSE 8000

# Запуск через entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

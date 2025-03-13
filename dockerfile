FROM python:3.10-slim

WORKDIR /app

# 1. Instalar dependências de sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    g++ \
    gcc \
    make \
    cmake \
    python3-dev \
    libsm6 \
    libxext6 \
    libxrender-dev

# 2. Copiar requirements primeiro (otimização de cache do Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copiar o restante do código
COPY . .

# 4. Criar diretório para rostos
RUN mkdir -p img_dbv

EXPOSE 80

CMD ["uvicorn", "main:api", "--host", "0.0.0.0", "--port", "80"]
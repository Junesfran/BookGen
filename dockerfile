# 1. Usamos una imagen de Python ligera basada en Debian
FROM python:3.11-slim

# 2. Evitamos que Python genere archivos .pyc y forzamos que los logs salgan en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Instalamos dependencias del sistema para compilación y audio
# En Linux, esto reemplaza a las "Build Tools" de Visual Studio
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    ffmpeg \
    libsndfile1 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 4. Establecemos el directorio de trabajo
WORKDIR /app

# 5. Instalamos las dependencias de Python
# Se hace en dos pasos para aprovechar la caché de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copiamos el resto de los archivos del proyecto
COPY . .

# 7. Creamos una carpeta para los modelos para que no se descarguen cada vez
RUN mkdir -p /root/.local/share/tts

# 8. Comando para ejecutar tu aplicación
# Asegúrate de que el nombre del archivo sea el correcto
CMD ["python", "app.py"]
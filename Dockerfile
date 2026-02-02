# Usar imagen oficial ligera de Python 3.10
# (Slim es mejor que Alpine para librerías de data science como pandas/numpy)
FROM python:3.10-slim

# Evitar que Python genere archivos .pyc y buffer de salida
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias del sistema si fueran necesarias para compilar
# RUN apt-get update && apt-get install -y build-essential gcc

# Copiar requirements primero para aprovechar caché de docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Dar permisos de ejecución al script de inicio
RUN chmod +x start.sh

# Exponer el puerto (Cloud Run default: 8080)
EXPOSE 8080

# Usar el script como comando de inicio
CMD ["./start.sh"]

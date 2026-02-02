#!/bin/bash

echo "🚀 Iniciando proceso de arranque..."

# 1. Ejecutar Ingestión (Crear Index)
# Cloud Run tiene un sistema de archivos efímero. Si el contenedor se reinicia, 
# el índice borrado se recrea aquí.
echo "📚 Verificando/Creando índice vectorial con ingest.py..."
python ingest.py

if [ $? -ne 0 ]; then
    echo "❌ Error en ingestión. Abortando inicio."
    exit 1
fi

echo "✅ Ingestión completada."

# 2. Iniciar Servidor
echo "🔥 Iniciando Uvicorn en puerto $PORT..."
# Cloud Run inyecta la variable $PORT automáticamente (default 8080)
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}

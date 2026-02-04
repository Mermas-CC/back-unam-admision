#!/bin/bash

echo "🚀 Iniciando proceso de arranque..."

# 1. Ejecutar Ingestión (Opcional - Ahora se maneja via Admin API o pre-generado)
# Se ha eliminado la ejecución automática para acelerar el arranque.
echo "ℹ️ Ingestión automática desactivada. Usa el Admin API para procesar archivos."

echo "✅ Ingestión completada."

# 2. Iniciar Servidor
echo "🔥 Iniciando Uvicorn en puerto $PORT..."
# Cloud Run inyecta la variable $PORT automáticamente (default 8080)
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}

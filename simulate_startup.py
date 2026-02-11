import os
import time
import asyncio
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.llms.gemini import Gemini
from llama_index.core.embeddings import BaseEmbedding
import google.generativeai as genai
from typing import List

# --- SETUP REPLICA FROM main.py ---
load_dotenv()
PERSIST_DIR = "./chroma_db"

embedding_cache = {}

class GeminiEmbedding(BaseEmbedding):
    def __init__(self, model: str = "models/gemini-embedding-001"):
        super().__init__()
        self._model = model
    def _get_query_embedding(self, query: str) -> List[float]:
        return [0.0] * 768 # Dummy for simulation speed
    def _get_text_embedding(self, text: str) -> List[float]:
        return [0.0] * 768 # Dummy for simulation speed
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return [0.0] * 768
    async def _aget_text_embedding(self, text: str) -> List[float]:
        return [0.0] * 768

def simulate():
    start_total = time.time()
    
    print("🚀 Iniciando Simulación de Arranque Backend...")
    
    # 1. Cargar Entorno
    t0 = time.time()
    load_dotenv()
    t_env = time.time() - t0
    print(f"⏱️  Carga de .env: {t_env:.4f}s")
    
    # 2. Configurar Modelos (LlamaIndex Settings)
    t1 = time.time()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    Settings.llm = Gemini(model="models/gemini-2.0-flash", max_output_tokens=1024)
    Settings.embed_model = GeminiEmbedding(model="models/gemini-embedding-001")
    t_models = time.time() - t1
    print(f"⏱️  Configuración de modelos: {t_models:.4f}s")
    
    # 3. Cargar Índice (ChromaDB)
    t2 = time.time()
    if not os.path.exists(PERSIST_DIR):
        print("❌ Error: No se encontró chroma_db.")
        return
        
    db = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = db.get_or_create_collection("admision_unap")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=Settings.embed_model
    )
    t_index = time.time() - t2
    print(f"⏱️  Carga de índice vectorial ({chroma_collection.count()} chunks): {t_index:.4f}s")
    
    total_time = time.time() - start_total
    print(f"\n✅ REPORTE FINAL: El backend se levanta en {total_time:.2f} segundos.")
    print(f"   (Con data preprocesada en {PERSIST_DIR})")

if __name__ == "__main__":
    simulate()

import os
import asyncio
from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.llms.gemini import Gemini
import google.generativeai as genai
from llama_index.core.embeddings import BaseEmbedding
from typing import List
import shutil

# --- CARGAR VARIABLES DE ENTORNO ---
load_dotenv()

if not os.getenv("GOOGLE_API_KEY") or not os.getenv("GEMINI_API_KEY"):
    print("❌ Error: Faltan variables de entorno (GOOGLE_API_KEY o GEMINI_API_KEY).")
    # No exit() here so it can be imported safely even if envs are missing (though it will fail later)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- CLASES DE UTILIDAD (Replicada de main.py para independencia) ---
class GeminiEmbedding(BaseEmbedding):
    def __init__(self, model: str = "models/embedding-001"):
        super().__init__()
        self._model = model

    def _get_query_embedding(self, query: str) -> List[float]:
        result = genai.embed_content(model=self._model, content=query)
        return result["embedding"]

    def _get_text_embedding(self, text: str) -> List[float]:
        result = genai.embed_content(model=self._model, content=text)
        return result["embedding"]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

# --- CONFIGURACIÓN ---
Settings.llm = Gemini(model="models/gemini-2.0-flash", max_output_tokens=1024)
Settings.embed_model = GeminiEmbedding(model="models/embedding-001")
PERSIST_DIR = "./.chroma_db_v2"
DATA_DIR = "./data"

def ingest_documents(force_rebuild=False):
    print(f"🚀 Iniciando proceso de ingestión (Force Rebuild: {force_rebuild})...")
    
    # 1. Cargar documentos (Dinámico: escanea la carpeta data)
    documentos_raw = []
    if not os.path.exists(DATA_DIR):
        print(f"⚠️  Directorio {DATA_DIR} no existe. Creando...")
        os.makedirs(DATA_DIR)
        
    archivos = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt") or f.endswith(".pdf")]
    
    for archivo in archivos:
        path = os.path.join(DATA_DIR, archivo)
        try:
            print(f"📄 Procesando: {archivo}")
            # TODO: Mejorar loader para PDF si es necesario. Por ahora asumimos texto.
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                documentos_raw.append(Document(text=text, metadata={"filename": archivo}))
        except Exception as e:
            print(f"❌ Error leyendo {archivo}: {e}")

    if not documentos_raw:
        print("❌ No hay documentos para procesar en ./data.")
        return

    # 2. Procesar y dividir (Chunking)
    print("✂️  Dividiendo documentos (Sentence Splitter)...")
    from llama_index.core.node_parser import SentenceSplitter
    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    nodes = splitter.get_nodes_from_documents(documentos_raw)
    print(f"✅ Se generaron {len(nodes)} chunks.")

    # 3. Indexar en ChromaDB
    print(f"💾 Guardando en ChromaDB ({PERSIST_DIR})...")
    
    if force_rebuild and os.path.exists(PERSIST_DIR):
         print("♻️ Rebuild forzado: Eliminando base de datos antigua...")
         shutil.rmtree(PERSIST_DIR)

    try:
        db = chromadb.PersistentClient(path=PERSIST_DIR)
        chroma_collection = db.get_or_create_collection("admision_unap")
    except Exception as e:
        print(f"⚠️ Error conectando a ChromaDB ({e}).")
        print("💥 Intentando recuperar regenerando desde cero...")
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        db = chromadb.PersistentClient(path=PERSIST_DIR)
        chroma_collection = db.get_or_create_collection("admision_unap")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Crea el índice y persiste usando from_documents para mayor confiabilidad
    print("🧠 Creando índice vectorial (esto puede tardar unos segundos)...")
    index = VectorStoreIndex.from_documents(
        documentos_raw, 
        storage_context=storage_context,
        node_parser=splitter,
        show_progress=True
    )
    
    # Verificación final
    final_count = chroma_collection.count()
    print(f"✅ Ingestión completada. Nodos en ChromaDB: {final_count}")
    if final_count == 0:
        print("❌ Error crítico: La base de datos sigue vacía después de la ingestión.")
    else:
        print("🚀 Índice actualizado exitosamente.")

if __name__ == "__main__":
    # Por defecto, si se ejecuta directo, hacemos rebuild si no existe, o append?
    # Para consistencia con deployment, mejor rebuild si se llama manual para asegurar estado limpio.
    ingest_documents(force_rebuild=True)

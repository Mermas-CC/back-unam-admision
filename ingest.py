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

# --- CARGAR VARIABLES DE ENTORNO ---
load_dotenv()

if not os.getenv("GOOGLE_API_KEY") or not os.getenv("GEMINI_API_KEY"):
    print("❌ Error: Faltan variables de entorno (GOOGLE_API_KEY o GEMINI_API_KEY).")
    exit()

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
Settings.llm = Gemini(model="gemini-2.5-flash", max_output_tokens=1024)
Settings.embed_model = GeminiEmbedding(model="models/embedding-001")
PERSIST_DIR = "./chroma_db"
DATA_DIR = "./data"

def ingest_documents():
    print("🚀 Iniciando proceso de ingestión...")
    
    # 1. Cargar documentos específicos
    documentos_raw = []
    archivos = ["REGLAMENTO_ADMISION.txt", "PROSPECTO_ADMISION.txt"]
    
    for archivo in archivos:
        path = os.path.join(DATA_DIR, archivo)
        try:
            if os.path.exists(path):
                print(f"📄 Procesando: {archivo}")
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                    documentos_raw.append(Document(text=text, metadata={"filename": archivo}))
            else:
                print(f"⚠️  Advertencia: No se encontró {archivo}")
        except Exception as e:
            print(f"❌ Error leyendo {archivo}: {e}")

    if not documentos_raw:
        print("❌ No hay documentos para procesar.")
        return

    # 2. Procesar y dividir (Chunking)
    print("✂️  Dividiendo documentos (Semantic Chunking)...")
    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=Settings.embed_model
    )
    nodes = splitter.get_nodes_from_documents(documentos_raw)
    print(f"✅ Se generaron {len(nodes)} chunks semánticos.")

    # 3. Indexar en ChromaDB
    print(f"💾 Guardando en ChromaDB ({PERSIST_DIR})...")
    
    # Resetear/Crear nueva colección
    db = chromadb.PersistentClient(path=PERSIST_DIR)
    
    # Opcional: Borrar colección anterior para asegurar limpieza
    try:
        db.delete_collection("admision_unap")
        print("♻️  Colección anterior eliminada.")
    except:
        pass

    chroma_collection = db.get_or_create_collection("admision_unap")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    print("✅ Ingestión completada exitosamente. Índice listo para main.py")

if __name__ == "__main__":
    ingest_documents()

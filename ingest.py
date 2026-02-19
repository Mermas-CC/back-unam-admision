import os
import asyncio
import numpy as np
import shutil
from typing import List

from dotenv import load_dotenv
from tqdm import tqdm

import chromadb
import google.generativeai as genai

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import TextNode
from llama_index.core.embeddings import BaseEmbedding
from llama_index.llms.gemini import Gemini


# --- CARGAR VARIABLES DE ENTORNO ---
load_dotenv()

if not os.getenv("GOOGLE_API_KEY") or not os.getenv("GEMINI_API_KEY"):
    print("❌ Error: Faltan variables de entorno (GOOGLE_API_KEY o GEMINI_API_KEY).")
    # No exit() here so it can be imported safely even if envs are missing (though it will fail later)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- CLASES DE UTILIDAD (Replicada de main.py para independencia) ---
class GeminiEmbedding(BaseEmbedding):
    def __init__(self, model: str = "models/gemini-embedding-001"):
        super().__init__()
        self._model = model

    async def _aget_text_embedding(self, text: str) -> List[float]:
        # Implementación asíncrona real con reintentos
        import time
        max_retries = 5
        for attempt in range(max_retries):
            try:
                result = genai.embed_content(model=self._model, content=text)
                return result["embedding"]
            except Exception as e:
                err_str = str(e)
                if "429" in err_str and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + 1
                    print(f"⚠️ 429 Rate Limit (Embed). Esperando {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                raise e

    def _get_text_embedding(self, text: str) -> List[float]:
        # Para compatibilidad con llamadas síncronas
        import time
        result = genai.embed_content(model=self._model, content=text)
        time.sleep(0.1) # Pequeño delay para no saturar si es síncrono
        return result["embedding"]

    async def _aget_text_embedding_batch(self, texts: List[str], **kwargs) -> List[List[float]]:
        import sys
        from tqdm.asyncio import tqdm
        show_progress = kwargs.get("show_progress", True)
        
        semaphore = asyncio.Semaphore(20) # Límite de concurrencia optimizado
        
        async def sem_embed(text):
            async with semaphore:
                return await self._aget_text_embedding(text)

        pbar = tqdm(total=len(texts), desc="🧠 Embeddings Semánticos", disable=not show_progress, file=sys.stdout, dynamic_ncols=True)
        
        tasks = [sem_embed(text) for text in texts]
        all_embeddings = []
        
        # Procesamos en chunks para mantener la barra de progreso fluida
        chunk_size = 50
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i:i + chunk_size]
            res = await asyncio.gather(*chunk)
            all_embeddings.extend(res)
            pbar.update(len(chunk))
            sys.stdout.flush()
            
        pbar.close()
        return all_embeddings

    def _get_text_embedding_batch(self, texts: List[str], **kwargs) -> List[List[float]]:
        # Envoltorio síncrono para batching
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Si el loop ya corre, usamos un executor para no bloquear
            return loop.run_until_complete(self._aget_text_embedding_batch(texts, **kwargs))
        return asyncio.run(self._aget_text_embedding_batch(texts, **kwargs))

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await self._aget_text_embedding(query)

# --- CONFIGURACIÓN ---
Settings.llm = Gemini(model="models/gemini-2.0-flash", max_output_tokens=1024)
Settings.embed_model = GeminiEmbedding(model="models/gemini-embedding-001")
PERSIST_DIR = "./chroma_db_v2"
DATA_DIR = "./data"

async def semantic_chunking_fast(
    documentos: List[Document], 
    embed_model: GeminiEmbedding, 
    buffer_size: int = 1, 
    similarity_threshold: float = 0.80,
    min_chunk_chars: int = 350,
    max_chunk_chars: int = 1500
):
    """
    Divide documentos basándose en saltos de similitud semántica.
    Usa un 'buffer' de oraciones para suavizar los embeddings y evitar fragmentación excesiva.
    """
    from llama_index.core.node_parser.text.utils import split_by_sentence_tokenizer
    sentence_splitter = split_by_sentence_tokenizer()
    all_nodes = []
    
    for doc_idx, doc in enumerate(documentos, 1):
        filename = doc.metadata.get('filename', 'unknown')
        print(f"\n📝 Procesando ({doc_idx}/{len(documentos)}): {filename}")
        sentences = sentence_splitter(doc.text)
        if len(sentences) <= 1: 
            if sentences:
                all_nodes.append(TextNode(text=sentences[0], metadata=doc.metadata))
            continue
        
        # 1. Crear 'puntos combinados' para suavizar el contexto
        # Cada punto i es la unión de oraciones de [i - buffer_size] a [i + buffer_size]
        combined_sentences = []
        for i in range(len(sentences)):
            start = max(0, i - buffer_size)
            end = min(len(sentences), i + buffer_size + 1)
            combined_sentences.append(" ".join(sentences[start:end]))
        
        print(f"   -> {len(sentences)} oraciones ({len(combined_sentences)} puntos de análisis).")
        print(f"   -> Calculando embeddings de puntos...")
        
        embeddings = await embed_model._aget_text_embedding_batch(combined_sentences)
        embeddings_np = [np.array(e) for e in embeddings]
        
        # 2. Calcular distancias entre puntos adyacentes
        distances = []
        for i in range(len(embeddings_np) - 1):
            vec_a, vec_b = embeddings_np[i], embeddings_np[i+1]
            norm_a, norm_b = np.linalg.norm(vec_a), np.linalg.norm(vec_b)
            # Similitud del coseno
            sim = np.dot(vec_a, vec_b) / (norm_a * norm_b) if (norm_a > 0 and norm_b > 0) else 0
            distances.append(sim)
        
        # 3. Determinar puntos de corte
        splits = [0]
        for i, sim in enumerate(distances):
            if sim < similarity_threshold:
                splits.append(i + 1)
        splits.append(len(sentences))
        
        # 4. Generar chunks iniciales (por grupo de oraciones entre splits)
        initial_sentence_groups = []
        for i in range(len(splits) - 1):
            start, end = splits[i], splits[i+1]
            initial_sentence_groups.append(sentences[start:end])
            
        # 5. Post-procesamiento: Construir chunks respetando min/max tamaños
        doc_nodes = []
        current_sentences = []
        current_len = 0
        
        def flush_chunk():
            nonlocal current_sentences, current_len
            if current_sentences:
                chunk_text = " ".join(current_sentences).strip()
                if len(chunk_text) > 40:  # filtro de ruido
                    doc_nodes.append(TextNode(text=chunk_text, metadata=doc.metadata))
            current_sentences = []
            current_len = 0
        
        for group in initial_sentence_groups:
            for sent in group:
                sent_len = len(sent)
                
                # Si agregar esta oración excedería el máximo y ya tenemos contenido suficiente
                if current_len + sent_len > max_chunk_chars and current_len >= min_chunk_chars:
                    flush_chunk()
                
                current_sentences.append(sent)
                current_len += sent_len
                
                # Si ya excedimos el máximo (incluso sin contenido previo), forzar corte
                if current_len > max_chunk_chars:
                    flush_chunk()
            
            # Al terminar un grupo semántico, evaluar si el chunk actual ya es suficiente
            if current_len >= min_chunk_chars:
                flush_chunk()
        
        # Flush del último chunk pendiente
        flush_chunk()
            
        print(f"   ✅ {len(doc_nodes)} chunks generados (Optimizado).")
        all_nodes.extend(doc_nodes)
        
    return all_nodes

async def ingest_documents(force_rebuild: bool = False):
    """
    Carga documentos, los divide semánticamente y los indexa en ChromaDB.
    """
    from llama_index.core.node_parser import SentenceSplitter
    
    if force_rebuild and os.path.exists(PERSIST_DIR):
        print(f"🧹 Forzando reconstrucción. Eliminando {PERSIST_DIR}...")
        import shutil
        shutil.rmtree(PERSIST_DIR)

    # Configurar GlobalSettings
    embed_model = GeminiEmbedding()
    Settings.embed_model = embed_model
    
    # 1. Cargar documentos
    print(f"📂 Cargando documentos desde {DATA_DIR}...")
    documentos = []
    if not os.path.exists(DATA_DIR):
        print(f"⚠️  Directorio {DATA_DIR} no existe. Creando...")
        os.makedirs(DATA_DIR)
        
    archivos = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt") or f.endswith(".pdf")]
    
    for archivo in tqdm(archivos, desc="📄 Leyendo archivos"):
        path = os.path.join(DATA_DIR, archivo)
        try:
            # TODO: Mejorar loader para PDF si es necesario. Por ahora asumimos texto.
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                documentos.append(Document(text=text, metadata={"filename": archivo}))
        except Exception as e:
            print(f"❌ Error leyendo {archivo}: {e}")
    
    if not documentos:
        print("⚠️ No hay documentos para procesar.")
        return

    # 2. División de texto (CHUNK Semántico)
    print(f"🧠 Iniciando Semantic Chunking en {len(documentos)} documentos...")
    
    # --- LOGICA SEMÁNTICA (ACTIVA) ---
    nodes = await semantic_chunking_fast(documentos, embed_model)
    
    # --- LOGICA FIJA (COMENTADA) ---
    # splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    # nodes = splitter.get_nodes_from_documents(documentos)
    
    print(f"✨ Total de nodos generados: {len(nodes)}")

    # 3. Indexar en ChromaDB
    print(f"💾 Guardando en ChromaDB ({PERSIST_DIR})...")
    
    try:
        db = chromadb.PersistentClient(path=PERSIST_DIR)
        chroma_collection = db.get_or_create_collection("admision_unap")
    except Exception as e:
        print(f"⚠️ Error conectando a ChromaDB ({e}).")
        print("💥 Intentando recuperar regenerando desde cero...")
        if os.path.exists(PERSIST_DIR):
            import shutil
            shutil.rmtree(PERSIST_DIR)
        db = chromadb.PersistentClient(path=PERSIST_DIR)
        chroma_collection = db.get_or_create_collection("admision_unap")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Crea el índice y persiste usando from_documents para mayor confiabilidad
    print("🧠 Creando índice vectorial (esto puede tardar unos segundos)...")
    index = VectorStoreIndex(
        nodes, 
        storage_context=storage_context,
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
    asyncio.run(ingest_documents(force_rebuild=True))

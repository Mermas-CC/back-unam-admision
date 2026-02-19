import os
import asyncio
from typing import List
import numpy as np
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
import sys

from llama_index.core import Document, Settings
from llama_index.core.node_parser.text.utils import split_by_sentence_tokenizer
from llama_index.core.schema import TextNode
from llama_index.core.embeddings import BaseEmbedding
import google.generativeai as genai

# --- CARGAR VARIABLES DE ENTORNO ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class GeminiEmbedding(BaseEmbedding):
    def __init__(self, model: str = "models/gemini-embedding-001"):
        super().__init__()
        self._model = model

    async def _aget_text_embedding(self, text: str) -> List[float]:
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
        # Fallback sync
        result = genai.embed_content(model=self._model, content=text)
        return result["embedding"]

    async def _aget_text_embedding_batch(self, texts: List[str], **kwargs) -> List[List[float]]:
        import sys
        from tqdm.asyncio import tqdm
        show_progress = kwargs.get("show_progress", True)
        
        semaphore = asyncio.Semaphore(20) # Aumentado para mayor velocidad
        
        async def sem_embed(text):
            async with semaphore:
                return await self._aget_text_embedding(text)

        pbar = tqdm(total=len(texts), desc="🧠 Embeddings", disable=not show_progress, file=sys.stdout, dynamic_ncols=True)
        
        tasks = []
        for text in texts:
            tasks.append(sem_embed(text))
            
        all_embeddings = []
        # Ejecutar en pequeños lotes de tareas para actualizar la barra de progreso
        chunk_size = 50
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i:i + chunk_size]
            res = await asyncio.gather(*chunk)
            all_embeddings.extend(res)
            pbar.update(len(chunk))
            sys.stdout.flush()
            
        pbar.close()
        
        if all_embeddings:
            print(f"   -> Embeddings totales: {len(all_embeddings)} (Dim: {len(all_embeddings[0])})")
        return all_embeddings

    def _get_query_embedding(self, query: str) -> List[float]:
        # Para pruebas internas, usamos el loop si es necesario o directo
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # No podemos bloquear aquí, pero este método suele ser llamado por LlamaIndex sync
                # En este script test lo llamamos poco.
                return genai.embed_content(model=self._model, content=query)["embedding"]
            return loop.run_until_complete(self._aget_text_embedding(query))
        except:
            return genai.embed_content(model=self._model, content=query)["embedding"]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await self._aget_text_embedding(query)

async def semantic_chunking_fast(documentos: List[Document], embed_model: GeminiEmbedding, similarity_threshold: float = 0.82):
    """
    Divide documentos basándose en saltos de similitud semántica.
    """
    from llama_index.core.node_parser.text.utils import split_by_sentence_tokenizer
    sentence_splitter = split_by_sentence_tokenizer()
    all_nodes = []
    
    for doc_idx, doc in enumerate(documentos, 1):
        filename = doc.metadata.get('filename', 'unknown')
        print(f"\n📝 Procesando ({doc_idx}/{len(documentos)}): {filename}")
        sentences = sentence_splitter(doc.text)
        if not sentences: continue
        
        print(f"   -> {len(sentences)} oraciones encontradas. Calculando embeddings...")
        
        # Obtenemos los embeddings
        embeddings = await embed_model._aget_text_embedding_batch(sentences)
        
        # Aseguramos lista de arrays numpy 1D
        embeddings_np = [np.array(e) for e in embeddings]
        
        print(f"   -> Analizando límites semánticos (umbral: {similarity_threshold})...")
        splits = [0]
        # Recorremos cada par de oraciones adyacentes
        for i in range(len(embeddings_np) - 1):
            vec_a = embeddings_np[i]
            vec_b = embeddings_np[i+1]
            
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            
            if norm_a == 0 or norm_b == 0:
                sim = 0
            else:
                sim = np.dot(vec_a, vec_b) / (norm_a * norm_b)
            
            if sim < similarity_threshold:
                splits.append(i + 1)
        
        splits.append(len(sentences))
        
        # Crear los nodos
        doc_nodes = []
        for i in range(len(splits) - 1):
            start = splits[i]
            end = splits[i+1]
            chunk_text = " ".join(sentences[start:end]).strip()
            if len(chunk_text) > 40:
                doc_nodes.append(TextNode(text=chunk_text, metadata=doc.metadata))
        
        print(f"   ✅ {len(doc_nodes)} chunks generados.")
        all_nodes.extend(doc_nodes)
        
    return all_nodes

async def main():
    DATA_DIR = "./data-test"
    if not os.path.exists(DATA_DIR):
        print(f"❌ No existe la carpeta {DATA_DIR}")
        return

    archivos = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")][:2] # Solo 2 archivos para prueba rápida
    documentos = []
    for archivo in archivos:
        with open(os.path.join(DATA_DIR, archivo), "r", encoding="utf-8", errors="ignore") as f:
            documentos.append(Document(text=f.read(), metadata={"filename": archivo}))

    if not documentos:
        print("❌ No hay documentos.")
        return

    embed_model = GeminiEmbedding()
    print(f"🚀 Iniciando Prueba de Chunking Semántico en {len(documentos)} archivos...")
    
    nodes = await semantic_chunking_fast(documentos, embed_model)
    
    print(f"\n✨ RESUMEN:")
    print(f"Total de nodos generados: {len(nodes)}")
    if nodes:
        print(f"Ejemplo del primer nodo (primeros 200 caracteres):\n---")
        print(nodes[0].text[:200] + "...")

if __name__ == "__main__":
    asyncio.run(main())

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.gemini import Gemini
import google.generativeai as genai

# Cargar configuración
load_dotenv()
PERSIST_DIR = "./chroma_db_v2"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Usar GEMINI_API_KEY como en main.py
genai.configure(api_key=GEMINI_API_KEY or GOOGLE_API_KEY)

from llama_index.core.embeddings import BaseEmbedding
from typing import List

# Configurar LlamaIndex (mismas que en main.py)
Settings.llm = Gemini(model="models/gemini-2.0-flash", max_output_tokens=1024)

class GeminiEmbedding(BaseEmbedding):
    def __init__(self, model_name: str = "models/gemini-embedding-001"):
        super().__init__()
        self._model = model_name

    def _get_query_embedding(self, query: str) -> List[float]:
        result = genai.embed_content(model=self._model, content=query)
        return result["embedding"]

    def _get_text_embedding(self, text: str) -> List[float]:
        result = genai.embed_content(model=self._model, content=text)
        return result["embedding"]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

Settings.embed_model = GeminiEmbedding()

def eval_level_1_clustering():
    print("\n🧪 [NIVEL 1] Ejecutando Clustering Sandbox...")
    db = chromadb.PersistentClient(path=PERSIST_DIR)
    try:
        collection = db.get_collection("admision_unap")
    except:
        print(f"❌ Error: No se encontró la colección 'admision_unap' en {PERSIST_DIR}")
        return
    
    results = collection.get(include=['embeddings', 'metadatas', 'documents'])
    embeddings = np.array(results['embeddings'])
    metadatas = results['metadatas']
    
    if len(embeddings) < 2:
        print("❌ No hay suficientes datos para clustering.")
        return

    # Reducción de dimensión PCA (2D)
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(embeddings)

    # Identificar documentos únicos para colorear
    docs_labels = [m.get('file_name', 'unknown') for m in metadatas]
    unique_docs = list(set(docs_labels))
    
    plt.figure(figsize=(10, 7))
    for i, doc in enumerate(unique_docs):
        mask = [d == doc for d in docs_labels]
        plt.scatter(reduced_vectors[mask, 0], reduced_vectors[mask, 1], label=doc, alpha=0.7)

    plt.title("Clustering Semántico de Chunks (Nivel 1)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_path = "rag_clusters.png"
    plt.savefig(plot_path)
    print(f"✅ Visualización guardada en: {plot_path}")

def eval_level_1_similarity():
    print("\n🧪 [NIVEL 1] Calculando Similitud Intra vs Inter Documento...")
    db = chromadb.PersistentClient(path=PERSIST_DIR)
    try:
        collection = db.get_collection("admision_unap")
    except:
        return
        
    results = collection.get(include=['embeddings', 'metadatas'])
    
    embeddings = np.array(results['embeddings'])
    metadatas = results['metadatas']
    
    doc_indices = {}
    for i, m in enumerate(metadatas):
        fname = m.get('file_name', 'unknown')
        if fname not in doc_indices: doc_indices[fname] = []
        doc_indices[fname].append(i)

    intra_sims = []
    inter_sims = []

    # Similitud Coseno de todos contra todos
    sim_matrix = cosine_similarity(embeddings)

    for doc, indices in doc_indices.items():
        if len(indices) < 2: continue
        
        # Intra: Similitud entre chunks del mismo doc
        doc_sims = sim_matrix[np.ix_(indices, indices)]
        # Tomar solo el triángulo superior (sin la diagonal)
        upper_tri = doc_sims[np.triu_indices_from(doc_sims, k=1)]
        intra_sims.extend(upper_tri)
        
        # Inter: Similitud con chunks de otros docs
        other_indices = [i for i in range(len(embeddings)) if i not in indices]
        if not other_indices: continue
        
        inter_sim_values = sim_matrix[np.ix_(indices, other_indices)]
        inter_sims.extend(inter_sim_values.flatten())

    mean_intra = np.mean(intra_sims) if intra_sims else 0
    mean_inter = np.mean(inter_sims) if inter_sims else 0

    print(f"📊 Mean Intra-doc Similarity: {mean_intra:.4f}")
    print(f"📊 Mean Inter-doc Similarity: {mean_inter:.4f}")
    
    if mean_intra > mean_inter + 0.1:
        print("✅ Resultado: Buena discriminación semántica.")
    elif mean_intra > mean_inter:
        print("⚠️ Resultado: Discriminación aceptable pero mejorable.")
    else:
        print("❌ Resultado: Embeddings débiles o mala segmentación.")

async def eval_level_2_sufficiency(queries):
    print("\n🧪 [NIVEL 2] Evaluación de Suficiencia (LLM Judge)...")
    model = Settings.llm
    
    # Inicializar VectorStoreIndex para retrieval
    db = chromadb.PersistentClient(path=PERSIST_DIR)
    try:
        chroma_collection = db.get_collection("admision_unap")
    except:
        return []
        
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    query_engine = index.as_query_engine(similarity_top_k=3)

    results = []
    
    for q in queries:
        print(f"🔍 Evaluando Query: '{q}'")
        retrieval = query_engine.query(q)
        context = "\n".join([n.text for n in retrieval.source_nodes])
        
        prompt = f"""
        Actúa como un Juez de Calidad RAG.
        PREGUNTA: {q}
        CONTEXTO RECUPERADO:
        ---
        {context}
        ---
        EVALUACIÓN: ¿El contexto recuperado contiene información suficiente para responder la pregunta de forma completa? 
        Responde exclusivamente en formato JSON:
        {{
            "sufficient": true/false,
            "confidence": 0.0-1.0,
            "reason": "breve explicación"
        }}
        """
        
        try:
            response = await model.acomplete(prompt)
            clean_text = response.text.strip()
            if '```json' in clean_text:
                clean_text = clean_text.split('```json')[1].split('```')[0].strip()
            elif '```' in clean_text:
                clean_text = clean_text.split('```')[1].strip()
                
            eval_data = json.loads(clean_text)
            eval_data['query'] = q
            results.append(eval_data)
        except Exception as e:
            print(f"❌ Error parseando respuesta del juez: {e}")

    # Calcular métricas
    if results:
        passed = sum(1 for r in results if r['sufficient'])
        rate = (passed / len(results)) * 100
        print(f"📈 Retrieval Sufficiency Rate: {rate:.2f}% ({passed}/{len(results)})")
    
    return results

if __name__ == "__main__":
    import asyncio
    
    # 1. Clustering
    eval_level_1_clustering()
    
    # 2. Similarity
    eval_level_1_similarity()
    
    # 3. Sufficiency Test
    test_queries = [
        "¿Cuáles son los requisitos de admisión?",
        "¿Cuándo es el examen de admisión 2025?",
        "¿Cómo se puede presentar un reclamo sobre los resultados?",
        "¿Cuáles son los costos de inscripción?",
        "¿Qué pasa si olvido mi DNI el día del examen?",
        "hola como estas" # Control: Debe fallar en suficiencia
    ]
    
    asyncio.run(eval_level_2_sufficiency(test_queries))

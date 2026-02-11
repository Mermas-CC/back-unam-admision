import os
import json
import asyncio
import time
import httpx
import numpy as np
from typing import List
from dotenv import load_dotenv
import chromadb
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from llama_index.core.embeddings import BaseEmbedding
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity

# --- 0. CONFIGURACIÓN ---
load_dotenv()
TEST_JSON_PATH = "test.json"
API_URL = "http://localhost:8000/chat"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configurar API de Gemini para Métricas (Local)
api_key = GEMINI_API_KEY or GOOGLE_API_KEY
genai.configure(api_key=api_key)

# --- 1. CLASE EMBEDDING (Para Similitud Local) ---
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

Settings.llm = Gemini(model="models/gemini-2.0-flash", api_key=api_key)
Settings.embed_model = GeminiEmbedding()

# --- 2. HELPER DE REINTENTOS CON BACKOFF (Para Métricas Locales) ---
async def retry_with_backoff(coro_factory, max_retries=8, initial_delay=1):
    """Ejecuta una función que devuelve una corrutina con reintentos."""
    for i in range(max_retries):
        try:
            return await coro_factory()
        except Exception as e:
            msg = str(e)
            if "429" in msg or "resource exhausted" in msg.lower():
                delay = initial_delay * (2 ** i)
                print(f"   ⚠️ [DEBUG] Límite de tasa (429) detectado en '{coro_factory.__name__}'. Reintentando intento {i+1}/{max_retries} en {delay}s...")
                await asyncio.sleep(delay)
            else:
                raise e

# --- 3. FUNCIONES DE EVALUACIÓN ---
async def evaluate_relevance(question: str, expected: str, generated: str) -> dict:
    prompt = f"""
    Actúa como un evaluador experto de sistemas RAG.
    Determina si la respuesta generada es correcta vs la esperada.
    PREGUNTA: {question}
    RESPUESTA ESPERADA: {expected}
    RESPUESTA GENERADA: {generated}
    Responde exclusivamente en JSON: {{"is_relevant": true/false, "score": 0.0-1.0, "explanation": "..."}}
    """
    async def _call_llm():
        print(f"      🤖 [DEBUG] Llamando LLM para evaluar relevancia...")
        response = await Settings.llm.acomplete(prompt)
        text = response.text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].strip()
        return json.loads(text)

    try:
        return await retry_with_backoff(_call_llm)
    except Exception as e:
        print(f"⚠️ [DEBUG] Error final evaluando relevancia: {e}")
        return {"is_relevant": False, "score": 0.0, "explanation": f"Error: {e}"}

async def get_embedding(text: str, label: str = "text") -> np.ndarray:
    async def _call_embedding():
        print(f"      🧬 [DEBUG] Llamando Embedding para: {label} ({len(text)} chars)")
        return await Settings.embed_model._aget_text_embedding(text)
    emb = await retry_with_backoff(_call_embedding)
    return np.array(emb).reshape(1, -1)

async def calculate_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2: return 0.0
    emb1 = await get_embedding(text1, "esperada")
    emb2 = await get_embedding(text2, "generada")
    return float(cosine_similarity(emb1, emb2)[0][0])

# --- 4. FUNCIÓN PARA LLAMAR AL BACKEND ---
async def query_backend(client, question: str) -> str:
    """Llama al endpoint de chat del backend desplegado."""
    payload = {
        "message": question,
        "history": []
    }
    
    async def _call_api():
        print(f"      🌐 [DEBUG] Llamando API Backend: {API_URL}...")
        full_response = ""
        # Usamos streaming como en el stress test para mayor robustez
        async with client.stream("POST", API_URL, json=payload, timeout=None) as response:
            if response.status_code != 200:
                raise Exception(f"Backend error: {response.status_code}")
            async for chunk in response.aiter_text():
                if chunk.strip():
                    full_response += chunk
        return full_response

    return await retry_with_backoff(_call_api)

async def main():
    print(f"🚀 Iniciando evaluación RAG via LOCALHOST API (8000)...")
    
    if not os.path.exists(TEST_JSON_PATH):
        print("❌ test.json no encontrado.")
        return

    with open(TEST_JSON_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # Limpiar resultados previos para forzar re-evaluación con el backend local
    for item in test_data:
        for key in ["generated_answer", "cosine_similarity", "is_relevant", "relevance_score", "relevance_explanation", "error"]:
            item.pop(key, None)

    total = len(test_data)
    processed_count = 0

    async with httpx.AsyncClient(timeout=None) as client:
        for item in test_data:
            # Reanudar si ya tiene resultado completo
            if all(k in item for k in ["generated_answer", "cosine_similarity", "is_relevant"]):
                print(f"⏩ [{item['id']}/{total}] Saltando (ya procesado)")
                processed_count += 1
                continue

            question = item.get("question")
            expected = item.get("expected_answer")
            print(f"🔍 [{item['id']}/{total}] Procesando: {question[:60]}...")
            
            try:
                # 1. Query RAG via Backend
                generated = await query_backend(client, question)
                item["generated_answer"] = generated
                
                # 2. Similitud
                sim_score = await calculate_similarity(expected, generated)
                item["cosine_similarity"] = round(sim_score, 4)
                
                # 3. Relevancia LLM
                rel_eval = await evaluate_relevance(question, expected, generated)
                item["is_relevant"] = rel_eval.get("is_relevant", False)
                item["relevance_score"] = rel_eval.get("score", 0.0)
                item["relevance_explanation"] = rel_eval.get("explanation", "")
                
                print(f"   📊 Similidad: {item['cosine_similarity']} | Relevante: {item['is_relevant']}")
                processed_count += 1
                
                # Guardar después de cada pregunta
                with open(TEST_JSON_PATH, 'w', encoding='utf-8') as f:
                    json.dump(test_data, f, indent=2, ensure_ascii=False)
                
                # Pequeña pausa
                await asyncio.sleep(0.5)

            except Exception as e:
                print(f"❌ Error en [{item['id']}]: {e}")
                item["error"] = str(e)
                with open(TEST_JSON_PATH, 'w', encoding='utf-8') as f:
                    json.dump(test_data, f, indent=2, ensure_ascii=False)
                # Opcional: break si es un error sistémico

    print(f"\n✅ Evaluación finalizada: {processed_count}/{total} preguntas procesadas.")

if __name__ == "__main__":
    asyncio.run(main())

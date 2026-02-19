import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Depends
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader, HTTPBasic, HTTPBasicCredentials
import secrets
from pydantic import BaseModel
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.llms.gemini import Gemini
from llama_index.core.embeddings import BaseEmbedding
import google.generativeai as genai
from typing import List, Optional
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
import shutil
# Importar nuestra función de ingestión
from ingest import ingest_documents, DATA_DIR, PERSIST_DIR

# --- 0. CARGAR VARIABLES DE ENTORNO ---
load_dotenv()

# --- 1. CONFIGURAR EMBEDDINGS CON CACHÉ ---
embedding_cache = {}

class GeminiEmbedding(BaseEmbedding):
    def __init__(self, model: str = "models/gemini-embedding-001"):
        super().__init__()
        self._model = model

    def _get_query_embedding(self, query: str) -> List[float]:
        if query in embedding_cache:
            return embedding_cache[query]
            
        result = genai.embed_content(
            model=self._model,
            content=query,
        )
        embedding = result["embedding"]
        embedding_cache[query] = embedding
        return embedding

    def _get_text_embedding(self, text: str) -> List[float]:
        result = genai.embed_content(
            model=self._model,
            content=text,
        )
        return result["embedding"]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_query_embedding, query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_text_embedding, text)

if not os.getenv("GOOGLE_API_KEY"):
    print("❌ Error: La variable de entorno GOOGLE_API_KEY no fue encontrada.")
    exit()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("❌ Error: La variable de entorno GEMINI_API_KEY no fue encontrada.")
    exit()

# Configurar Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# --- 1. CONFIGURAR MODELOS LLAMA_INDEX ---
print("⚙️ Configurando modelos...")

Settings.llm = Gemini(model="models/gemini-2.0-flash", max_output_tokens=2048)
Settings.embed_model = GeminiEmbedding(model="models/gemini-embedding-001")

# --- 2. CARGAR/VERIFICAR ÍNDICE ---
# Ya no crasheamos si no existe, para permitir subida inicial y primer ingest.
# Pero intentamos cargar si existe.

vector_index = None
chroma_collection = None

def cargar_indice():
    global vector_index, chroma_collection
    if not os.path.exists(PERSIST_DIR):
        print(f"⚠️  Advertencia: No se encontró '{PERSIST_DIR}'. El sistema iniciará sin conocimiento RAG. Usa /admin/ingest.")
        vector_index = None
        chroma_collection = None
        return False
    
    print(f"📂 Cargando índice vectorial desde '{PERSIST_DIR}'...")
    try:
        db = chromadb.PersistentClient(path=PERSIST_DIR)
        chroma_collection = db.get_or_create_collection("admision_unap")
        
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        nuevo_indice = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=Settings.embed_model
        )
        # Solo actualizamos las globales si la carga fue exitosa
        vector_index = nuevo_indice
        print(f"✅ Índice vectorial cargado exitosamente. (Nodos: {chroma_collection.count()})")
        return True
    except Exception as e:
        print(f"❌ Error crítico cargando ChromaDB: {e}")
        import traceback
        traceback.print_exc()
        vector_index = None
        chroma_collection = None
        return False

cargar_indice()

# --- 5. FASTAPI APP & ADMIN ROUTES ---
app = FastAPI()

# --- AUTH CON SEGURIDAD ---
ADMIN_USER = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASSWORD", "secret_unam_2025")
security = HTTPBasic()

async def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USER)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASS)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Credenciales inválidas",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# --- ADMIN ENDPOINTS ---

@app.get("/admin/status", dependencies=[Depends(verify_admin)])
async def admin_status():
    """Estado del sistema: archivos y vectores."""
    files = []
    if os.path.exists(DATA_DIR):
        files = os.listdir(DATA_DIR)
    
    vectors = 0
    if chroma_collection:
        vectors = chroma_collection.count()
        
    return {
        "status": "online",
        "persisted_db": os.path.exists(PERSIST_DIR),
        "vector_count": vectors,
        "files_count": len(files),
        "files": files
    }

@app.get("/admin/metrics", dependencies=[Depends(verify_admin)])
async def admin_metrics():
    """Métricas avanzadas de calidad RAG."""
    if not chroma_collection:
        return {"error": "Index not loaded"}
    
    results = chroma_collection.get(include=['metadatas'])
    metadatas = results['metadatas']
    
    doc_distribution = {}
    for m in metadatas:
        fname = m.get('file_name', 'unknown')
        doc_distribution[fname] = doc_distribution.get(fname, 0) + 1
        
    return {
        "doc_distribution": doc_distribution,
        "total_chunks": len(metadatas),
        "avg_chunks_per_doc": len(metadatas) / len(doc_distribution) if doc_distribution else 0
    }

@app.post("/admin/ingest", dependencies=[Depends(verify_admin)])
async def trigger_ingest(force: bool = False):
    """Disparar proceso de ingestión manual."""
    try:
        print(f"🔧 Admin trigger: Ingest (Force={force})")
        # Ejecutar ingestión (esto bloquea el thread, idealmente background task, pero RAG suele ser rápido si hay pocos files)
        # Para evitar bloquear todo, usamos run_in_executor o background tasks de FastAPI
        # Por simplicidad ahora: directo
        await ingest_documents(force_rebuild=force)
        
        # Recargar índice en memoria
        if cargar_indice():
            return {"message": "Ingestión completada y índice recargado exitosamente.", "status": "ok"}
        else:
             print("❌ Falló la recarga del índice post-ingestión.")
             return {"message": "Ingestión completada, pero no pudo cargarse el índice en memoria. Revisa los logs del servidor.", "status": "warning"}
    except Exception as e:
        print(f"❌ Error en /admin/ingest: {e}")
        return {"message": f"Error durante la ingestión: {str(e)}", "status": "error"}

@app.get("/admin/files", dependencies=[Depends(verify_admin)])
async def list_files():
    if not os.path.exists(DATA_DIR):
        return []
    return os.listdir(DATA_DIR)

@app.post("/admin/files", dependencies=[Depends(verify_admin)])
async def upload_file(file: UploadFile = File(...)):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    file_path = os.path.join(DATA_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"filename": file.filename, "message": "Archivo subido correctamente."}

@app.delete("/admin/files/{filename}", dependencies=[Depends(verify_admin)])
async def delete_file(filename: str):
    file_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"message": f"Archivo {filename} eliminado."}
    raise HTTPException(status_code=404, detail="Archivo no encontrado")

# --- 4. CONFIGURAR MOTOR DE CONSULTAS (Helper dinámico) ---
def get_retriever():
    global vector_index
    if vector_index:
        return vector_index.as_retriever(similarity_top_k=4)
    return None

def get_query_engine():
    global vector_index
    if vector_index:
        return vector_index.as_query_engine(similarity_top_k=4, streaming=True)
    return None

print("✅ Sistema RAG inicializado (esperando consultas).")

# --- CONFIGURAR CORS ---
# Esto permite que el frontend de React (que se ejecutará en otro puerto)
# se comunique con este backend.
# Para producción, es recomendable restringir los orígenes.
origins = [
    "*",  # Puerto común para Vite
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(SessionMiddleware, secret_key=os.urandom(24))

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class ChatRequest(BaseModel):
    message: str
    history: List[dict]


async def llamar_llm_streaming(prompt: str):
    """Llamada a Gemini streaming por google.generativeai (Async) con reintentos para error 429"""
    max_retries = 3
    retry_delay = 2  # segundos

    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = await model.generate_content_async(
                prompt, 
                stream=True,
                generation_config=genai.GenerationConfig(max_output_tokens=2048)
            )
            async for chunk in response:
                try:
                    if chunk.text:
                        yield chunk.text
                except ValueError:
                    # Si el chunk no tiene texto (por safety o finish_reason), lo ignoramos
                    pass
            return  # Éxito, salimos del bucle de reintentos

        except Exception as e:
            error_str = str(e)
            if "429" in error_str and attempt < max_retries - 1:
                print(f"⚠️ Cuota agotada (429). Reintento {attempt + 1}/{max_retries} en {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue
            
            print(f"❌ Error llamando a Gemini: {e}")
            if "429" in error_str:
                yield "El servidor está temporalmente saturado (límite de cuota alcanzado). Por favor, espera unos segundos e intenta de nuevo."
            else:
                yield "Lo siento, hubo un problema al conectar con el servicio de IA."
            break

def generar_prompt(pregunta, contexto, historial):
    historial_texto = "\n".join(
        [f"{msg['role']}: {msg['parts'][0]}" for msg in historial]
    )
    
    return f"""
## ROL Y OBJETIVO
Eres el Asistente Virtual de Admisiones de la UNAM - un guía amigable, motivador y EXPRESIVO que ayuda a estudiantes de secundaria a navegar el proceso de admisión universitaria. Tu misión es hacer que la información compleja sea **fácil de entender** y que cada estudiante se sienta apoyado y confiado en su camino a la universidad.
Si preguntan cual es tu proposito, que sea el de facilitar la vida relacionada a la Universidad Nacional de Moquegua, o cosas similares.
## ESTILO DE COMUNICACIÓN
- **Expresivo y Cercano**: Usa un tono cálido y entusiasta. No seas robótico, sé humano
- **Claro y Simple**: Explica conceptos complejos con palabras sencillas. Evita jerga innecesaria
- **Didáctico**: Usa ejemplos, analogías o comparaciones cuando ayuden a clarificar
- **Organizado**: Estructura tu respuesta con listas numeradas o con viñetas cuando presentes múltiples puntos
- **Empático**: Reconoce que el proceso de admisión puede ser abrumador y muestra comprensión

## REGLAS Y CONOCIMIENTO
1. **Prioridad de Fuentes:** Tu fuente de verdad principal es el **"Historial de la Conversación"**. Úsalo SIEMPRE para responder preguntas sobre la conversación actual
2. **Uso del Contexto RAG:** Usa el **"Contexto Relevante"** únicamente para responder preguntas sobre el proceso de admisión
3. **Combinación Inteligente:** Si una pregunta depende del historial, combina ambas fuentes para dar una respuesta coherente
4. **Alias y Abreviaturas:** Reconoce **"UNAM"** como la abreviatura oficial de **"Universidad Nacional de Moquegua"**
5. **Manejo de Incertidumbre:** Si no tienes la información, admítelo claramente y sugiere consultar las fuentes oficiales
6. **Privacidad Absoluta:** NUNCA pidas, almacenes o repitas información personal del usuario
7. **Enfoque Único:** Si preguntan por temas no relacionados con admisión, redirige amablemente
8. **PROHIBICIÓN DE CUADROS Y TABLAS:** Está PROHIBIDO usar formato de tablas de Markdown. Usa párrafos claros o listas simples
9. **SIN ENLACES:** No proporciones URLs. Si necesitas mencionar una página, descríbela textualmente
10. **SIN MENCIONES DE FUENTES:** No menciones nombres de archivos o documentos (ej. "Según el Prospecto", "En el reglamento") en tu respuesta. Integra la información de forma natural y directa.


## CONTENIDO SITUACIONAL PARA FECHAS

1. CRONOGRAMA DE INSCRIPCIÓN Y COSTOS
CUADRO N° 1: CRONOGRAMA DE INSCRIPCIÓN DEL CONCURSO DE ADMISIÓN 2026-I
Las fechas especificas para el proceso de admision se detallan en el siguiente cronograma:

## Cronograma de Admisión 2026-I

* Inscripción al Examen Extraordinario 2026-I: Del 20 de enero al 11 de marzo de 2026.
* Inscripción al Examen Extraordinario (Plan Integral de Reparaciones y Víctimas de Terrorismo): Del 20 de enero al 27 de febrero de 2026.
* Toma de imágenes e identificación biométrica (Sede Moquegua): 12 y 13 de marzo de 2026.
* Evaluación de expedientes (Sede Moquegua): 13 de marzo de 2026.
* Examen de Admisión Extraordinario (Sede Moquegua): 15 de marzo de 2026.
* Inscripción al Examen Ordinario General: Del 20 de enero al 17 de marzo de 2026.
* Toma de imágenes e identificación biométrica (Ordinario): Del 16 al 18 de marzo de 2026.
* Examen de Admisión Ordinario - Canal A (Sede Moquegua): 21 de marzo de 2026.
* Examen de Admisión Ordinario - Canal B y C (Moquegua e Ilo): 22 de marzo de 2026.

2. NUMEROS DE CONTACTO DIRECTO Y OFICINAS
Para consultas directas, los postulantes pueden comunicarse a los siguientes números de contacto y oficinas:

Numero telefonico de contacto de Moquegua:
- Central Telefónica: (+51) 923236099

Numero telefonico de contacto de Ilo:
- Central Telefónica: (+51) 912428484

Consulta por WhatsApp al número: 923236099.

3. PAGOS POR DERECHO DE EXAMEN DE ADMISIÓN
Realizar el pago correspondiente en el Banco de la Nación o agencias/agentes del Banco de la Nación y en Tesorería de la Universidad Nacional de Moquegua.
Los montos que deben abonar los postulantes por derecho de inscripción, según su Modalidad de Ingreso y el tipo de Colegio donde culminaron sus Estudios Secundarios o Universidades de procedencia, son los siguientes:

**Proceso Ordinario:**
*   El examen ordinario tiene un costo de S/ 350.00.
*   El participante evaluativo debe pagar S/ 200.00.

**Proceso Extraordinario:**
*   Los postulantes que provienen de los primeros puestos de colegio deben pagar S/ 300.00.
*   Los egresados de COAR deben pagar S/ 300.00.
*   Los titulados o graduados deben pagar S/ 450.00.
*   Los deportistas destacados deben pagar S/ 300.00.
*   Las personas con discapacidad deben pagar S/ 120.00.
*   Los postulantes por Convenio Andrés Bello deben pagar S/ 350.00.
*   El traslado interno de la UNAM tiene un costo de S/ 300.00.
*   El traslado externo de otras universidades tiene un costo de S/ 400.00.

**Otros Pagos:**
*   El duplicado de carné tiene un costo de S/ 6.00.
*   La constancia de ingresantes (uso externo) tiene un costo de S/ 8.00.

Todo pago se realiza luego de la primera fase de preinscripción. Si el pago se realiza en el Banco de la Nación, el váucher debe ser subido a la plataforma virtual de inscripción o canjeado por un comprobante de pago en la Unidad de Tesorería (caja) de la UNAM, para su respectiva validación.



4. CUADRO DE CARRERAS Y VACANTES

## Datos de Admisión – Formato Estructurado

### Categorías
- ESCUELAS_PROFESIONALES
- CEPRE_UNAM_FASE_I
- CEPRE_UNAM_FASE_II
- PRIMEROS_PUESTOS_COLEGIOS
- EGRESADOS_COAR
- TITULADOS_GRADUADOS
- DEPORTISTAS_DESTACADOS
- PERSONAS_CON_DISCAPACIDAD
- VICTIMAS_TERRORISMO
- PLAN_INTEGRAL_REPARACION
- TRASLADOS_EXTERNOS
- TRASLADOS_INTERNOS
- CONVENIO_ANDRES_BELLO
- TOTAL_EXTRAORDINARIO
- TOTAL_ORDINARIO
- TOTAL_INGRESANTES

---

### Sede Central Moquegua

**Ingeniería de Minas**  
Valores:  
[5, 6, 2, 2, 1, 1, 3, 1, 1, 1, 1, 1, 14, 20, 45]

**Ingeniería Agroindustrial**  
Valores:  
[5, 5, 2, 2, 1, 0, 2, 0, 0, 1, 1, 0, 9, 16, 35]

**Ingeniería Civil**  
Valores:  
[5, 6, 2, 2, 2, 1, 3, 1, 1, 1, 1, 0, 14, 25, 50]

**Medicina**  
Valores:  
[7, 6, 2, 2, 1, 1, 2, 0, 1, 1, 1, 0, 11, 11, 35]

**Contabilidad**  
Valores:  
[5, 6, 2, 2, 0, 1, 2, 0, 1, 0, 0, 0, 8, 16, 35]

**Derecho**  
Valores:  
[5, 6, 2, 2, 2, 1, 2, 0, 0, 0, 0, 0, 9, 15, 35]

**Gestión Pública y Desarrollo Social**  
Valores:  
[6, 7, 2, 2, 2, 1, 3, 1, 1, 2, 2, 1, 17, 30, 60]

---

### Filial Ilo

**Ingeniería Pesquera**  
Valores:  
[7, 7, 2, 2, 1, 0, 2, 0, 1, 1, 2, 1, 12, 14, 40]

**Ingeniería Ambiental**  
Valores:  
[5, 6, 2, 2, 1, 0, 2, 0, 1, 1, 1, 0, 10, 24, 45]

**Ingeniería de Sistemas e Informática**  
Valores:  
[6, 7, 2, 3, 1, 1, 3, 1, 0, 1, 1, 1, 14, 23, 50]

**Contabilidad**  
Valores:  
[5, 6, 2, 2, 0, 1, 2, 1, 1, 0, 0, 0, 9, 15, 35]

**Derecho**  
Valores:  
[5, 6, 2, 2, 2, 1, 2, 0, 0, 0, 0, 0, 9, 15, 35]

**Administración**  
Valores:  
[5, 6, 3, 2, 1, 1, 3, 0, 1, 1, 1, 1, 14, 25, 50]

---

### Totales Generales

Valores:  
[71, 80, 27, 27, 15, 10, 31, 5, 9, 10, 11, 5, 150, 249, 550]

5. DATOS EXTRA
- La Universidad Nacional de Moquegua (UNAM) es una institución pública de educación superior
-La universidad ofrece 2 examenes al año, donde las carreras disponibles son distintas en cada examen.
-Informacion sensible se le manda al estudiante por el correo directamente, despues de realizar su pago.
-No se pueden hacer modificaciones despues de haber realizado el pago.
-En preguntas relacionadas a pagos, derivar a contactos o a la pagina oficial.
-Para preguntas sobre inscripcion derivar a whatsapp o a la pagina oficial.
-Las sedes de los examenes son Moquegua: Prolongación Calle Ancash S/N, Ex Cuartel Mariscal Nieto e Ilo: Urb. Ciudad Jardín S/N.
-Cuando te pregunten por nombres de rector, autoridades, etc, esto es lo que tienes que tener en cuenta:

Dr. Hugo Ruben Marca Maquera es el Rector

Dr. Alejandro Manuel Ecos Espino es el Vicerrector Académico


Dr. Jhony Mayta Hancoo es el Vicerrector de InvestigaciónDr. 




DIRECTOR DE LA DIRECCIÓN DE ADMISIÓN
DR. JOSÉ ANTONIO VALERIANO ZAPANA

MIEMBROS DE LA COMISIÓN CENTRAL DE ADMISIÓN
Presidente del Comité Central de Admisión:
Dr. Arquímedes León Vargas Luque

Miembro del Comité Central de Admisión Sede Moquegua
Dr. Ronald Raúl Arce Coaquira

Miembro del Comité Central de Admisión Filial Ilo
Dra. Maribel Estela Coaguila Mamani
## FORMATO Y ESTRUCTURA DE LA RESPUESTA
Tu respuesta DEBE seguir esta estructura de formato para ser clara y visualmente atractiva:
1. **Cuerpo de la Respuesta:**
   - **Concisión Extrema:** Sé extremadamente directo y evita introducciones largas.
   - **Optimización Vertical:** Prioriza el uso de listas cortas para que la respuesta sea legible en pantallas pequeñas.
   - **Si la respuesta describe un proceso o una secuencia de pasos, DEBES usar una lista numerada (1., 2., 3.) para guiar al usuario.**
   - # CÓDIGO CORREGIDO
   # PROMPT CORREGIDO Y MÁS PRECISO
- Utiliza **negritas** para resaltar conceptos clave **dentro de las mismas oraciones**, sin crear líneas nuevas solo para ellos. Por ejemplo, escribe 'El costo es de **S/ 300.00**.' en lugar de poner '**S/ 300.00**' en una línea separada.
   - Estructura la información compleja en **listas de puntos** (*) si no es un proceso secuencial.
   - Mantén los párrafos cortos y directos.
3. **Separador Visual:** Después del cuerpo de tu respuesta, inserta una línea horizontal usando ---

4. **Preguntas de Seguimiento:**  
Debajo del separador, escribe entre 1 y 2 preguntas relevantes, siempre formuladas en **primera persona** (yo/mi), como si el usuario mismo las estuviera haciendo.  

No uses títulos, encabezados ni palabras como "sugerencias" o "preguntas proactivas".  
Cada pregunta debe ir en una línea nueva y comenzar con un asterisco (*) seguido de un espacio.  

Nunca uses expresiones en segunda persona como “¿Quieres...?”, “¿Te gustaría...?”, “¿Necesitas...?”, “¿Quieres que te explique...?”.  

✅ Ejemplo correcto (no lo uses literalmente, solo como referencia de formato):
---
* ¿Cómo puedo inscribirme en el examen de admisión?  
* ¿Qué documentos debo presentar para postular?  

❌ Ejemplo incorrecto (no uses este formato):
---
* ¿Quieres inscribirte en el examen de admisión?  
* ¿Te gustaría saber qué documentos necesitas para postular?  
---

**Historial de la Conversación Actual:**
{historial_texto}

**Contexto Relevante para la Nueva Pregunta:**
{contexto}

**Nueva Pregunta del Usuario:**
{pregunta}

**Respuesta:**
"""

async def judge_context_sufficiency(query: str, nodes: list) -> dict:
    """Usa el LLM para evaluar si el contexto recuperado es suficiente para responder (con reintentos)."""
    if not nodes:
        return {"sufficient": False, "reason": "No se recuperaron documentos", "confidence": 1.0}
    
    context = "\n".join([n.text if hasattr(n, 'text') else n.get_content() for n in [getattr(node, 'node', node) for node in nodes]])
    prompt = f"""
    Eres un auditor de calidad RAG para la UNAM (Universidad Nacional de Moquegua). 
    PREGUNTA DEL USUARIO: {query}
    CONTEXTO RECUPERADO DE DOCUMENTOS:
    ---
    {context}
    ---
    ¿El CONTEXTO contiene información relevante y real para responder a la PREGUNTA?
    Responde estrictamente en formato JSON: {{"sufficient": true/false, "confidence": 0.0-1.0}}
    """
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            # Usamos el modelo configurado en Settings
            response = await Settings.llm.acomplete(prompt)
            cleaned = response.text.strip().replace('```json', '').replace('```', '')
            import json
            return json.loads(cleaned)
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                print(f"⚠️ Error 429 en Judge. Reintentando...")
                await asyncio.sleep(1)
                continue
            print(f"⚠️ Error en LLM Judge: {e}")
            return {"sufficient": True, "confidence": 0.5} # Fallback optimista para no bloquear al usuario

def evaluar_necesidad_rag(pregunta: str) -> bool:
    """Detecta si la pregunta requiere buscar en documentos de admisión."""
    keywords_admision = [
        "requisitos", "examen", "costo", "pago", "fecha", "cronograma", 
        "vacante", "carrera", "inscripcion", "inscripción", "postular",
        "documento", "voucher", "admisión", "ordinario", "extraordinario",
        "cepre", "unt", "moquegua", "ilo", "filial", "sede"
    ]
    
    pregunta_lower = pregunta.lower()
    
    # 1. Si contiene palabras clave de admisión, definitivamente necesita RAG
    if any(kw in pregunta_lower for kw in keywords_admision):
        return True
        
    # 2. Si es muy corta (ej: "hola", "buen día"), probablemente no necesita RAG
    if len(pregunta_lower.split()) <= 2:
        return False
        
    # 3. Preguntas de identidad suelen no necesitar RAG (ya están en el system prompt)
    identidad = ["quien eres", "qué eres", "ayudame", "tu nombre", "ayudarme", "como te llamas"]
    if any(id_kw in pregunta_lower for id_kw in identidad):
        return False

    # Por defecto, si no estamos seguros, usamos RAG por seguridad
    return True

async def generar_respuesta_stream(pregunta: str, historial: list):
    # Recuperar contexto: los top-k chunks relevantes
    print("\n" + "="*80)
    print(f"🔍 NUEVA CONSULTA: {pregunta}")
    print("="*80)
    
    # 0. Traffic Controller: ¿Realmente necesitamos RAG?
    necesita_rag = evaluar_necesidad_rag(pregunta)
    
    if not necesita_rag:
        print("⚡ TRAFFIC: Saltando RAG (Conversacional)")
        prompt = generar_prompt(pregunta, "No se requiere contexto para esta interacción conversacional.", historial)
        async for chunk in llamar_llm_streaming(prompt):
            yield chunk
        return

    # 1. Recuperar contexto...
    retriever = get_retriever()
    
    if retriever is None:
        print("⚠️ AVISO: No hay retriever disponible (índice no cargado).")
        yield "Lo siento, mi base de conocimientos sobre admisión está actualmente en mantenimiento o no pudo ser cargada. ¿Deseas que intente responder con lo que sé o prefieres esperar unos minutos?\n\n---"
        yield "\n* ¿Cuándo estará disponible el sistema?\n* ¿Cómo puedo contactar con admisión?"
        return

    try:
        print(f"🔍 DEBUG: Ejecutando retrieval...")
        source_nodes = await retriever.aretrieve(pregunta)
        print(f"🔍 DEBUG: Nodos recuperados: {len(source_nodes)}")
        
        # Imprimir chunks completos para debugging
        print("\n" + "="*80)
        print("📦 CHUNKS RECUPERADOS (COMPLETOS):")
        print("="*80)
        for idx, source_node in enumerate(source_nodes, 1):
            node = getattr(source_node, 'node', source_node)
            metadata = getattr(node, 'metadata', {})
            chunk_text = node.text if hasattr(node, 'text') else node.get_content()
            score = getattr(source_node, 'score', 'N/A')
            
            print(f"\n📄 CHUNK #{idx}")
            print(f"   📁 Archivo: {metadata.get('filename', 'Unknown')}")
            print(f"   📊 Score: {score}")
            print(f"   📝 Contenido:")
            print(f"   {'-'*76}")
            # Imprimir el texto con indentación
            for line in chunk_text.split('\n'):
                print(f"   {line}")
            print(f"   {'-'*76}")
        print("="*80 + "\n")
        
        # 2. Juzgar Suficiencia
        judge = await judge_context_sufficiency(pregunta, source_nodes)
        print(f"⚖️ JUDGE RESULT: {judge}")

        if not judge.get("sufficient") and judge.get("confidence", 0) > 0.7:
            print("🚫 RECHAZADO: Contexto insuficiente.")
            yield "Lo siento, no encontré información específica en los documentos oficiales de la UNAM que me permita responder a tu pregunta con seguridad. ¿Deseas consultar sobre requisitos, fechas de examen o carreras disponibles?\n\n---"
            yield "\n* ¿Cuáles son los requisitos de admisión?\n* ¿Cuándo es el próximo examen?"
            return

        # 3. Formatear contexto con etiquetas de fuente para cada chunk
        contexto_partes = []
        source_files = set()
        
        for source_node in source_nodes:
            node = getattr(source_node, 'node', source_node)
            metadata = getattr(node, 'metadata', {})
            
            # Obtener el texto del chunk
            chunk_text = node.text if hasattr(node, 'text') else node.get_content()
            
            # Obtener y limpiar el nombre del archivo
            if 'filename' in metadata:
                filename = metadata['filename'].replace('.txt', '').replace('_', ' ')
                source_files.add(filename)
                # Formatear el chunk con su fuente
                contexto_partes.append(f"[Fuente: {filename}]\n{chunk_text}\n")
            else:
                # Si no hay metadata, incluir el chunk sin etiqueta
                contexto_partes.append(f"{chunk_text}\n")
        
        # Unir todos los chunks con separadores
        contexto = "\n---\n".join(contexto_partes)
        
        # 4. Generar respuesta
        prompt = generar_prompt(pregunta, contexto, historial)
        
        async for chunk in llamar_llm_streaming(prompt):
            yield chunk
            
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"❌ Error CRÍTICO en flujo RAG:\n{error_msg}")
        yield "Ocurrió un error procesando tu consulta. Por favor, intenta de nuevo. (Detalle técnico: error en motor RAG)"

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    request.session.pop('chat_history', None)
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(request: Request, chat_request: ChatRequest):
    pregunta = chat_request.message
    historial = chat_request.history

    if not pregunta:
        return StreamingResponse("Error: no se recibió ninguna pregunta.", status_code=400)

    async def response_generator():
        nonlocal historial
        full_response = ""
        suggested_questions = []

        async for chunk in generar_respuesta_stream(pregunta, historial):
            full_response += chunk
            if "---" in chunk:
                parts = chunk.split("---")
                full_response = parts[0]
                suggested_questions = [q.strip() for q in parts[1].split("\n") if q.strip() and q.startswith("*")]
                suggested_questions = [q.replace("*", "").strip() for q in suggested_questions]

            yield chunk

        historial.append({"role": "user", "parts": [pregunta]})
        historial.append({"role": "model", "parts": [full_response], "suggestedQuestions": suggested_questions})

        if len(historial) > 10:
            historial = historial[-10:]

        request.session['chat_history'] = historial

    return StreamingResponse(response_generator(), media_type='text/event-stream')


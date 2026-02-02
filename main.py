import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.llms.gemini import Gemini
from llama_index.core.embeddings import BaseEmbedding
import google.generativeai as genai
from typing import List
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware

# --- 0. CARGAR VARIABLES DE ENTORNO ---
load_dotenv()

# --- 1. CONFIGURAR EMBEDDINGS CON CACHÉ ---
embedding_cache = {}

class GeminiEmbedding(BaseEmbedding):
    def __init__(self, model: str = "models/embedding-001"):
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

Settings.llm = Gemini(model="gemini-2.5-flash-lite", max_output_tokens=1024)
Settings.embed_model = GeminiEmbedding(model="models/embedding-001")

# --- 2. CARGAR ÍNDICE EXISTENTE (MODO PRODUCCIÓN) ---
PERSIST_DIR = "./chroma_db"

if not os.path.exists(PERSIST_DIR):
    print(f"❌ ERROR CRÍTICO: No se encontró el directorio '{PERSIST_DIR}'.")
    print("👉 EJECUTA PRIMERO: python ingest.py")
    print("   El servidor no puede iniciar sin el índice vectorial.")
    # No usamos exit(1) directo para permitir reinicios en dev si se arregla, 
    # pero en producción esto causará un crash loop (correcto behavior).
    # Sin embargo, para uvicorn en reload, un raise es mejor.
    raise RuntimeError(f"Falta el índice vectorial en {PERSIST_DIR}")

print(f"📂 Cargando índice vectorial desde '{PERSIST_DIR}'...")
try:
    db = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = db.get_or_create_collection("admision_unap")
    
    # Debug: Verificar contenido de la colección
    collection_count = chroma_collection.count()
    print(f"🔍 DEBUG: Documentos en ChromaDB al cargar: {collection_count}")
    
    if collection_count == 0:
         print("⚠️  WARNING: La colección está vacía! Ejecuta ingest.py")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=Settings.embed_model
    )
    print("✅ Índice vectorial cargado.")
except Exception as e:
    print(f"❌ Error cargando ChromaDB: {e}")
    raise e


# --- 4. CONFIGURAR MOTOR DE CONSULTAS ---
print("🚀 Configurando motor de consultas RAG...")
query_engine = index.as_query_engine(similarity_top_k=4)

print("✅ Sistema RAG listo para consultas.")

# --- 5. FASTAPI APP ---
app = FastAPI()

# --- HEALTH CHECK ENDPOINT ---
@app.get("/health")
async def health_check():
    """Endpoint para verificar que el servicio está funcionando. 
    Cloud Run usa esto para el startup probe."""
    return {"status": "healthy", "message": "RAG System is running"}

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
    allow_credentials=True,
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
    """Llamada a Gemini streaming por google.generativeai (Async)"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        response = await model.generate_content_async(prompt, stream=True)
        async for chunk in response:
            try:
                if chunk.text:
                    yield chunk.text
            except ValueError:
                # Si el chunk no tiene texto (por safety o finish_reason), lo ignoramos
                pass
    except Exception as e:
        print(f"❌ Error llamando a Gemini: {e}")
        yield " "

def generar_prompt(pregunta, contexto, historial):
    historial_texto = "\n".join(
        [f"{msg['role']}: {msg['parts'][0]}" for msg in historial]
    )
    return f"""
## ROL Y OBJETIVO
Actúa EXCLUSIVAMENTE como Ayudante de Admision UNAM, un asistente virtual experto y amigable, cuyo único propósito es guiar a estudiantes de secundaria en el proceso de admisión universitaria. Tu tono debe ser siempre motivador, claro y alentador.
## REGLAS Y CONOCIMIENTO
1. **Prioridad de Fuentes:** Tu fuente de verdad principal es el **"Historial de la Conversación"**. Úsalo SIEMPRE para responder preguntas sobre la conversación actual (ej: \"¿qué te pregunté antes?\", \"¿a qué te referías con...?\")
2. **Uso del Contexto RAG:** Usa el **"Contexto Relevante"** únicamente para responder preguntas sobre el proceso de admisión universitaria (requisitos, fechas, costos, etc.).
3. **Combinación Inteligente:** Si una pregunta sobre la admisión depende del historial, combina ambas fuentes para dar una respuesta coherente.
4. **Alias y Abreviaturas:** Reconoce **\"UNAM\"** como la abreviatura oficial de **\"Universidad Nacional de Moquegua\"** y úsalas indistintamente.
5. **Manejo de Incertidumbre:** Si ninguna fuente contiene la respuesta, admítelo claramente y sugiere al usuario consultar las fuentes oficiales.
6. **Privacidad Absoluta:** NUNCA pidas, almacenes o repitas información personal del usuario.
7. **Enfoque Único:** Si el usuario pregunta por temas no relacionados con la admisión, redirige amablemente la conversación a tu propósito principal.
8. **Comportamiento** No des saludo a menos que el usuario lo haga primero. Responde de manera concisa y directa, evitando redundancias.


## CONTENIDO SITUACIONAL PARA FECHAS

1. CRONOGRAMA DE INSCRIPCIÓN Y COSTOS
CUADRO N° 1: CRONOGRAMA DE INSCRIPCIÓN DEL CONCURSO DE ADMISIÓN 2025-11
Las fechas especificas para el proceso de admision se detallan en el siguiente cronograma:


ITEM | DESCRIPCIÓN | FECHAS
--- | --- | ---
1 | TOMA DE IMÁGENES, IDENTIFICACIÓN BIOMÉTRICA Y GENERACIÓN DE CARNET DE POSTULANTE CEPRE-III (sede Moquegua y Filial Ilo) | 21 al 24 de julio al de 2025
2 | EXAMEN DE ADMISIÓN CENTRO DE ESTUDIOS PRE UNIVERSITARIO 2025-III | 27 de julio de 2025
3 | PUBLICACION DE RESULTADOS | 27 de julio de 2025
4 | INSCRIPCIÓN AL EXAMEN EXTRAORDINARIO 2025-11 | Del 27 de junio al 29 de julio 2025
5 | INSCRIPCION AL EXAMEN EXTRAORDINARIO PLAN INTEGRAL DE REPARACIONES Y VICTIMAS DE TERRORISMO | Del 20 de junio al 18 de julio 2025
6 | PROCESO DE EVALUACION Y VALIDACION DE DOCUMENTOS DE PERSONAS CON DISCAPACIDAD MODALIDAD EXTRAORDINARIO | 31 de julio 2025 (postulantes examen extraordinario)
7 | TOMA DE IMÁGENES, IDENTIFICACION BIOMÉTRICA Y GENERACIÓN DE CARNET DE POSTULANTE (sede Moquegua) | 30 y 31 de julio 2025 (postulantes examen extraordinario)
8 | EXAMEN DE ADMISIÓN EXTRAORDINARIO (solo en la sede de Moquegua) | 03 de agosto de 2025
9 | PUBLICACIÓN DE RESULTADOS | 03 de agosto de 2025
10 | INSCRIPCIÓN AL EXAMEN ORDINARIO GENERAL 2025-11 | Del 13 de junio al 01 de agosto 2025
11 | INSCRIPCIÓN EXTEMPORANEO AL EXAMEN ORDINARIO GENERAL 2025-11 | 04 al 06 de agosto 2025 (para postulantes del examen extraordinario que no alcanzaron vacante y regiones lejanas)
12 | TOMA DE IMÁGENES, IDENTIFICACIÓN BIOMÉTRICA Y GENERACIÓN DE CARNET DE POSTULANTE (sede Moquegua y Filial Ilo) | 04 al 08 de agosto 2025 (postulantes examen ordinario) 07 y 08 de agosto 2025 (regiones lejanas)
13 | EXAMEN DE ADMISIÓN ORDINARIO-CANAL BY C (sede Moquegua y Filial Ilo) | 10 de agosto de 2025
14 | PUBLICACIÓN DE RESULTADOS | 10 de agosto de 2025

El Canal B corresponde a las carreras de ingenierías:
o Ingeniería de Minas
o Ingeniería Agroindustrial
o Ingeniería Civil
o Ingeniería Pesquera
o Ingeniería Ambiental
o Ingeniería de Sistemas e Informática

El canal C corresponde a las carreras de ciencias sociales:
 o Gestión Pública y Desarrollo Social.
 o Administración

2. NUMEROS DE CONTACTO DIRECTO Y OFICINAS
Para consultas directas, los postulantes pueden comunicarse a los siguientes números de contacto y oficinas:

Numero telefonico de contacto de Moquegua:
- Central Telefónica: (+51) 923236099

Numero telefonico de contacto de Ilo:
- Central Telefónica: (+51) 912428484

Numero telefonico de contacto Admision whatsapp:

https://wa.me/923236099 (quiero que esto lo pongas como un hipervinculo que parezca un boton, que abra a otra pagina, no cambie la pagina)

3. PAGOS POR DERECHO DE EXAMEN DE ADMISIÓN
Realizar el pago correspondiente en el Banco de la Nación o agencias/agentes del Banco de la Nación y en Tesorería de la Universidad Nacional de Moquegua.
Los montos que deben abonar los postulantes por derecho de inscripción, según su Modalidad de Ingreso y el tipo de Colegio donde culminaron sus Estudios Secundarios o Universidades de procedencia, son los siguientes:

N | CONCEPTO | MONTO
-- | --- | ---
**EXAMEN ORDINARIO** | | 
1 | Examen Ordinario | S/ 350.00
**EXAMEN EXTRAORDINARIO** | | 
1 | Titulados o graduados universitarios. | S/ 450.00
2 | Traslado Externo de Otras Universidades | S/ 400.00
3 | Traslado Interno | S/ 350.00
4 | Primer y segundo puesto de II.EE. y Egresados COAR (2023 2024) | S/ 300.00
5 | Deportistas Destacados (Ley N°28036) | S/ 300.00
6 | Personas con Discapacidad (Ley N° 29973) | S/ 120.00
7 | Convenio Andrés Bello (D.S. N 012-99-ED) | S/ 350.00
8 | Victimas de Terrorismo, según Ley N° 27277 y Plan Integral de Reparaciones, según Ley N° 28592. | Exonerado

Todo pago se realizará luego de la primera fase de preinscripción.
Solo en el caso los pagos en el Banco de la Nación, luego el váucher deberá subirlo a la plataforma virtual de inscripción o ser canjeado por un comprobante de pago en la Unidad de Tesorería (caja) de la UNAM, para su respectiva validación.


4. CUADRO DE CARRERAS Y VACANTES


2. CUADRO DE VACANTES
CUADRO N° 2: CUADRO DE VACANTES PARA EL PROCESO DE ADMISIÓN 2025-11
En la Sede Central Moquegua:

Ingeniería de Minas: 40 vacantes (10 CEPRE, 12 Extraordinario, 18 Ordinario).

Ingeniería Agroindustrial: 35 vacantes (10 CEPRE, 11 Extraordinario, 14 Ordinario).

Ingeniería Civil: 50 vacantes (15 CEPRE, 17 Extraordinario, 18 Ordinario).

Gestión Pública y Desarrollo Social: 60 vacantes (13 CEPRE, 24 Extraordinario, 23 Ordinario).

En la Filial Ilo:

Ingeniería Pesquera: 37 vacantes (14 CEPRE, 13 Extraordinario, 10 Ordinario).

Ingeniería Ambiental: 46 vacantes (12 CEPRE, 12 Extraordinario, 22 Ordinario).

Ingeniería de Sistemas e Informática: 60 vacantes (14 CEPRE, 24 Extraordinario, 22 Ordinario).

Administración: 50 vacantes (13 CEPRE, 15 Extraordinario, 22 Ordinario).

Totales generales: 378 vacantes (101 CEPRE, 128 Extraordinario, 149 Ordinario).
*(Nota: La tabla original contiene un desglose detallado de las vacantes del proceso extraordinario que aquí se presentan como un total por carrera para mantener la legibilidad).*

NOTA:
1. Las vacantes no cubiertas en el proceso CEPREUNAM y Extraordinario, serán adicionadas al número de vacantes del proceso ordinario.

5. DATOS EXTRA
- La Universidad Nacional de Moquegua (UNAM) es una institución pública de educación superior
-La universidad ofrece 2 examenes al año, donde las carreras disponibles son distintas en cada examen.
-Informacion sensible se le manda al estudiante por el correo directamente, despues de realizar su pago.
-No se pueden hacer modificaciones despues de haber realizado el pago.
-En preguntas relacionadas a pagos, derivar a contactos o a la pagina oficial.
-Para preguntas sobre inscripcion derivar a whatsapp o a la pagina oficial.

## FORMATO Y ESTRUCTURA DE LA RESPUESTA
Tu respuesta DEBE seguir esta estructura de formato para ser clara y visualmente atractiva:
1. **Cuerpo de la Respuesta:**
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

async def generar_respuesta_stream(pregunta: str, historial: list):
    # Recuperar contexto: los top-k chunks relevantes
    print("\n" + "="*80)
    print(f"🔍 NUEVA CONSULTA: {pregunta}")
    print("="*80)
    
    # Debug: Verificar si el query_engine está funcionando
    try:
        print(f"🔍 DEBUG: Ejecutando query...")
        # LlamaIndex soporta aquery para consultas asíncronas
        resultado = await query_engine.aquery(pregunta)
        print(f"🔍 DEBUG: Tipo de resultado: {type(resultado)}")
        print(f"🔍 DEBUG: Resultado tiene source_nodes: {hasattr(resultado, 'source_nodes')}")
        
        if hasattr(resultado, 'source_nodes'):
            print(f"🔍 DEBUG: Número de source_nodes: {len(resultado.source_nodes)}")
            if len(resultado.source_nodes) == 0:
                print("⚠️  WARNING: No se encontraron source_nodes")
                # Verificar si hay documentos en el índice
                try:
                    # Intentar acceder al vector store para diagnóstico
                    collection_count = chroma_collection.count()
                    print(f"🔍 DEBUG: Documentos en ChromaDB: {collection_count}")
                except Exception as e:
                    print(f"❌ Error accediendo a ChromaDB: {e}")
        
    except Exception as e:
        print(f"❌ Error ejecutando query: {e}")
        print(f"❌ Tipo de error: {type(e)}")
        import traceback
        traceback.print_exc()
        # Crear resultado vacío para continuar
        class EmptyResult:
            def __init__(self):
                self.source_nodes = []
        resultado = EmptyResult()
    
    # Mostrar chunks recuperados en terminal
    source_nodes = getattr(resultado, 'source_nodes', [])
    print(f"\n📚 CHUNKS RECUPERADOS ({len(source_nodes)} encontrados):")
    print("-"*60)
    
    for i, node in enumerate(source_nodes, 1):
        print(f"\n🔸 CHUNK #{i}:")
        print(f"   Score: {getattr(node, 'score', 'N/A')}")
        # Mostrar las primeras 200 caracteres del chunk
        node_text = getattr(node, 'node', node)
        if hasattr(node_text, 'text'):
            text_content = node_text.text
        elif hasattr(node_text, 'get_content'):
            text_content = node_text.get_content()
        else:
            text_content = str(node_text)
        
        text_preview = text_content[:200] + "..." if len(text_content) > 200 else text_content
        print(f"   Contenido: {text_preview}")
        
        # Mostrar metadata si está disponible
        if hasattr(node_text, 'metadata') and node_text.metadata:
            print(f"   Metadata: {node_text.metadata}")
        
        print("-"*40)
    
    print(f"\n✅ RESPUESTA GENERADA PARA: {pregunta}")
    print("="*80 + "\n")
    
    contexto = str(resultado)
    prompt = generar_prompt(pregunta, contexto, historial)
    async for chunk in llamar_llm_streaming(prompt):
        yield chunk

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


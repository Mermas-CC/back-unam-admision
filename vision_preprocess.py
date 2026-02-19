import os
import sys
import io
import time
from dotenv import load_dotenv
import pypdfium2 as pdfium
from PIL import Image
import google.generativeai as genai

# Cargar API Key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    print("❌ Error: GEMINI_API_KEY no configurada en el archivo .env")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)

def process_page_with_vision(image_bytes, page_num):
    """
    Envía la imagen de la página a Gemini para transcripción y OCR avanzado.
    """
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    prompt = f"""
Eres un experto en extracción de datos y OCR, especializado en preparar contenido para sistemas RAG (Retrieval-Augmented Generation). Se te proporciona una imagen de la PÁGINA {page_num} de un prospecto universitario.

TU MISIÓN:
1. TRANSCRIPCIÓN LITERAL: Transcribe el texto de forma literal manteniendo la estructura lógica (títulos, subtítulos, viñetas).
2. DISEÑO DE DOBLE COLUMNA: Si la página tiene dos columnas, léelas correctamente: primero toda la columna de la izquierda y luego toda la de la derecha. NO mezcles líneas.
3. INTERPRETACIÓN DE TABLAS (CRÍTICO): 
   - NO te limites a crear una tabla en Markdown.
   - Tu objetivo es EXPLICAR y NARRAR el contenido de la tabla de tal forma que un LLM pueda entender perfectamente cada dato sin ambigüedad.
   - Para cada fila/celda importante, genera una oración descriptiva. Ejemplo: "Para la carrera de Medicina en la Sede Moquegua, el número de vacantes para el examen ordinario es 11 y para CEPRE es 7, sumando un total de 35 ingresantes".
   - Si la tabla es muy grande, agrúpala de forma lógica pero asegúrate de que toda la información numérica sea transcrita narrativamente.
4. CONTEXTO: Si hay tablas de pagos o cronogramas, descríbelos como reglas de negocio. Ejemplo: "El derecho de examen para estudiantes de colegios nacionales es de S/ 350.00".
5. LIMPIEZA: Ignora elementos decorativos.

RESPUESTA (solo el contenido transcrito e interpretado narrativamente):
"""

    image_part = {
        "mime_type": "image/jpeg",
        "data": image_bytes
    }

    try:
        response = model.generate_content([prompt, image_part])
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Error al procesar página {page_num} con Gemini Vision: {str(e)}"

def vision_preprocess(pdf_path, output_path, start_page=1, end_page=None):
    """
    Convierte un PDF a imágenes y las procesa con Gemini Vision.
    """
    if not os.path.exists(pdf_path):
        print(f"❌ El archivo {pdf_path} no existe.")
        return

    print(f"🚀 Iniciando procesamiento VISION/OCR de: {pdf_path}")
    
    try:
        pdf = pdfium.PdfDocument(pdf_path)
        total_pages = len(pdf)
        
        # Ajustar rango
        actual_end = end_page if end_page and end_page <= total_pages else total_pages
        
        print(f"📊 Rango de páginas: {start_page} a {actual_end} (Total a procesar: {actual_end - start_page + 1})")

        with open(output_path, "w", encoding="utf-8") as out_file:
            for i in range(start_page - 1, actual_end):
                page_num = i + 1
                print(f"📸 Renderizando página {page_num}...")
                
                # Renderizar página (scale=3 para ~216 DPI, buen balance entre calidad y tamaño)
                page = pdf[i]
                bitmap = page.render(scale=3)
                pil_image = bitmap.to_pil()
                
                # Convertir a bytes (JPEG para reducir tamaño enviado a la API)
                img_byte_arr = io.BytesIO()
                pil_image.save(img_byte_arr, format='JPEG', quality=85)
                img_bytes = img_byte_arr.getvalue()
                
                print(f"🤖 Procesando página {page_num} con Gemini Vision...")
                transcription = process_page_with_vision(img_bytes, page_num)
                
                # Escribir al archivo
                out_file.write(f"\n\n--- INICIO PÁGINA {page_num} (VISION OCR) ---\n\n")
                out_file.write(transcription)
                out_file.write(f"\n\n--- FIN PÁGINA {page_num} ---\n")
                out_file.flush() # Guardar progreso por si falla
                
                print(f"✅ Página {page_num} completada.")
                
                # Pequeña pausa para evitar límites de cuota agresivos si es necesario
                # time.sleep(1)

        print(f"\n🏁 ¡Procesamiento finalizado con éxito!")
        print(f"📝 Resultado guardado en: {output_path}")

    except Exception as e:
        print(f"❌ Error crítico durante el procesamiento: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python vision_preprocess.py <input.pdf> <output.txt> [--start N] [--end M]")
        sys.exit(1)
        
    input_pdf = sys.argv[1]
    output_txt = sys.argv[2]
    
    start_page = 1
    end_page = None
    
    if "--start" in sys.argv:
        start_page = int(sys.argv[sys.argv.index("--start") + 1])
    if "--end" in sys.argv:
        end_page = int(sys.argv[sys.argv.index("--end") + 1])
        
    vision_preprocess(input_pdf, output_txt, start_page, end_page)

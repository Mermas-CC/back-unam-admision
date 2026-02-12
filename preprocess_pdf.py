import pdfplumber
import sys
import os
import re
from dotenv import load_dotenv
import google.generativeai as genai

# Cargar API Key desde .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def interpret_table_with_llm(table_data):
    """
    Envía los datos de una tabla a Gemini para que los interprete en un formato más usable.
    """
    if not GEMINI_API_KEY:
        return "⚠️ Error: GEMINI_API_KEY no configurada. No se pudo interpretar la tabla con LLM."

    # Convertir tabla (lista de listas) a una representación simple
    raw_table_str = ""
    for row in table_data:
        raw_table_str += "| " + " | ".join([str(cell).strip() if cell else "" for cell in row]) + " |\n"

    prompt = f"""
Eres un experto en extracción de datos y RAG (Retrieval Augmented Generation).
He extraído la siguiente tabla de un documento PDF, pero los datos crudos pueden ser difíciles de interpretar para un modelo de lenguaje en el futuro.

TU MISIÓN:
1. Analiza la estructura y el contenido de la tabla.
2. Convierte esta tabla en un formato narrativo, descriptivo o estructurado que sea EXTREMADAMENTE fácil de entender para un LLM.
3. Asegúrate de no perder ninguna información numérica o clave.
4. Si la tabla tiene encabezados claros, úsalos para dar contexto a cada valor.

TABLA CRUDA:
{raw_table_str}

RESPUESTA (solo el contenido interpretado, sin introducciones):
"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Error al procesar con LLM: {str(e)}"

def get_cutoff_page(pdf_path, total_pages):
    """
    Pregunta al usuario hasta qué página desea procesar con LLM.
    Las páginas posteriores se tratarán como anexos.
    """
    print(f"\n{'='*60}")
    print(f"📄 Archivo: {os.path.basename(pdf_path)}")
    print(f"📊 Total de páginas: {total_pages}")
    print(f"{'='*60}")
    
    while True:
        try:
            user_input = input(f"¿Hasta qué página deseas procesar con LLM? (1-{total_pages}): ").strip()
            cutoff = int(user_input)
            
            if 1 <= cutoff <= total_pages:
                print(f"✅ Páginas 1-{cutoff}: Procesamiento con LLM")
                if cutoff < total_pages:
                    print(f"📎 Páginas {cutoff+1}-{total_pages}: Anexos (sin LLM)")
                print(f"{'='*60}\n")
                return cutoff
            else:
                print(f"⚠️  Por favor, introduce un número entre 1 y {total_pages}")
        except ValueError:
            print("⚠️  Por favor, introduce un número válido")
        except KeyboardInterrupt:
            print("\n\n⚠️  Proceso interrumpido por el usuario")
            sys.exit(0)

def preprocess_pdf(pdf_path, output_path, use_llm=False, cutoff_page=None):
    """
    Lee un PDF, extrae texto y tablas, y guarda todo en un archivo de texto plano.
    Opcionalmente interpreta las tablas con un LLM hasta la página de corte.
    
    Args:
        pdf_path: Ruta al archivo PDF
        output_path: Ruta de salida para el archivo de texto
        use_llm: Si True, usa LLM para interpretar tablas
        cutoff_page: Página límite para usar LLM. Después de esta página, todo es anexo sin LLM
    """
    if not os.path.exists(pdf_path):
        print(f"❌ Error: El archivo '{pdf_path}' no existe.")
        return

    print(f"📄 Procesando PDF: {pdf_path} (Modo LLM: {'Activado' if use_llm else 'Desactivado'})...")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            
            # Si se usa LLM pero no se especificó cutoff, preguntar
            if use_llm and cutoff_page is None:
                cutoff_page = get_cutoff_page(pdf_path, total_pages)
            
            with open(output_path, "w", encoding="utf-8") as out_file:
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    print(f"📖 Procesando página {page_num}/{total_pages}...")
                    
                    # Determinar si esta página debe procesarse con LLM
                    is_appendix = use_llm and cutoff_page and page_num > cutoff_page
                    
                    if is_appendix:
                        print(f"  📎 Página de ANEXO. Extracción sin LLM.")
                    
                    # 1. Extraer texto plano
                    page_text = page.extract_text()
                    if page_text:
                        out_file.write(f"\n--- PÁGINA {page_num} ---\n\n")
                        out_file.write(page_text)
                        out_file.write("\n")

                    # 2. Extraer tablas
                    tables = page.extract_tables()
                    if tables:
                        for table_index, table in enumerate(tables):
                            out_file.write(f"\n[TABLA {table_index + 1} en página {page_num}]\n")
                            
                            # Solo usamos LLM si está activado Y no es una página de anexo
                            if use_llm and not is_appendix:
                                print(f"  🤖 Interpretando tabla {table_index + 1} con LLM...")
                                interpreted = interpret_table_with_llm(table)
                                out_file.write("\n--- Interpretación de Tabla por LLM ---\n")
                                out_file.write(interpreted)
                                out_file.write("\n---------------------------------------\n")
                            else:
                                # Modo normal o Anexo (Markdown simplificado)
                                if is_appendix:
                                    out_file.write("\n--- Tabla de Anexo (Sin interpretación LLM) ---\n")
                                for row in table:
                                    clean_row = [str(cell).replace('\n', ' ').strip() if cell else "" for cell in row]
                                    out_file.write("| " + " | ".join(clean_row) + " |\n")
                                if is_appendix:
                                    out_file.write("-----------------------------------------------\n")
                            out_file.write("\n")

        print(f"✅ ¡Éxito! El contenido se ha guardado en: {output_path}")

    except Exception as e:
        print(f"❌ Ocurrió un error inesperado al procesar {pdf_path}: {e}")

def batch_process(use_llm=False):
    """
    Procesa todos los PDFs en la carpeta 'pdf/' y los guarda en 'txt/'.
    Si use_llm es True, preguntará el límite de página para cada archivo.
    """
    input_dir = "pdf"
    output_dir = "txt"
    
    if not os.path.exists(input_dir):
        print(f"📁 Creando carpeta de entrada: {input_dir}")
        os.makedirs(input_dir)
        print(f"ℹ️  Coloca tus archivos PDF en '{input_dir}' y vuelve a ejecutar.")
        return

    if not os.path.exists(output_dir):
        print(f"📁 Creando carpeta de salida: {output_dir}")
        os.makedirs(output_dir)

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print(f"⚠️  No se encontraron archivos PDF en '{input_dir}'.")
        return

    print(f"\n🚀 Iniciando procesamiento por lote de {len(pdf_files)} archivos...\n")
    
    for pdf_file in pdf_files:
        input_path = os.path.join(input_dir, pdf_file)
        # Cambiamos extensión a .txt
        output_name = os.path.splitext(pdf_file)[0] + ".txt"
        output_path = os.path.join(output_dir, output_name)
        
        # En modo batch, cada archivo pide su propio cutoff si se usa LLM
        preprocess_pdf(input_path, output_path, use_llm, cutoff_page=None)
        print()  # Línea en blanco entre archivos
    
    print("🏁 Procesamiento por lote finalizado.")

if __name__ == "__main__":
    # Si se pasa --llm o no hay argumentos suficientes para modo p2p, usamos batch
    use_llm = "--llm" in sys.argv
    
    # Si no se especifican archivos de entrada/salida (o se usa --batch), activamos modo lote
    if len(sys.argv) < 3 or "--batch" in sys.argv:
        batch_process(use_llm)
    else:
        # Modo tradicional por si el usuario quiere procesar uno solo
        input_pdf = sys.argv[1]
        output_txt = sys.argv[2]
        preprocess_pdf(input_pdf, output_txt, use_llm)

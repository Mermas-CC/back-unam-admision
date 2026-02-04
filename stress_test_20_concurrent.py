import asyncio
import httpx
import time
import numpy as np

# ==========================
# CONFIGURACIÓN
# ==========================
API_URL = "https://back-unam-admision-1001169872215.us-central1.run.app/chat"
USUARIOS_POR_LOTE = 30
INTERVALO_ENTRE_LOTES = 1
NUMERO_DE_LOTES = 10  # Haremos 3 rondas por defecto
TIMEOUT = None

PREGUNTAS = [
    # ... (Keep existing prompt list logic implicitly by not changing lines 13-64 if possible, 
    # but I need to replace the config section which is at top, and the loop logic. 
    # Since I can't easily skip lines in replace_file_content mid-file without exact context,
    # I will replace the top config and the main async function.)
]
# I will make a smaller replacement for Config and a separate one for the function.

# Replacement 1: CONFIG


PREGUNTAS = [
"¿Cuánto cuesta el examen ordinario?",
"¿Cuánto cuesta el examen extraordinario?",
"¿Cuántas vacantes hay para Ingeniería de Sistemas e Informática?",
"¿Cuántas vacantes hay para Ingeniería Civil?",
"¿Cuántas vacantes hay para Administración?",
"¿Cuántas vacantes hay para Gestión Pública y Desarrollo Social?",
"¿Cuántas vacantes hay para Ingeniería Ambiental?",
"¿Cuántas vacantes hay para Ingeniería Pesquera?",
"¿Cuáles son las fechas del examen ordinario?",
"¿Cuáles son las fechas del examen extraordinario?",
"¿Hasta cuándo puedo inscribirme al examen de admisión?",
"¿Dónde puedo hacer el pago del derecho de examen?",
"¿Puedo pagar el examen en el Banco de la Nación?",
"¿Qué debo hacer después de realizar el pago del examen?",
"¿Dónde debo subir el voucher de pago?",
"¿Qué carreras se ofrecen en la sede Moquegua?",
"¿Qué carreras se ofrecen en la filial Ilo?",
"¿Qué carreras pertenecen al canal B?",
"¿Qué carreras pertenecen al canal C?",
"¿Cuándo se publican los resultados del examen ordinario?",
"¿Cuándo se publican los resultados del examen extraordinario?",
"¿Cuántas vacantes totales ofrece la universidad?",
"¿Qué documentos necesito para inscribirme al examen de admisión?",
"¿Puedo postular si soy trasladado de otra universidad?",
"¿Cuánto cuesta el examen para traslado externo?",
"¿Cuánto cuesta el examen para traslado interno?",
"¿Los deportistas destacados pagan el examen?",
"¿Las personas con discapacidad pagan el examen?",
"¿Dónde se realiza el examen de admisión extraordinario?",
"¿El examen extraordinario se rinde en la filial Ilo?",
"¿Qué es el CEPRE UNAM?",
"¿Cuántas vacantes hay para el CEPRE?",
"¿Las vacantes no cubiertas se acumulan al examen ordinario?",
"¿Cuántos exámenes de admisión hay al año?",
"¿Puedo modificar mis datos después de pagar el examen?",
"¿Cómo puedo comunicarme con la oficina de admisión?",
"¿Cuál es el número de contacto de la sede Moquegua?",
"¿Cuál es el número de contacto de la filial Ilo?",
"¿La universidad tiene atención por WhatsApp?",
"¿Dónde queda la Universidad Nacional de Moquegua?",
"¿Qué carreras de ingeniería ofrece la UNAM?",
"¿Qué carreras de ciencias sociales ofrece la UNAM?",
"¿El pago del examen es reembolsable?",
"¿Qué pasa si no alcanzo vacante en el examen extraordinario?",
"¿Puedo postular si soy víctima de terrorismo?",
"¿Cuál es el costo del examen para titulados universitarios?",
"¿En qué fechas se toma la identificación biométrica?",
"¿Dónde recojo mi carnet de postulante?",
"¿Qué hago si tengo problemas con mi inscripción?"

]

# ==========================
# MEDICIÓN INDIVIDUAL
# ==========================
# ==========================
# MEDICIÓN INDIVIDUAL
# ==========================
async def medir_usuario(client, user_id):
    pregunta = PREGUNTAS[user_id % len(PREGUNTAS)]
    historial = []

    payload = {
        "message": pregunta,
        "history": historial
    }

    inicio = time.perf_counter()
    primer_token = None
    status = 0
    error = None

    try:
        async with client.stream("POST", API_URL, json=payload) as response:
            status = response.status_code
            if status == 200:
                async for chunk in response.aiter_text():
                    if chunk.strip():
                        if primer_token is None:
                            primer_token = time.perf_counter() - inicio
            else:
                error = f"Status {status}"
    except Exception as e:
        error = str(e)
        status = -1

    total = time.perf_counter() - inicio

    # Si hubo error, tiempos son 0 o invalidos para metricas
    if error or status != 200:
        primer_token = 0
        total = 0

    # Print progress immediately
    estado_icon = "✅" if status == 200 else "❌"
    print(f"User {user_id:02d}: {estado_icon} (Total: {total:.2f}s)")
    
    return {
        "user": user_id,
        "ttft": primer_token,
        "total": total,
        "status": status,
        "error": error
    }

# ==========================
# STRESS TEST
# ==========================
async def stress_test():
    print(f"\n🚀 Lanzando prueba de carga: {NUMERO_DE_LOTES} lotes de {USUARIOS_POR_LOTE} usuarios (Intervalo: {INTERVALO_ENTRE_LOTES}s)...\n")
    
    todos_resultados = []

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for lote in range(NUMERO_DE_LOTES):
            print(f"--- Lote {lote + 1}/{NUMERO_DE_LOTES} ---")
            
            # Offset user IDs to be unique across batches
            offset = lote * USUARIOS_POR_LOTE
            tareas = [
                medir_usuario(client, offset + i)
                for i in range(USUARIOS_POR_LOTE)
            ]

            resultados_lote = await asyncio.gather(*tareas)
            todos_resultados.extend(resultados_lote)
            
            if lote < NUMERO_DE_LOTES - 1:
                print(f"⏳ Esperando {INTERVALO_ENTRE_LOTES} segundos para el siguiente lote...")
                await asyncio.sleep(INTERVALO_ENTRE_LOTES)

    resultados = todos_resultados
    exitosos = [r for r in resultados if r["status"] == 200]
    fallidos = [r for r in resultados if r["status"] != 200]
    
    ttfts = np.array([r["ttft"] for r in exitosos])
    totals = np.array([r["total"] for r in exitosos])

    # ==========================
    # RESULTADOS
    # ==========================
    print("\n📊 RESULTADOS POR USUARIO (Completo)")
    print("-" * 50)

    for r in resultados: # Mostrar todos
        estado = "✅" if r["status"] == 200 else "❌"
        info_extra = f"| Error: {r['error']}" if r["error"] else ""
        print(
            f"Usuario {r['user']:02d} | {estado} "
            f"TTFT: {r['ttft']:.3f}s | "
            f"Total: {r['total']:.3f}s {info_extra}"
        )

    print("\n📈 MÉTRICAS AGREGADAS")
    print("-" * 50)
    
    total_reqs = len(resultados)
    num_exitos = len(exitosos)
    num_fallos = len(fallidos)
    
    print(f"Usuarios simultáneos (Total): {USUARIOS_POR_LOTE * NUMERO_DE_LOTES}")
    print(f"TASA DE ÉXITO: {num_exitos}/{total_reqs} ({(num_exitos/total_reqs)*100:.1f}%)")
    print(f"FALLOS: {num_fallos}")
    
    if num_fallos > 0:
        print("\n🔍 DESGLOSE DE FALLOS:")
        for r in fallidos:
            print(f" - Usuario {r['user']}: {r['error']}")

    if num_exitos > 0:
        print(f"\n⏱️ TIEMPOS (Solo Exitosos):")
        print(f"TTFT promedio: {ttfts.mean():.3f}s")
        print(f"TTFT P50: {np.percentile(ttfts, 50):.3f}s")
        print(f"TTFT P95: {np.percentile(ttfts, 95):.3f}s")
        print(f"TTFT P99: {np.percentile(ttfts, 99):.3f}s")
        print()
        print(f"Total promedio: {totals.mean():.3f}s")
        print(f"Total P50: {np.percentile(totals, 50):.3f}s")
        print(f"Total P95: {np.percentile(totals, 95):.3f}s")
        print(f"Total P99: {np.percentile(totals, 99):.3f}s")
    else:
        print("\n⚠️ No hay métricas de tiempo porque todas las peticiones fallaron.")


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    asyncio.run(stress_test())


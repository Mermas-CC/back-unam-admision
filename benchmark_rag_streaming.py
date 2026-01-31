import asyncio
import httpx
import time
from statistics import mean

# ==========================
# CONFIGURACIÓN
# ==========================
API_URL = "http://localhost:8000/chat"
NUM_INTERACTIONS = 5

PREGUNTAS = [
    "¿Cuánto cuesta el examen ordinario?",
    "¿Cuántas vacantes hay para Ingeniería de Sistemas?",
    "¿Cuáles son las fechas del examen extraordinario?",
    "¿Dónde puedo hacer el pago del examen?",
    "¿Qué carreras hay en la sede Ilo?"
]

# ==========================
# FUNCIÓN DE PRUEBA
# ==========================
async def medir_interaccion(client, pregunta, historial):
    payload = {
        "message": pregunta,
        "history": historial
    }

    tiempos = {
        "ttft": None,
        "total": None
    }

    inicio = time.perf_counter()
    primer_token_recibido = False

    async with client.stream("POST", API_URL, json=payload) as response:
        async for chunk in response.aiter_text():
            if chunk.strip():
                if not primer_token_recibido:
                    tiempos["ttft"] = time.perf_counter() - inicio
                    primer_token_recibido = True

    tiempos["total"] = time.perf_counter() - inicio
    return tiempos


# ==========================
# BENCHMARK GENERAL
# ==========================
async def benchmark():
    resultados = []
    historial = []

    async with httpx.AsyncClient(timeout=None) as client:
        for i in range(NUM_INTERACTIONS):
            pregunta = PREGUNTAS[i % len(PREGUNTAS)]
            print(f"\n▶ Interacción {i + 1}: {pregunta}")

            tiempos = await medir_interaccion(client, pregunta, historial)
            resultados.append(tiempos)

            print(f"   ⏱️ Primer token: {tiempos['ttft']:.3f} s")
            print(f"   ⏱️ Tiempo total: {tiempos['total']:.3f} s")

            historial.append({"role": "user", "parts": [pregunta]})
            historial.append({"role": "model", "parts": ["respuesta simulada"]})

    # ==========================
    # RESULTADOS FINALES
    # ==========================
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE RENDIMIENTO")
    print("=" * 50)

    ttfts = [r["ttft"] for r in resultados]
    totals = [r["total"] for r in resultados]

    for i, r in enumerate(resultados, 1):
        print(
            f"Interacción {i}: "
            f"TTFT = {r['ttft']:.3f}s | "
            f"Total = {r['total']:.3f}s"
        )

    print("\n📈 PROMEDIOS")
    print(f"TTFT promedio: {mean(ttfts):.3f} s")
    print(f"Tiempo total promedio: {mean(totals):.3f} s")


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    asyncio.run(benchmark())


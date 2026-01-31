from locust import HttpUser, task, between
import time

class ChatUser(HttpUser):
    # Simula un tiempo de espera (think time) entre 5 y 15 segundos entre acciones
    wait_time = between(5, 15)
    
    @task
    def chat_interaction(self):
        """
        Simula un usuario enviando un mensaje al chat y esperando la respuesta completa.
        """
        headers = {"Content-Type": "application/json"}
        payload = {
            "message": "Hola, ¿cuándo es el examen de admisión y cuánto cuesta?",
            "history": [] 
        }
        
        start_time = time.time()
        
        # Usamos stream=True para manejar la respuesta SSE (Server-Sent Events)
        # name="/chat" agrupa todas las peticiones bajo esta etiqueta en las estadísticas
        with self.client.post("/chat", json=payload, headers=headers, stream=True, catch_response=True, name="/chat") as response:
            if response.status_code == 200:
                try:
                    # Es crucial consumir el stream para simular que el usuario recibe toda la respuesta.
                    # Si no iteramos, Locust podría contar la request como terminada apenas recibe los headers.
                    byte_count = 0
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            byte_count += len(chunk)
                    
                    # Opcional: Validar que recibimos datos reales
                    if byte_count < 10:
                        response.failure("Respuesta demasiado corta")
                    else:
                        response.success()
                        
                except Exception as e:
                    response.failure(f"Error leyendo stream: {e}")
            else:
                response.failure(f"Fallo con status: {response.status_code}")

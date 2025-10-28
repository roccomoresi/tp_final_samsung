"""
simular_usuario_local.py
------------------------
Simula interacciones de usuario directamente en el entorno local del bot.
No usa Telegram: ejecuta las funciones internas para generar registros realistas.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


import random
import time
from analysis.sentiment_analysis import analizar_sentimiento
from analysis.habit_recommender import generar_recomendacion
from utils.memory_manager import update_memory, get_memory, clear_memory
from utils.progress_logger import add_log
import json

# Cargar dataset local
def cargar_dataset():
    with open("data/dataset.json", "r", encoding="utf-8") as f:
        return json.load(f)

dataset = cargar_dataset()
USER_ID = 999999  # id simulado de usuario

mensajes = [
    "Hoy me siento estresado y sin energía.",
    "Me frustra no poder mantener hábitos saludables.",
    "Comí demasiado por ansiedad.",
    "Hoy me siento más tranquilo.",
    "Estoy contento con mis avances.",
    "Hoy estoy motivado y con ganas de comer mejor.",
    "Fue un día normal, sin muchos cambios.",
    "Estoy algo cansado, pero tranquilo.",
    "Estoy orgulloso de mí, mejoré mi alimentación.",
    "Hoy me siento desanimado, necesito motivación."
]

print("🤖 Simulando conversación del usuario con el bot...\n")

for mensaje in mensajes:
    sentimiento = analizar_sentimiento(mensaje)
    memoria_anterior = get_memory(USER_ID)

    if memoria_anterior:
        anterior = memoria_anterior["sentimiento"]
        if anterior == "NEG" and sentimiento == "POS":
            print("🌞 Me alegra verte mejor que antes.")
        elif anterior == "POS" and sentimiento == "NEG":
            print("💛 Te noto más apagado, pero tranquilo, eso también pasa.")

    respuesta = generar_recomendacion(mensaje, sentimiento, USER_ID, dataset)

    print(f"🗣️ Usuario: {mensaje}")
    print(f"🤖 Bot: {respuesta}\n")

    update_memory(USER_ID, sentimiento, respuesta)
    add_log(USER_ID, mensaje, sentimiento, respuesta)
    time.sleep(random.uniform(0.5, 1.2))

print("\n✅ Simulación completada. Revisá data/user_logs.json para ver el historial generado.")

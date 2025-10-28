"""
emotion_trends.py
-----------------
Lee el historial de progreso (user_logs.json) y genera un gráfico
con la evolución de los estados emocionales del usuario a lo largo del tiempo.
"""

import json
import matplotlib.pyplot as plt
from datetime import datetime

LOG_FILE = "data/user_logs.json"

def cargar_logs():
    """Carga los registros del archivo JSON."""
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error al cargar logs: {e}")
        return []

def generar_grafico(user_id=None):
    """Genera un gráfico de líneas mostrando la evolución emocional."""
    logs = cargar_logs()
    if not logs:
        print("❌ No hay registros para mostrar.")
        return

    # Filtrar por usuario (opcional)
    if user_id:
        logs = [log for log in logs if log["user_id"] == str(user_id)]

    if not logs:
        print(f"⚠️ No se encontraron registros para el usuario {user_id}.")
        return

    # Convertir fechas y sentimientos a valores numéricos
    fechas = [datetime.strptime(log["fecha"], "%Y-%m-%d %H:%M:%S") for log in logs]
    sentimientos = [log["sentimiento"] for log in logs]

    valores = []
    for s in sentimientos:
        if s == "POS":
            valores.append(1)
        elif s == "NEG":
            valores.append(-1)
        else:
            valores.append(0)

    # Crear el gráfico
    plt.figure(figsize=(9, 5))
    plt.plot(fechas, valores, marker="o", linestyle="-", linewidth=2)
    plt.title("Evolución Emocional del Usuario", fontsize=14)
    plt.xlabel("Fecha y hora", fontsize=12)
    plt.ylabel("Sentimiento (1=Positivo, 0=Neutro, -1=Negativo)", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generar_grafico()

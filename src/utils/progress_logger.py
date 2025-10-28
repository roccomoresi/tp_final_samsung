"""
progress_logger.py
------------------
Registra todas las interacciones de los usuarios (fecha, emoción, mensaje, recomendación)
para generar un historial de progreso.
"""

import json
import os
from datetime import datetime

LOG_FILE = "data/user_logs.json"

def _load_logs():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def _save_logs(logs):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

def add_log(user_id, texto, sentimiento, recomendacion):
    """Agrega una nueva entrada al historial de progreso."""
    logs = _load_logs()
    log_entry = {
        "user_id": str(user_id),
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mensaje": texto,
        "sentimiento": sentimiento,
        "recomendacion": recomendacion
    }
    logs.append(log_entry)
    _save_logs(logs)
    print(f"🗒️ Registro agregado → {user_id}: {sentimiento}")

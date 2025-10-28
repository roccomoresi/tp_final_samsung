"""
memory_manager.py
-----------------
Gestiona la memoria temporal del bot por usuario.
Guarda el último sentimiento detectado y la última recomendación entregada.
"""

import json
import os

MEMORY_FILE = "data/user_memory.json"


def _load_memory():
    """Carga la memoria desde disco (si existe)."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def _save_memory(memory):
    """Guarda la memoria actualizada en disco."""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def update_memory(user_id, sentimiento, recomendacion):
    """Actualiza el estado emocional y última recomendación del usuario."""
    memory = _load_memory()
    user_id = str(user_id)

    memory[user_id] = {
        "sentimiento": sentimiento,
        "ultima_recomendacion": recomendacion
    }

    _save_memory(memory)


def get_memory(user_id):
    """Obtiene la memoria del usuario si existe."""
    memory = _load_memory()
    return memory.get(str(user_id), None)


def clear_memory(user_id):
    """Elimina la memoria de un usuario específico."""
    memory = _load_memory()
    user_id = str(user_id)
    if user_id in memory:
        del memory[user_id]
        _save_memory(memory)

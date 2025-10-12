user_memory = {}

def get_user_history(user_id):
    """Devuelve el historial de conversación de un usuario."""
    return user_memory.setdefault(user_id, [])

def add_message(user_id, role, content):
    """Agrega un mensaje al historial."""
    user_memory.setdefault(user_id, []).append({"role": role, "content": content})

def clear_memory(user_id):
    """Borra la memoria de un usuario."""
    user_memory[user_id] = []
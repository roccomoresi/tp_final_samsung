"""
habit_recommender.py
--------------------
Genera recomendaciones personalizadas basadas en el sentimiento detectado
y la información del dataset.json.
"""

import random


def generar_recomendacion(texto, sentimiento, user_id, dataset=None):
    """
    Genera una recomendación personalizada basada en el sentimiento detectado
    y la información del dataset.json.
    """
    if dataset is None:
        return "⚠️ No se pudo acceder a los datos de recomendaciones."

    # Obtener todas las recomendaciones disponibles
    recomendaciones = dataset.get("recomendaciones", {})

    # Buscar si alguna palabra clave del texto coincide con las del dataset
    texto_lower = texto.lower()
    for clave in recomendaciones.keys():
        if clave in texto_lower:
            respuestas = recomendaciones[clave]
            if isinstance(respuestas, list):
                return random.choice(respuestas)
            else:
                return respuestas

    # Si no hay coincidencia directa, usar el sentimiento detectado
    if sentimiento == "NEG":
        posibles = ["ansiedad", "estrés", "frustración", "culpa"]
    elif sentimiento == "POS":
        posibles = ["motivación", "autocuidado"]
    else:
        posibles = ["descanso", "alimentacion", "bienestar"]

    # Elegir aleatoriamente entre esas claves disponibles
    for clave in posibles:
        if clave in recomendaciones:
            return recomendaciones[clave]

    # Si no hay ninguna recomendación aplicable
    return "🌱 Recordá que cada paso cuenta. Cuidarte también es escucharte 💛"

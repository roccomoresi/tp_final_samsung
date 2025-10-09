"""
habit_recommender.py
--------------------
Genera recomendaciones personalizadas de bienestar alimenticio
según el texto del usuario y su estado emocional detectado.
"""

import random

# Diccionario base de respuestas organizadas por sentimiento
RECOMENDACIONES = {
    "positivo": [
        "👏 ¡Excelente actitud! Seguí así, recordá que la constancia es la clave 🥗",
        "💪 Me encanta tu energía. Aprovechá para planear una comida balanceada hoy.",
        "🌞 Estás en un buen momento, no olvides hidratarte bien y disfrutar del proceso."
    ],
    "neutro": [
        "🙂 Recordá escuchar a tu cuerpo. ¿Tenés hambre real o es más por hábito?",
        "🤔 Podés aprovechar este momento para elegir algo liviano y nutritivo.",
        "🧘‍♂️ Comer despacio y sin distracciones ayuda a conectar con tus sensaciones."
    ],
    "negativo": [
        "😔 Entiendo cómo te sentís. A veces la ansiedad se disfraza de hambre, intentá respirar profundo antes de comer.",
        "🍵 Cuando sientas ansiedad, probá tomar agua o un té antes de decidir qué comer.",
        "🫶 No te castigues por tener un mal día. Lo importante es volver a elegir bien en la próxima comida."
    ]
}

# Palabras clave para detectar contextos específicos
CONTEXTOS = {
    "ansiedad": "La ansiedad por comer es común. Tratá de identificar si tu cuerpo realmente tiene hambre o si busca consuelo. Respirá profundo antes de decidir.",
    "dulce": "Si te dan ganas de algo dulce, podés optar por frutas o yogures naturales. Son opciones más saludables 🍓",
    "hambre": "Comer con hambre real es importante. Elegí alimentos que te den energía sostenida: proteínas, frutas, cereales integrales 🍎",
    "triste": "Cuando te sientas bajón, tratá de no refugiarte en la comida. A veces una caminata corta o hablar con alguien puede ayudarte 🌱",
    "feliz": "Si estás de buen ánimo, aprovechá para cocinar algo saludable que te guste mucho. Disfrutar también es parte del bienestar 🍽️"
}


def detectar_contexto(texto_usuario: str) -> str:
    """
    Detecta si el texto del usuario menciona alguna palabra clave
    asociada a un contexto alimenticio o emocional.
    """
    texto = texto_usuario.lower()
    for palabra, respuesta in CONTEXTOS.items():
        if palabra in texto:
            return respuesta
    return ""


def generar_recomendacion(texto_usuario: str, sentimiento: str) -> str:
    """
    Genera una recomendación según el sentimiento detectado y el contenido del texto.
    """
    sentimiento = sentimiento.lower()

    # Buscar contexto específico (ej: "ansiedad", "dulce", etc.)
    contexto = detectar_contexto(texto_usuario)

    # Elegir recomendación base según sentimiento
    if "pos" in sentimiento:
        base = random.choice(RECOMENDACIONES["positivo"])
    elif "neg" in sentimiento:
        base = random.choice(RECOMENDACIONES["negativo"])
    else:
        base = random.choice(RECOMENDACIONES["neutro"])

    # Combinar ambas partes si hay contexto detectado
    if contexto:
        return f"{base}\n\n💡 {contexto}"
    else:
        return base


if __name__ == "__main__":
    # Ejemplo de prueba
    ejemplo_texto = "Me siento ansioso por comer algo dulce"
    ejemplo_sentimiento = "negativo"
    print(generar_recomendacion(ejemplo_texto, ejemplo_sentimiento))

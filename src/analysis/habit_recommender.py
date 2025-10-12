# src/analysis/habit_recommender.py
from groq import Groq
from utils.memory_manager import get_user_history, add_message
import os
from dotenv import load_dotenv

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generar_recomendacion(texto: str, sentimiento: str, user_id: int) -> str:
    """
    Genera una recomendación personalizada combinando:
    - texto del usuario
    - su emoción detectada
    - historial de conversación
    """
    history = get_user_history(user_id)
    add_message(user_id, "user", texto)

    tono = {
        "POS": "El usuario se siente positivo y motivado 😄. Reforzá hábitos saludables y metas nuevas.",
        "NEG": "El usuario se siente decaído o frustrado 😔. Sé empático y alentador, ofrecé pasos pequeños.",
        "NEU": "El usuario está neutro 😐. Usá un tono tranquilo y profesional, proponé algo equilibrado."
    }.get(sentimiento, "El usuario tiene un estado emocional neutro.")

    system_prompt = f"""
Sos un coach virtual de bienestar y alimentación consciente 🍎.
Tu misión es ayudar al usuario a mejorar sus hábitos alimenticios y emocionales.

Indicaciones:
- Sé empático, cálido y motivador.
- Basate en su estado emocional: {tono}
- Respondé con consejos prácticos y personalizados.
- Usá lenguaje natural, cercano y con emojis relacionados a salud y comida.
- No des consejos médicos; solo hábitos saludables y motivación.
    """

    chat_completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.6,
        max_tokens=500,
        messages=[{"role": "system", "content": system_prompt}] + history
    )

    respuesta = chat_completion.choices[0].message.content.strip()
    add_message(user_id, "assistant", respuesta)
    return respuesta

"""
bot_voz.py
----------
Bot de Telegram con reconocimiento de voz, análisis de sentimientos
y recomendaciones de bienestar alimenticio personalizadas.
Ahora con memoria y tono adaptativo. 🧠🍎
"""

import os
import time
import json
import telebot as tlb
from dotenv import load_dotenv
from groq import Groq
import base64

# Importaciones del proyecto
from analysis.sentiment_analysis import analizar_sentimiento
from analysis.habit_recommender import generar_recomendacion
from utils.memory_manager import clear_memory, update_memory, get_memory
from utils.progress_logger import add_log


# ==============================
# 📦 CARGAR DATASET LOCAL
# ==============================
def cargar_dataset():
    try:
        with open("data/dataset.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error al cargar dataset: {e}")
        return {}

dataset = cargar_dataset()

# ==============================
# 🔧 CONFIGURACIÓN
# ==============================
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not TELEGRAM_TOKEN or not GROQ_API_KEY:
    raise ValueError("❌ Falta el TOKEN o la API Key en el archivo .env")

bot = tlb.TeleBot(TELEGRAM_TOKEN)
groq_client = Groq(api_key=GROQ_API_KEY)


# ==============================
# 🎙️ FUNCIÓN DE AUDIO → TEXTO
# ==============================
def transcribir_audio(message: tlb.types.Message) -> str:
    """Convierte audio a texto usando Whisper."""
    try:
        file_info = bot.get_file(message.voice.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        temp_file = "temp_voice.ogg"

        with open(temp_file, "wb") as f:
            f.write(downloaded_file)

        with open(temp_file, "rb") as file:
            transcripcion = groq_client.audio.transcriptions.create(
                file=(temp_file, file.read()),
                model="whisper-large-v3-turbo",
                prompt="Usuario hablando en español sobre alimentación o emociones",
                response_format="json",
                language="es",
                temperature=0.8
            )

        os.remove(temp_file)
        return transcripcion.text
    except Exception as e:
        print(f"Error al transcribir: {e}")
        return None


# ==============================
# 🤖 MANEJADORES
# ==============================
@bot.message_handler(commands=["start", "reset"])
def send_welcome(message: tlb.types.Message):
    """Mensaje inicial o reseteo de memoria."""
    clear_memory(message.from_user.id)
    bienvenida = (
        "🌱 ¡Hola! Soy tu asistente de bienestar alimenticio. 🧠🍎\n\n"
        "Contame cómo te sentís o mandame un audio. "
        "Te voy a ayudar con consejos personalizados sobre tus hábitos, emociones y alimentación. 💬✨"
    )
    print(f"👤 Nuevo usuario conectado: {message.from_user.id}")

    bot.reply_to(message, bienvenida)


@bot.message_handler(content_types=["text"])
def handle_text_message(message: tlb.types.Message):
    """Procesa mensajes de texto."""
    texto = message.text
    user_id = message.from_user.id
    bot.send_chat_action(message.chat.id, "typing")

    sentimiento = analizar_sentimiento(texto)

    # 🧠 Comparar con memoria previa
    memoria_anterior = get_memory(user_id)
    if memoria_anterior:
        anterior = memoria_anterior["sentimiento"]
        if anterior == "NEG" and sentimiento == "POS":
            bot.send_message(message.chat.id, "🌞 Me alegra verte mejor que antes.")
        elif anterior == "POS" and sentimiento == "NEG":
            bot.send_message(message.chat.id, "💛 Te noto más apagado, pero tranquilo, eso también pasa.")

    # 💬 Generar respuesta principal
    respuesta = generar_recomendacion(texto, sentimiento, user_id, dataset)
    bot.reply_to(message, respuesta)

    # 💾 Guardar nuevo estado en memoria
    update_memory(user_id, sentimiento, respuesta)
    add_log(user_id, texto, sentimiento, respuesta)
    print(f"🧠 Memoria actualizada para {user_id}: {sentimiento}")


@bot.message_handler(content_types=["voice"])
def handle_voice_message(message: tlb.types.Message):
    """Procesa mensajes de voz."""
    user_id = message.from_user.id
    bot.send_chat_action(message.chat.id, "typing")

    texto = transcribir_audio(message)
    print(f"🎧 Transcripción detectada: {texto}")

    if not texto:
        bot.reply_to(message, "❌ No pude entender el audio. Probá de nuevo por favor.")
        return

    sentimiento = analizar_sentimiento(texto)

    # 🧠 Comparar con memoria previa
    memoria_anterior = get_memory(user_id)
    if memoria_anterior:
        anterior = memoria_anterior["sentimiento"]
        if anterior == "NEG" and sentimiento == "POS":
            bot.send_message(message.chat.id, "🌞 Me alegra verte mejor que antes.")
        elif anterior == "POS" and sentimiento == "NEG":
            bot.send_message(message.chat.id, "💛 Te noto más apagado, pero tranquilo, eso también pasa.")

    respuesta = generar_recomendacion(texto, sentimiento, user_id, dataset)
    bot.reply_to(message, respuesta)

    update_memory(user_id, sentimiento, respuesta)
    add_log(user_id, texto, sentimiento, respuesta)
    print(f"🧠 Memoria actualizada para {user_id}: {sentimiento}")



    # ==============================
# 🖼️ MANEJADOR DE IMÁGENES
# ==============================

@bot.message_handler(content_types=["photo"])
def handle_photo_message(message: tlb.types.Message):
    """Procesa imágenes enviadas por el usuario."""
    bot.send_chat_action(message.chat.id, "typing")

    try:
        foto = message.photo[-1]  # la de mayor resolución
        file_info = bot.get_file(foto.file_id)
        archivo_descargado = bot.download_file(file_info.file_path)

        imagen_base64 = imagen_a_base64(archivo_descargado)
        if not imagen_base64:
            bot.reply_to(message, "❌ No pude procesar la imagen.")
            return

        descripcion = describir_imagen_con_groq(imagen_base64)
        if descripcion:
            bot.reply_to(message, f"🖼️ *Descripción de la imagen:*\n\n{descripcion}", parse_mode="Markdown")
        else:
            bot.reply_to(message, "❌ No pude analizar la imagen. Intentá con otra.")
    except Exception as e:
        print(f"Error en handle_photo_message: {e}")
        bot.reply_to(message, "⚠️ Ocurrió un error al analizar tu imagen.")



    # ==============================
# 🖼️ FUNCIÓN: IMAGEN → DESCRIPCIÓN
# ==============================

def imagen_a_base64(ruta_o_bytes_imagen):
    """Convierte una imagen a base64."""
    try:
        if isinstance(ruta_o_bytes_imagen, bytes):
            return base64.b64encode(ruta_o_bytes_imagen).decode("utf-8")
        else:
            with open(ruta_o_bytes_imagen, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error al convertir imagen: {e}")
        return None


def describir_imagen_con_groq(imagen_base64):
    """Envía la imagen a Groq Vision y devuelve una descripción."""
    try:
        completion = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe detalladamente esta imagen en español."},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{imagen_base64}"}}
                    ],
                }
            ],
            temperature=0.7,
            max_tokens=800,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error al describir imagen: {e}")
        return None


# ==============================
# 🚀 INICIO DEL BOT
# ==============================
if __name__ == "__main__":
    print("🤖 Bot de bienestar alimenticio iniciado con inteligencia mejorada.")
    try:
        bot.infinity_polling(timeout=20, long_polling_timeout=10)
    except KeyboardInterrupt:
        print("\n🛑 Bot detenido manualmente por el usuario.")
    except Exception as e:
        print(f"Error en el bot: {e}")
        print("Reiniciando en 5 segundos...")
        time.sleep(5)

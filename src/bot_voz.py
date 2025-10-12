"""
bot_voz.py
----------
Bot de Telegram con reconocimiento de voz, análisis de sentimientos
y recomendaciones de bienestar alimenticio personalizadas.
Ahora con memoria y tono adaptativo. 🧠🍎
"""

import os
import time
import telebot as tlb
from dotenv import load_dotenv
from groq import Groq

# Importaciones del proyecto
from analysis.sentiment_analysis import analizar_sentimiento
from analysis.habit_recommender import generar_recomendacion
from utils.memory_manager import clear_memory

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
    bot.reply_to(message, bienvenida)

@bot.message_handler(content_types=["text"])
def handle_text_message(message: tlb.types.Message):
    """Procesa mensajes de texto."""
    texto = message.text
    user_id = message.from_user.id
    bot.send_chat_action(message.chat.id, "typing")

    sentimiento = analizar_sentimiento(texto)
    respuesta = generar_recomendacion(texto, sentimiento, user_id)
    bot.reply_to(message, respuesta)

@bot.message_handler(content_types=["voice"])
def handle_voice_message(message: tlb.types.Message):
    """Procesa mensajes de voz."""
    user_id = message.from_user.id
    bot.send_chat_action(message.chat.id, "typing")

    texto = transcribir_audio(message)
    if not texto:
        bot.reply_to(message, "❌ No pude entender el audio. Probá de nuevo por favor.")
        return

    sentimiento = analizar_sentimiento(texto)
    respuesta = generar_recomendacion(texto, sentimiento, user_id)
    bot.reply_to(message, respuesta)

# ==============================
# 🚀 INICIO DEL BOT
# ==============================
if __name__ == "__main__":
    print("🤖 Bot de bienestar alimenticio iniciado con inteligencia mejorada.")
    while True:
        try:
            bot.polling(non_stop=True, interval=0, timeout=20)
        except Exception as e:
            print(f"Error en el bot: {e}")
            print("Reiniciando en 5 segundos...")
            time.sleep(5)

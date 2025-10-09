"""
bot_voz.py
----------
Bot de Telegram con reconocimiento de voz, análisis de sentimientos
y recomendaciones de bienestar alimenticio personalizadas.
"""

import os
import time
from groq import Groq
import telebot as tlb
from dotenv import load_dotenv

# 🔹 Importaciones del proyecto
from src.analysis.sentiment_analysis import analizar_sentimiento
from src.analysis.habit_recommender import generar_recomendacion

# ==============================
# 🔧 CONFIGURACIÓN INICIAL
# ==============================
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not TELEGRAM_TOKEN:
    raise ValueError("❌ Falta el TELEGRAM_TOKEN en el archivo .env")
if not GROQ_API_KEY:
    raise ValueError("❌ Falta el GROQ_API_KEY en el archivo .env")

bot = tlb.TeleBot(TELEGRAM_TOKEN)
groq_client = Groq(api_key=GROQ_API_KEY)

# ==============================
# 🎙️ FUNCIONES DE TRANSCRIPCIÓN
# ==============================
def transcribir_audio(message: tlb.types.Message) -> str:
    """
    Descarga y transcribe un audio enviado por el usuario
    usando el modelo Whisper de Groq.
    """
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
# 🤖 MANEJADORES DEL BOT
# ==============================
@bot.message_handler(commands=["start"])
def send_welcome(message: tlb.types.Message):
    """
    Mensaje de bienvenida al iniciar el bot.
    """
    bienvenida = (
        "🌱 ¡Hola! Soy tu asistente de bienestar alimenticio.\n\n"
        "Podés contarme cómo te sentís o enviarme un audio, y te voy a ayudar "
        "con consejos personalizados sobre tu alimentación, emociones y hábitos. 🍎✨"
    )
    bot.reply_to(message, bienvenida)


@bot.message_handler(content_types=["text"])
def handle_text_message(message: tlb.types.Message):
    """
    Maneja mensajes de texto: analiza sentimiento y genera recomendación.
    """
    texto = message.text
    bot.send_chat_action(message.chat.id, "typing")

    sentimiento = analizar_sentimiento(texto)
    respuesta = generar_recomendacion(texto, sentimiento)

    bot.reply_to(message, respuesta)


@bot.message_handler(content_types=["voice"])
def handle_voice_message(message: tlb.types.Message):
    """
    Maneja mensajes de voz: transcribe, analiza y responde.
    """
    bot.send_chat_action(message.chat.id, "typing")

    texto = transcribir_audio(message)
    if not texto:
        bot.reply_to(message, "❌ No pude entender el audio. Por favor, probá de nuevo.")
        return

    sentimiento = analizar_sentimiento(texto)
    respuesta = generar_recomendacion(texto, sentimiento)

    bot.reply_to(message, respuesta)


# ==============================
# 🚀 INICIO DEL BOT
# ==============================
if __name__ == "__main__":
    print("🤖 Bot de bienestar alimenticio iniciado correctamente.")
    while True:
        try:
            bot.polling(non_stop=True, interval=0, timeout=20)
        except Exception as e:
            print(f"Error en el bot: {e}")
            print("Reiniciando en 5 segundos...")
            time.sleep(5)

"""
bot_voz.py 
--------------------------------------
Bot de Telegram con las 4 funcionalidades de IA requeridas:
1. ✅ Análisis de sentimientos (Transformers)
2. ✅ Respuestas automáticas con dataset propio
3. ✅ Audio → Texto optimizado (Groq Whisper)
4. ✅ Análisis de imágenes (Groq Vision)

Proyecto Final - Samsung Innovation Campus 2025
Equipo: Guadalupe · Fabiana · Rocco
"""

import os
import time
import json
import tempfile
import telebot as tlb
from dotenv import load_dotenv
from groq import Groq

# Importaciones del proyecto
from analysis.sentiment_analysis import analizar_sentimiento
from analysis.habit_recommender import generar_recomendacion
from analysis.image_analysis import analizar_imagen_comida, generar_feedback_visual
from utils.memory_manager import clear_memory, update_memory, get_memory
from utils.progress_logger import add_log


# ==============================
# 📦 CARGAR DATASET LOCAL
# ==============================
def cargar_dataset():
    """Carga el dataset de recomendaciones personalizadas."""
    try:
        with open("data/dataset.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error al cargar dataset: {e}")
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

# Crear directorios necesarios
os.makedirs("data/temp", exist_ok=True)


# ==============================
# 🎙️ AUDIO → TEXTO OPTIMIZADO
# ==============================
def speech_to_text(audio_bytes: bytes) -> str:
    """
    Convierte bytes de audio de Telegram a texto usando Groq Whisper.
    Método optimizado sin necesidad de FFmpeg.
    
    Args:
        audio_bytes: Datos binarios del audio
    
    Returns:
        str: Transcripción del audio en español
    """
    try:
        # Guardar temporalmente el audio
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name

        # Transcribir con Groq Whisper
        with open(temp_audio_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=audio_file,
                language="es",
                prompt="Usuario hablando sobre alimentación o emociones",
                response_format="json",
                temperature=0.7
            )

        # Limpiar archivo temporal
        os.remove(temp_audio_path)
        
        texto = transcription.text
        print(f"🎤 Audio transcrito: '{texto[:50]}...'")
        return texto
        
    except Exception as e:
        print(f"❌ Error en transcripción: {e}")
        return None


# ==============================
# 🤖 MANEJADORES DE COMANDOS
# ==============================
@bot.message_handler(commands=["start", "reset"])
def send_welcome(message: tlb.types.Message):
    """Mensaje inicial o reseteo de memoria."""
    user_id = message.from_user.id
    username = message.from_user.username or "Usuario"
    
    clear_memory(user_id)
    
    bienvenida = (
        f"🌱 *¡Hola {username}! Soy Menta, tu asistente de bienestar alimenticio.* 🧠🍎\n\n"
        "Podés interactuar conmigo de 3 formas:\n\n"
        "💬 *Texto:* Contame cómo te sentís\n"
        "🎤 *Audio:* Mandame un mensaje de voz\n"
        "📸 *Foto:* Enviame una imagen de tu comida\n\n"
        "_Uso IA para analizar tus emociones y darte consejos personalizados._ ✨"
    )
    
    print(f"👤 Usuario conectado: {user_id} (@{username})")
    bot.reply_to(message, bienvenida, parse_mode="Markdown")


@bot.message_handler(commands=["ayuda", "help"])
def send_help(message: tlb.types.Message):
    """Muestra comandos disponibles y funcionalidades."""
    ayuda = (
        "📚 *Comandos disponibles:*\n\n"
        "/start - Iniciar conversación\n"
        "/reset - Reiniciar memoria del bot\n"
        "/progreso - Ver tu evolución emocional\n"
        "/stats - Estadísticas de uso\n"
        "/ayuda - Mostrar esta ayuda\n\n"
        "🤖 *Funcionalidades de IA:*\n"
        "• Análisis de sentimientos (NLP)\n"
        "• Reconocimiento de voz (Whisper)\n"
        "• Análisis de imágenes (Vision AI)\n"
        "• Recomendaciones personalizadas\n\n"
        "💡 *Tip:* Hablame con naturalidad, entiendo español argentino perfectamente."
    )
    bot.reply_to(message, ayuda, parse_mode="Markdown")


@bot.message_handler(commands=["progreso", "progress"])
def mostrar_progreso(message: tlb.types.Message):
    """Muestra un resumen detallado del progreso emocional."""
    user_id = str(message.from_user.id)
    
    try:
        # Cargar logs
        with open("data/user_logs.json", "r", encoding="utf-8") as f:
            logs = json.load(f)
        
        # Filtrar por usuario
        user_logs = [log for log in logs if log["user_id"] == user_id]
        
        if not user_logs:
            bot.reply_to(message, "📊 Aún no tenés registros. Empezá a contarme cómo te sentís.")
            return
        
        # Estadísticas
        positivos = sum(1 for log in user_logs if log["sentimiento"] == "POS")
        negativos = sum(1 for log in user_logs if log["sentimiento"] == "NEG")
        neutros = sum(1 for log in user_logs if log["sentimiento"] == "NEU")
        total = len(user_logs)
        
        porc_pos = (positivos / total) * 100
        porc_neg = (negativos / total) * 100
        porc_neu = (neutros / total) * 100
        
        # Último sentimiento
        ultimo = user_logs[-1]
        fecha_ultimo = ultimo["fecha"].split()[0]  # Solo fecha
        
        # Construir mensaje
        resumen = (
            f"📊 *Tu progreso emocional:*\n\n"
            f"📈 Total de interacciones: *{total}*\n\n"
            f"✅ Positivos: {positivos} ({porc_pos:.1f}%)\n"
            f"⚠️ Negativos: {negativos} ({porc_neg:.1f}%)\n"
            f"➖ Neutros: {neutros} ({porc_neu:.1f}%)\n\n"
            f"🕐 Última interacción: {fecha_ultimo}\n\n"
        )
        
        # Motivación personalizada
        if porc_pos >= 70:
            resumen += "🌟 *¡Excelente!* Tu estado emocional es muy positivo. Seguí así."
        elif porc_pos >= 50:
            resumen += "💪 *¡Muy bien!* Vas por buen camino. Seguí trabajando en tu bienestar."
        elif porc_pos >= 30:
            resumen += "🌱 *Avanzando:* Hay altibajos, pero estás progresando. Seguí adelante."
        else:
            resumen += "💙 *Estoy acá para ayudarte.* Juntos vamos a mejorar tus hábitos paso a paso."
        
        bot.reply_to(message, resumen, parse_mode="Markdown")
        
    except FileNotFoundError:
        bot.reply_to(message, "📊 Aún no hay datos para mostrar. Empezá a interactuar conmigo.")
    except Exception as e:
        print(f"❌ Error al mostrar progreso: {e}")
        bot.reply_to(message, "⚠️ No pude cargar tu progreso. Intentá más tarde.")


@bot.message_handler(commands=["stats"])
def mostrar_stats(message: tlb.types.Message):
    """Estadísticas generales del bot (solo para admins o todos los usuarios)."""
    try:
        with open("data/user_logs.json", "r", encoding="utf-8") as f:
            logs = json.load(f)
        
        total_interacciones = len(logs)
        usuarios_unicos = len(set(log["user_id"] for log in logs))
        
        stats = (
            f"📊 *Estadísticas del Bot:*\n\n"
            f"👥 Usuarios únicos: {usuarios_unicos}\n"
            f"💬 Total de interacciones: {total_interacciones}\n"
            f"🤖 Modelo de sentimientos: RoBERTuito\n"
            f"🎤 Modelo de voz: Whisper V3 Turbo\n"
            f"📸 Modelo de visión: Llama 3.2 90B Vision\n\n"
            f"_Proyecto Samsung Innovation Campus 2025_"
        )
        
        bot.reply_to(message, stats, parse_mode="Markdown")
        
    except Exception as e:
        print(f"❌ Error en stats: {e}")
        bot.reply_to(message, "⚠️ No pude cargar las estadísticas.")


# ==============================
# 💬 MENSAJES DE TEXTO
# ==============================
@bot.message_handler(content_types=["text"])
def handle_text_message(message: tlb.types.Message):
    """Procesa mensajes de texto con análisis de sentimientos."""
    texto = message.text
    user_id = message.from_user.id
    
    # Ignorar comandos no reconocidos
    if texto.startswith("/"):
        bot.reply_to(message, "❓ Comando no reconocido. Usá /ayuda para ver opciones.")
        return
    
    bot.send_chat_action(message.chat.id, "typing")
    print(f"💬 Mensaje de {user_id}: {texto[:50]}...")

    # 🧠 Análisis de sentimiento
    sentimiento = analizar_sentimiento(texto)

    # 🔄 Comparar con memoria previa
    memoria_anterior = get_memory(user_id)
    if memoria_anterior:
        anterior = memoria_anterior["sentimiento"]
        if anterior == "NEG" and sentimiento == "POS":
            bot.send_message(message.chat.id, "🌞 Me alegra verte mejor que antes.")
            time.sleep(0.3)
        elif anterior == "POS" and sentimiento == "NEG":
            bot.send_message(message.chat.id, "💛 Te noto más apagado, pero tranquilo, eso también pasa.")
            time.sleep(0.3)

    # 💬 Generar respuesta personalizada
    respuesta = generar_recomendacion(texto, sentimiento, user_id, dataset)
    
    # Manejar respuestas múltiples
    if isinstance(respuesta, list):
        import random
        respuesta = random.choice(respuesta)
    
    bot.reply_to(message, respuesta)

    # 💾 Guardar estado
    update_memory(user_id, sentimiento, respuesta)
    add_log(user_id, texto, sentimiento, respuesta)


# ==============================
# 🎤 MENSAJES DE VOZ
# ==============================
@bot.message_handler(content_types=["voice"])
def handle_voice_message(message: tlb.types.Message):
    """Procesa mensajes de voz con transcripción automática."""
    user_id = message.from_user.id
    
    bot.send_chat_action(message.chat.id, "typing")
    bot.reply_to(message, "🎤 Escuchando tu audio...")

    try:
        # Descargar audio
        file_info = bot.get_file(message.voice.file_id)
        audio_bytes = bot.download_file(file_info.file_path)
        
        # Transcribir con función optimizada
        texto = speech_to_text(audio_bytes)
        
        if not texto:
            bot.send_message(message.chat.id, "❌ No pude entender el audio. Probá de nuevo.")
            return

        # Mostrar transcripción
        bot.send_message(message.chat.id, f"📝 Escuché: _{texto}_", parse_mode="Markdown")
        time.sleep(0.5)

        # 🧠 Análisis de sentimiento
        sentimiento = analizar_sentimiento(texto)

        # 🔄 Comparar con memoria
        memoria_anterior = get_memory(user_id)
        if memoria_anterior:
            anterior = memoria_anterior["sentimiento"]
            if anterior == "NEG" and sentimiento == "POS":
                bot.send_message(message.chat.id, "🌞 Me alegra verte mejor que antes.")
                time.sleep(0.3)
            elif anterior == "POS" and sentimiento == "NEG":
                bot.send_message(message.chat.id, "💛 Te noto más apagado, pero tranquilo, eso también pasa.")
                time.sleep(0.3)

        # Generar recomendación
        respuesta = generar_recomendacion(texto, sentimiento, user_id, dataset)
        
        if isinstance(respuesta, list):
            import random
            respuesta = random.choice(respuesta)
        
        bot.send_message(message.chat.id, f"💬 {respuesta}")

        # Guardar estado
        update_memory(user_id, sentimiento, respuesta)
        add_log(user_id, f"[VOZ] {texto}", sentimiento, respuesta)
        
    except Exception as e:
        print(f"❌ Error procesando voz: {e}")
        bot.send_message(message.chat.id, "⚠️ Hubo un error con el audio. Intentá de nuevo.")


# ==============================
# 📸 ANÁLISIS DE IMÁGENES
# ==============================
@bot.message_handler(content_types=["photo"])
def handle_photo_message(message: tlb.types.Message):
    """Analiza imágenes de comida con IA Vision."""
    user_id = message.from_user.id
    
    bot.send_chat_action(message.chat.id, "typing")
    bot.reply_to(message, "📸 Analizando tu comida con IA Vision...")

    try:
        # Descargar imagen (mejor calidad disponible)
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        # Guardar temporalmente
        temp_image_path = f"data/temp/{user_id}_comida_{int(time.time())}.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(downloaded_file)
        
        print(f"📸 Imagen recibida de {user_id}, analizando...")
        
        # Analizar con IA
        analisis = analizar_imagen_comida(temp_image_path)
        
        # Generar feedback amigable
        feedback = generar_feedback_visual(analisis)
        
        # Enviar respuesta
        bot.send_message(message.chat.id, feedback, parse_mode="Markdown")
        
        # Registrar en logs
        sentimiento_img = "NEU"
        if "evaluacion" in analisis:
            if analisis["evaluacion"] == "saludable":
                sentimiento_img = "POS"
            elif analisis["evaluacion"] == "poco_saludable":
                sentimiento_img = "NEG"
        
        alimentos = ", ".join(analisis.get("alimentos", ["imagen de comida"]))
        add_log(user_id, f"[FOTO] {alimentos}", sentimiento_img, feedback[:100])
        
        # Limpiar archivo temporal
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        print(f"✅ Imagen analizada exitosamente para {user_id}")
        
    except Exception as e:
        print(f"❌ Error al procesar imagen: {e}")
        bot.send_message(
            message.chat.id, 
            "⚠️ Hubo un problema al analizar tu imagen. Intentá con otra foto más clara."
        )


# ==============================
# 🚀 INICIO DEL BOT
# ==============================
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 MENTA - Asistente de Bienestar Alimenticio")
    print("   Samsung Innovation Campus 2025")
    print("   Equipo: Guadalupe · Fabiana · Rocco")
    print("=" * 60)
    print("\n✅ Funcionalidades de IA activas:")
    print("   1️⃣  Análisis de sentimientos (RoBERTuito)")
    print("   2️⃣  Respuestas con dataset personalizado")
    print("   3️⃣  Reconocimiento de voz (Whisper V3 Turbo)")
    print("   4️⃣  Análisis de imágenes (Llama 3.2 90B Vision)")
    print("\n🟢 Bot iniciado correctamente. Esperando mensajes...\n")
    
    try:
        bot.infinity_polling(timeout=30, long_polling_timeout=20)
    except KeyboardInterrupt:
        print("\n🛑 Bot detenido manualmente por el usuario.")
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        print("🔄 Reiniciando en 5 segundos...")
        time.sleep(5)
"""
menta_bot.py - BOT COMPLETO TODO-EN-UNO
========================================
Bot de Telegram con TODAS las funcionalidades de IA integradas en un solo archivo.

FUNCIONALIDADES INCLUIDAS:
1. ✅ Análisis de sentimientos (NLP con Transformers)
2. ✅ Comprensión de emociones basadas en texto
3. ✅ Respuestas automáticas con dataset propio
4. ✅ Audio → Texto (Speech-to-Text con Groq Whisper)
5. ✅ Análisis de imágenes (Vision AI con Groq)
6. ✅ Sistema de memoria contextual
7. ✅ Estadísticas y progreso del usuario

Proyecto Final - Samsung Innovation Campus 2025
Equipo: Guadalupe · Fabiola · Rocco

REQUISITOS:
- Python 3.10+
- pip install python-dotenv pyTelegramBotAPI groq transformers torch

CONFIGURACIÓN:
Crear archivo .env con:
    TELEGRAM_TOKEN=tu_token_aqui
    GROQ_API_KEY=tu_api_key_aqui
"""

import os
import json
import time
import base64
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional
import telebot as tlb
from dotenv import load_dotenv
from groq import Groq
from transformers import pipeline

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not TELEGRAM_TOKEN or not GROQ_API_KEY:
    raise ValueError("❌ Faltan credenciales. Configurá el archivo .env")

# Inicializar servicios
bot = tlb.TeleBot(TELEGRAM_TOKEN)
groq_client = Groq(api_key=GROQ_API_KEY)

# Crear directorios necesarios
os.makedirs("data", exist_ok=True)
os.makedirs("data/temp", exist_ok=True)

# Archivos de persistencia
MEMORY_FILE = "data/user_memory.json"
LOGS_FILE = "data/user_logs.json"
DATASET_FILE = "data/dataset.json"


# ============================================================================
# 1. ANÁLISIS DE SENTIMIENTOS (NLP)
# ============================================================================

print("🧠 Cargando modelo de análisis de sentimientos...")
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="pysentimiento/robertuito-sentiment-analysis"
    )
    print("✅ Modelo cargado correctamente")
except Exception as e:
    print(f"⚠️ Error cargando modelo: {e}")
    sentiment_analyzer = None


def analizar_sentimiento(texto: str) -> str:
    """
    Analiza el sentimiento de un texto en español.
    
    Args:
        texto: Texto a analizar
    
    Returns:
        "POS", "NEG" o "NEU"
    """
    if not texto or not sentiment_analyzer:
        return "NEU"
    
    try:
        resultado = sentiment_analyzer(texto[:512])[0]
        label = resultado["label"].upper()
        score = resultado["score"]
        
        print(f"💚 Sentimiento: {label} (confianza: {score:.2f})")
        
        if "POS" in label:
            return "POS"
        elif "NEG" in label:
            return "NEG"
        else:
            return "NEU"
    except Exception as e:
        print(f"⚠️ Error en análisis: {e}")
        return "NEU"


# ============================================================================
# 2. DATASET DE RECOMENDACIONES (Respuestas automáticas)
# ============================================================================

DATASET = {
    "recomendaciones": {
        "ansiedad": [
            "Tomate unos minutos para respirar y tomar agua. Evitá comer por impulso 🍵",
            "La ansiedad no se calma comiendo, sino entendiendo lo que sentís 💛",
            "Probá distraerte con algo que te guste antes de abrir la heladera 🎧"
        ],
        "estrés": [
            "Dale un descanso a tu mente. Una caminata corta puede ayudarte 🌿",
            "El estrés muchas veces se siente en el cuerpo. Hacé una pausa consciente 🧘‍♀️",
            "Respirá profundo y pensá: 'esto también va a pasar' 💨"
        ],
        "frustración": [
            "Cada pequeño cambio cuenta. No busques perfección, buscá constancia 💪",
            "No todo tiene que salir bien para que estés avanzando 🚶‍♂️",
            "Comer no es fallar. Aprender también es parte del proceso 🌱"
        ],
        "motivación": [
            "¡Excelente! Aprovechá esa energía para preparar una comida nutritiva 🥗",
            "Seguí así, estás construyendo hábitos que te van a hacer sentir bien 🌞",
            "Motivarte hoy es cuidar de vos mañana 💫"
        ],
        "culpa": [
            "No te castigues por lo que comiste. Enfocate en cómo querés sentirte mañana 🌻",
            "Tu valor no se mide por una comida. Se mide por cómo te tratás 💛",
            "Perdonarte también es parte del bienestar 🕊️"
        ],
        "tristeza": [
            "Está bien sentirse triste. No tenés que estar siempre perfecto 🌧️",
            "Cuidarte bien cuando estás triste es un acto de amor propio 💙",
            "Permitite sentir sin juzgarte. Esto también va a pasar 🌱"
        ],
        "hidratarse": [
            "Tomar agua es esencial para el bienestar físico y mental 💧",
            "Llevá siempre tu botella. A veces el cuerpo pide agua, no comida 🫗"
        ],
        "descanso": [
            "Dormir bien regula el apetito y mejora tu estado de ánimo 😴",
            "El descanso también es parte de una vida saludable 🌙"
        ]
    },
    "respuestas_generales": [
        "Recordá que cada paso cuenta. Cuidarte también es escucharte 💛",
        "Estoy acá para acompañarte en este camino hacia un bienestar integral 🌱",
        "Tu relación con la comida puede mejorar. Confía en el proceso 🌻"
    ]
}


def generar_recomendacion(texto: str, sentimiento: str) -> str:
    """
    Genera recomendación basada en el texto y sentimiento detectado.
    
    Args:
        texto: Mensaje del usuario
        sentimiento: Sentimiento detectado (POS/NEG/NEU)
    
    Returns:
        Recomendación personalizada
    """
    import random
    
    texto_lower = texto.lower()
    recomendaciones = DATASET["recomendaciones"]
    
    # Buscar palabras clave
    for clave, respuestas in recomendaciones.items():
        if clave in texto_lower:
            return random.choice(respuestas)
    
    # Recomendación según sentimiento
    if sentimiento == "NEG":
        posibles = ["ansiedad", "estrés", "frustración", "culpa", "tristeza"]
    elif sentimiento == "POS":
        posibles = ["motivación"]
    else:
        posibles = ["descanso", "hidratarse"]
    
    for clave in posibles:
        if clave in recomendaciones:
            respuestas = recomendaciones[clave]
            return random.choice(respuestas) if isinstance(respuestas, list) else respuestas
    
    return random.choice(DATASET["respuestas_generales"])


# ============================================================================
# 3. AUDIO → TEXTO (Speech-to-Text)
# ============================================================================

def speech_to_text(audio_bytes: bytes) -> Optional[str]:
    """
    Convierte audio a texto usando Groq Whisper.
    
    Args:
        audio_bytes: Datos binarios del audio
    
    Returns:
        Transcripción del audio o None si falla
    """
    try:
        # Guardar temporalmente
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
        
        # Transcribir
        with open(temp_audio_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=audio_file,
                language="es",
                prompt="Usuario hablando sobre alimentación o emociones",
                response_format="json"
            )
        
        # Limpiar
        os.remove(temp_audio_path)
        
        texto = transcription.text
        print(f"🎤 Audio transcrito: '{texto[:50]}...'")
        return texto
        
    except Exception as e:
        print(f"❌ Error en transcripción: {e}")
        return None


# ============================================================================
# 4. ANÁLISIS DE IMÁGENES (Computer Vision)
# ============================================================================

def analizar_imagen_comida(image_path: str) -> Dict[str, Any]:
    """
    Analiza imagen de comida con IA Vision.
    
    Args:
        image_path: Ruta a la imagen
    
    Returns:
        Dict con análisis nutricional
    """
    try:
        if not os.path.exists(image_path):
            return {"error": "Imagen no encontrada"}
        
        # Codificar imagen
        with open(image_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        print(f"📸 Analizando imagen...")
        
        # Prompt para análisis
        prompt = """Sos un nutricionista argentino. Analizá esta comida y respondé en JSON:

{
  "alimentos": ["alimento1", "alimento2"],
  "evaluacion": "saludable",
  "calorias_estimadas": "400-500 kcal",
  "aspectos_positivos": ["aspecto1"],
  "aspectos_mejorar": ["aspecto1"],
  "recomendacion": "Consejo breve y amigable"
}

evaluacion puede ser: "saludable", "moderada", o "poco_saludable"
Usá lenguaje argentino: vos, te, podés"""
        
        # Llamar a Groq Vision
        response = groq_client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    }
                ]
            }],
            temperature=0.7,
            max_tokens=1024
        )
        
        resultado = response.choices[0].message.content
        
        # Parsear JSON
        try:
            start = resultado.find('{')
            end = resultado.rfind('}') + 1
            if start != -1 and end > start:
                json_str = resultado[start:end]
                return json.loads(json_str)
        except:
            pass
        
        return {
            "alimentos": ["Comida detectada"],
            "evaluacion": "detectada",
            "recomendacion": resultado[:150]
        }
        
    except Exception as e:
        print(f"❌ Error en análisis de imagen: {e}")
        return {
            "error": str(e),
            "alimentos": [],
            "evaluacion": "error",
            "recomendacion": "⚠️ Hubo un problema al analizar la imagen."
        }


def formatear_analisis_imagen(analisis: Dict) -> str:
    """Formatea el análisis de imagen para Telegram."""
    if "error" in analisis and not analisis.get("alimentos"):
        return analisis.get("recomendacion", "❌ Error al analizar")
    
    msg = "🍽️ *Análisis de tu comida:*\n\n"
    
    if analisis.get("alimentos"):
        msg += f"📋 *Identificado:* {', '.join(analisis['alimentos'])}\n\n"
    
    if analisis.get("evaluacion"):
        emoji = {"saludable": "✅", "moderada": "⚖️", "poco_saludable": "⚠️"}.get(
            analisis["evaluacion"].lower().replace(" ", "_"), "🔍"
        )
        msg += f"{emoji} *Evaluación:* {analisis['evaluacion'].capitalize()}\n\n"
    
    if analisis.get("calorias_estimadas"):
        msg += f"🔥 *Calorías:* {analisis['calorias_estimadas']}\n\n"
    
    if analisis.get("aspectos_positivos"):
        msg += "💚 *Lo bueno:*\n"
        for asp in analisis["aspectos_positivos"]:
            msg += f"  • {asp}\n"
        msg += "\n"
    
    if analisis.get("aspectos_mejorar"):
        msg += "🌱 *Podés mejorar:*\n"
        for asp in analisis["aspectos_mejorar"]:
            msg += f"  • {asp}\n"
        msg += "\n"
    
    if analisis.get("recomendacion"):
        msg += f"💡 *Consejo:* {analisis['recomendacion']}"
    
    return msg


# ============================================================================
# 5. SISTEMA DE MEMORIA CONTEXTUAL
# ============================================================================

def cargar_memoria() -> Dict:
    """Carga memoria desde archivo."""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}


def guardar_memoria(memoria: Dict):
    """Guarda memoria en archivo."""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memoria, f, ensure_ascii=False, indent=2)


def actualizar_memoria(user_id: int, sentimiento: str, recomendacion: str):
    """Actualiza la memoria del usuario."""
    memoria = cargar_memoria()
    user_key = str(user_id)
    now = datetime.now().isoformat()
    
    if user_key not in memoria:
        memoria[user_key] = {
            "primera_interaccion": now,
            "total_interacciones": 0,
            "estadisticas": {"positivos": 0, "negativos": 0, "neutros": 0}
        }
    
    user_data = memoria[user_key]
    user_data["ultima_interaccion"] = now
    user_data["total_interacciones"] += 1
    user_data["sentimiento_actual"] = sentimiento
    user_data["ultima_recomendacion"] = recomendacion
    
    # Actualizar estadísticas
    if sentimiento == "POS":
        user_data["estadisticas"]["positivos"] += 1
    elif sentimiento == "NEG":
        user_data["estadisticas"]["negativos"] += 1
    else:
        user_data["estadisticas"]["neutros"] += 1
    
    guardar_memoria(memoria)
    print(f"💾 Memoria actualizada: {user_id} → {sentimiento}")


def obtener_memoria(user_id: int) -> Optional[Dict]:
    """Obtiene la memoria de un usuario."""
    memoria = cargar_memoria()
    return memoria.get(str(user_id))


def obtener_estadisticas(user_id: int) -> Optional[Dict]:
    """Obtiene estadísticas del usuario."""
    user_data = obtener_memoria(user_id)
    if not user_data:
        return None
    
    total = user_data["total_interacciones"]
    stats = user_data["estadisticas"]
    
    return {
        "total": total,
        "positivos": stats["positivos"],
        "negativos": stats["negativos"],
        "neutros": stats["neutros"],
        "porcentaje_positivo": (stats["positivos"] / total * 100) if total > 0 else 0
    }


# ============================================================================
# 6. REGISTRO DE INTERACCIONES (Logs)
# ============================================================================

def agregar_log(user_id: int, mensaje: str, sentimiento: str, respuesta: str):
    """Registra una interacción en el log."""
    try:
        # Cargar logs existentes
        if os.path.exists(LOGS_FILE):
            with open(LOGS_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Agregar nueva entrada
        logs.append({
            "user_id": str(user_id),
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mensaje": mensaje[:100],  # Limitar longitud
            "sentimiento": sentimiento,
            "respuesta": respuesta[:100] if isinstance(respuesta, str) else str(respuesta)[:100]
        })
        
        # Guardar (mantener últimos 1000 registros)
        with open(LOGS_FILE, "w", encoding="utf-8") as f:
            json.dump(logs[-1000:], f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"⚠️ Error guardando log: {e}")


# ============================================================================
# 7. MANEJADORES DEL BOT DE TELEGRAM
# ============================================================================

@bot.message_handler(commands=["start", "reset"])
def cmd_start(message: tlb.types.Message):
    """Comando /start - Bienvenida."""
    user_id = message.from_user.id
    username = message.from_user.username or "Usuario"
    
    bienvenida = f"""🌱 *¡Hola {username}! Soy Menta, tu asistente de bienestar alimenticio.* 🧠🍎

Podés interactuar conmigo de 3 formas:

💬 *Texto:* Contame cómo te sentís
🎤 *Audio:* Mandame un mensaje de voz
📸 *Foto:* Enviame una imagen de tu comida

_Soy un bot que sirve para analizar tus emociones y darte consejos personalizados._ ✨"""
    
    print(f"👤 Usuario conectado: {user_id} (@{username})")
    bot.reply_to(message, bienvenida, parse_mode="Markdown")


@bot.message_handler(commands=["ayuda", "help"])
def cmd_ayuda(message: tlb.types.Message):
    """Comando /ayuda - Mostrar ayuda."""
    ayuda = """📚 *Comandos disponibles:*

/start - Iniciar conversación
/progreso - Ver tu evolución emocional
/ayuda - Mostrar esta ayuda

🤖 *Funcionalidades de IA:*
• Análisis de sentimientos (NLP)
• Reconocimiento de voz (Whisper)
• Análisis de imágenes (Vision AI)
• Recomendaciones personalizadas

💡 *Tip:* Hablame con naturalidad, entiendo español argentino perfectamente."""
    
    bot.reply_to(message, ayuda, parse_mode="Markdown")


@bot.message_handler(commands=["progreso"])
def cmd_progreso(message: tlb.types.Message):
    """Comando /progreso - Mostrar estadísticas."""
    user_id = message.from_user.id
    stats = obtener_estadisticas(user_id)
    
    if not stats:
        bot.reply_to(message, "📊 Aún no tenés registros. Empezá a contarme cómo te sentís.")
        return
    
    resumen = f"""📊 *Tu progreso emocional:*

📈 Total de interacciones: *{stats['total']}*

✅ Positivos: {stats['positivos']} ({stats['porcentaje_positivo']:.1f}%)
⚠️ Negativos: {stats['negativos']}
➖ Neutros: {stats['neutros']}

"""
    
    if stats['porcentaje_positivo'] >= 70:
        resumen += "🌟 *¡Excelente!* Tu estado emocional es muy positivo."
    elif stats['porcentaje_positivo'] >= 50:
        resumen += "💪 *¡Muy bien!* Vas por buen camino."
    else:
        resumen += "🌱 Estoy acá para ayudarte. Juntos vamos a mejorar."
    
    bot.reply_to(message, resumen, parse_mode="Markdown")


@bot.message_handler(content_types=["text"])
def handle_text(message: tlb.types.Message):
    """Maneja mensajes de texto."""
    texto = message.text
    user_id = message.from_user.id
    
    if texto.startswith("/"):
        bot.reply_to(message, "❓ Comando no reconocido. Usá /ayuda")
        return
    
    bot.send_chat_action(message.chat.id, "typing")
    print(f"💬 Mensaje de {user_id}: {texto[:50]}...")
    
    # Analizar sentimiento
    sentimiento = analizar_sentimiento(texto)
    
    # Detectar cambios emocionales
    memoria_anterior = obtener_memoria(user_id)
    if memoria_anterior:
        anterior = memoria_anterior.get("sentimiento_actual")
        if anterior == "NEG" and sentimiento == "POS":
            bot.send_message(message.chat.id, "🌞 Me alegra verte mejor que antes.")
            time.sleep(0.3)
        elif anterior == "POS" and sentimiento == "NEG":
            bot.send_message(message.chat.id, "💛 Te noto más apagado, pero tranquilo, eso también pasa.")
            time.sleep(0.3)
    
    # Generar respuesta
    respuesta = generar_recomendacion(texto, sentimiento)
    bot.reply_to(message, respuesta)
    
    # Guardar en memoria y logs
    actualizar_memoria(user_id, sentimiento, respuesta)
    agregar_log(user_id, texto, sentimiento, respuesta)


@bot.message_handler(content_types=["voice"])
def handle_voice(message: tlb.types.Message):
    """Maneja mensajes de voz."""
    user_id = message.from_user.id
    
    bot.send_chat_action(message.chat.id, "typing")
    bot.reply_to(message, "🎤 Escuchando tu audio...")
    
    try:
        # Descargar audio
        file_info = bot.get_file(message.voice.file_id)
        audio_bytes = bot.download_file(file_info.file_path)
        
        # Transcribir
        texto = speech_to_text(audio_bytes)
        
        if not texto:
            bot.send_message(message.chat.id, "❌ No pude entender el audio. Probá de nuevo.")
            return
        
        # Mostrar transcripción
        bot.send_message(message.chat.id, f"📝 Escuché: _{texto}_", parse_mode="Markdown")
        time.sleep(0.5)
        
        # Analizar sentimiento
        sentimiento = analizar_sentimiento(texto)
        
        # Generar respuesta
        respuesta = generar_recomendacion(texto, sentimiento)
        bot.send_message(message.chat.id, respuesta)
        
        # Guardar
        actualizar_memoria(user_id, sentimiento, respuesta)
        agregar_log(user_id, f"[VOZ] {texto}", sentimiento, respuesta)
        
    except Exception as e:
        print(f"❌ Error procesando voz: {e}")
        bot.send_message(message.chat.id, "⚠️ Hubo un error con el audio.")


@bot.message_handler(content_types=["photo"])
def handle_photo(message: tlb.types.Message):
    """Maneja imágenes de comida."""
    user_id = message.from_user.id
    
    bot.send_chat_action(message.chat.id, "typing")
    bot.reply_to(message, "📸 Analizando tu comida con IA Vision...")
    
    try:
        # Descargar imagen
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        # Guardar temporalmente
        temp_path = f"data/temp/{user_id}_{int(time.time())}.jpg"
        with open(temp_path, "wb") as f:
            f.write(downloaded_file)
        
        print(f"📸 Imagen de {user_id}, analizando...")
        
        # Analizar
        analisis = analizar_imagen_comida(temp_path)
        feedback = formatear_analisis_imagen(analisis)
        
        # Enviar respuesta
        bot.send_message(message.chat.id, feedback, parse_mode="Markdown")
        
        # Determinar sentimiento según evaluación
        sentimiento = "NEU"
        if analisis.get("evaluacion") == "saludable":
            sentimiento = "POS"
        elif analisis.get("evaluacion") in ["poco_saludable", "poco saludable"]:
            sentimiento = "NEG"
        
        # Guardar
        alimentos = ", ".join(analisis.get("alimentos", ["comida"]))
        agregar_log(user_id, f"[FOTO] {alimentos}", sentimiento, feedback[:100])
        
        # Limpiar
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        print(f"✅ Imagen analizada para {user_id}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        bot.send_message(message.chat.id, "⚠️ Hubo un problema al analizar la imagen.")


# ============================================================================
# 8. INICIO DEL BOT
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🤖 MENTA - Asistente de Bienestar Alimenticio con IA")
    print("   Samsung Innovation Campus 2025")
    print("   Equipo: Guadalupe · Fabiana · Rocco")
    print("=" * 70)
    print("\n✅ Funcionalidades activas:")
    print("   1️⃣  Análisis de sentimientos (RoBERTuito)")
    print("   2️⃣  Comprensión de emociones")
    print("   3️⃣  Respuestas con dataset personalizado")
    print("   4️⃣  Audio → Texto (Whisper V3 Turbo)")
    print("   5️⃣  Análisis de imágenes (Llama Vision 90B)")
    print("   6️⃣  Memoria contextual adaptativa")
    print("\n🟢 Bot iniciado correctamente. Esperando mensajes...\n")
    
    try:
        bot.infinity_polling(timeout=30, long_polling_timeout=20)
    except KeyboardInterrupt:
        print("\n🛑 Bot detenido manualmente.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("🔄 Reiniciando en 5 segundos...")
        time.sleep(5)
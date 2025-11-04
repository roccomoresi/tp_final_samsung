#!/usr/bin/env python3
"""
Bot_mejorado_dashboard.py
Versión extendida de MentaBot con dashboard SQLite + HTML
Incluye: texto, voz, imagen, análisis de sentimiento, memoria, logs y dashboard generador.

Requisitos:
- Python 3.10+
- pip install python-dotenv pyTelegramBotAPI groq transformers torch matplotlib pandas

Uso:
- Configurar .env con TELEGRAM_TOKEN y GROQ_API_KEY
- Ejecutar: python Bot_mejorado_dashboard.py

"""

import os
import json
import time
import base64
import tempfile
import sqlite3
from datetime import datetime
from typing import Dict, Any, Optional
import telebot as tlb
from dotenv import load_dotenv
from groq import Groq
from transformers import pipeline

# Visualización
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not TELEGRAM_TOKEN:
    raise ValueError("❌ Faltan credenciales TELEGRAM_TOKEN en .env")

# Inicializar servicios
bot = tlb.TeleBot(TELEGRAM_TOKEN)
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None

# Crear directorios necesarios
os.makedirs("data", exist_ok=True)
os.makedirs("data/temp", exist_ok=True)
os.makedirs("data/dashboard", exist_ok=True)

# Archivos de persistencia
MEMORY_FILE = "data/user_memory.json"
LOGS_FILE = "data/user_logs.json"
DATASET_FILE = "data/dataset.json"
DB_FILE = "data/menta.db"

# ============================================================================
# 0. BASE DE DATOS SQLITE - INTERACCIONES
# ============================================================================

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        timestamp TEXT,
        type TEXT,
        text TEXT,
        sentimiento TEXT,
        alimentos TEXT,
        evaluacion TEXT,
        recomendacion TEXT
    )
    """)
    conn.commit()
    conn.close()


def save_interaction(user_id: int, tipo: str, texto: str, sentimiento: str, alimentos: Optional[str], evaluacion: Optional[str], recomendacion: Optional[str]):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "INSERT INTO interactions (user_id, timestamp, type, text, sentimiento, alimentos, evaluacion, recomendacion) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (str(user_id), datetime.now().isoformat(), tipo, texto[:1000] if texto else None, sentimiento, alimentos, evaluacion, recomendacion)
    )
    conn.commit()
    conn.close()


def fetch_user_interactions(user_id: int) -> pd.DataFrame:
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM interactions WHERE user_id = ? ORDER BY timestamp", conn, params=(str(user_id),))
    conn.close()
    return df

# ============================================================================
# 1. ANÁLISIS DE SENTIMIENTOS (NLP)
# ============================================================================

print("🧠 Cargando modelo de análisis de sentimientos...")
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="pysentimiento/robertuito-sentiment-analysis"
    )
    print("✅ Modelo de sentimiento cargado correctamente")
except Exception as e:
    print(f"⚠️ Error cargando modelo: {e}")
    sentiment_analyzer = None


def analizar_sentimiento(texto: str) -> str:
    if not texto or not sentiment_analyzer:
        return "NEU"
    try:
        resultado = sentiment_analyzer(texto[:512])[0]
        label = resultado.get("label", "NEU").upper()
        if "POS" in label:
            return "POS"
        elif "NEG" in label:
            return "NEG"
        else:
            return "NEU"
    except Exception as e:
        print(f"⚠️ Error en análisis de sentimiento: {e}")
        return "NEU"

# ============================================================================
# 2. DATASET DE RECOMENDACIONES
# ============================================================================

DATASET = {
    "recomendaciones": {
        "ansiedad": [
            "Tomate unos minutos para respirar y tomar agua. Evitá comer por impulso 🍵",
            "La ansiedad no se calma comiendo, sino entendiendo lo que sentís 💛",
            "Probá distraerte con algo que te guste antes de abrir la heladera 🎧",
            "Trata de salir a trotar por 45', cuando vuelvas vas a sentir que comés con más conciencia 🏃‍♂️",
            "Este proceso enfocalo un día a la vez. No te exijas perfección 🌟"
        ],
        "estrés": [
            "Dale un descanso a tu mente. Una caminata corta puede ayudarte 🌿",
            "El estrés muchas veces se siente en el cuerpo. Hacé una pausa consciente 🧘‍♀️",
            "Respirá profundo y pensá: 'esto también va a pasar' 💨",
            "Dormí las horas correspondientes para que tu cuerpo y mente se recuperen bien 😴",
            "¿Sabés cuál es el mejor aliado para el estrés? El deporte regular 🏋️‍♂️"
        ],
        "frustración": [
            "Cada pequeño cambio cuenta. No busques perfección, buscá constancia 💪",
            "No todo tiene que salir bien para que estés avanzando 🚶‍♂️",
            "Comer no es fallar. Aprender también es parte del proceso 🌱",
            "El camino está lleno de obstáculos, pero cada paso te acerca a tu meta 🛤️",
            "Recordá por qué empezaste este camino. Eso te va a dar fuerzas para seguir adelante 🌟"
        ],
        "motivación": [
            "¡Excelente! Aprovechá esa energía para preparar una comida nutritiva 🥗",
            "Seguí así, estás construyendo hábitos que te van a hacer sentir bien 🌞",
            "Motivarte hoy es cuidar de vos mañana 💫",
            "Me encanta verte tan comprometido con tu bienestar. ¡A seguir así! 🚀",
            "Está buenísimo que estés motivado, pero no te rijas por ella para emprender tu camino a la salud. La constancia es la clave 🔑"
        ],
        "culpa": [
            "No te castigues por lo que comiste. Enfocate en cómo querés sentirte mañana 🌻",
            "Tu valor no se mide por una comida. Se mide por cómo te tratás 💛",
            "Perdonarte también es parte del bienestar 🕊️",
            "Centrate en la versión que querés ser, no en los errores del pasado 🌟"
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
        "Tu relación con la comida puede mejorar. Confía en el proceso 🌻",
        "Combiná proteínas, fibras y carbohidratos complejos para mantener tu energía 🍎",
        "Agregar color a tu plato es sumar nutrientes 🌈"
    ]
}


def generar_recomendacion(texto: str, sentimiento: str) -> str:
    import random
    texto_lower = (texto or "").lower()
    recomendaciones = DATASET["recomendaciones"]
    for clave, respuestas in recomendaciones.items():
        if clave in texto_lower:
            return random.choice(respuestas)
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
# 3. AUDIO -> TEXTO (Speech-to-Text)
# ============================================================================

def speech_to_text(audio_bytes: bytes) -> Optional[str]:
    if not groq_client:
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
        with open(temp_audio_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=audio_file,
                language="es",
                prompt="Usuario hablando sobre alimentaci\u00f3n o emociones",
                response_format="json"
            )
        os.remove(temp_audio_path)
        texto = transcription.text
        return texto
    except Exception as e:
        print(f"❌ Error en transcripción: {e}")
        return None

# ============================================================================
# 4. ANÁLISIS DE IMÁGENES
# ============================================================================


import os



import base64



import telebot



from groq import Groq



from dotenv import load_dotenv



from PIL import Image



import io



import requests





load_dotenv()





TOKEN_BOT_TELEGRAM = os.getenv('TOKEN_BOT_TELEGRAM')



CLAVE_API_GROQ = os.getenv('CLAVE_API_GROQ')



if not TOKEN_BOT_TELEGRAM:
    
    raise ValueError("TELEGRAM_BOT_TOKEN no está configurado en las variables de entorno")



if not CLAVE_API_GROQ:
   
    raise ValueError("GROQ_API_KEY no está configurado en las variables de entorno")



bot = telebot.TeleBot(TOKEN_BOT_TELEGRAM)



cliente_groq = Groq(api_key=CLAVE_API_GROQ)



def imagen_a_base64(ruta_o_bytes_imagen):
    
    """Convierte una imagen a base64 para enviarla a Groq"""
   
    
    try:
       
        if isinstance(ruta_o_bytes_imagen, bytes):
            
            return base64.b64encode(ruta_o_bytes_imagen).decode('utf-8')
       
        
        else:
            
            with open(ruta_o_bytes_imagen, "rb") as archivo_imagen:
                
                return base64.b64encode(archivo_imagen.read()).decode('utf-8')
   
    
    except Exception as e:
        
        print(f"Error al convertir imagen a base64: {e}")
       
        
        return None





def describir_imagen_con_groq(imagen_base64):
   
    """Envía la imagen a Groq y obtiene la descripción"""
   
   
    try:
        
        completado_chat = cliente_groq.chat.completions.create(
            
            messages=[
                
                {
                    
                    "role": "user",
                   
                    
                    "content": [
                        
                        {
                            
                            "type": "text",
                           
                            
                            "text": "Por favor, describe esta imagen de manera detallada y clara en español. Incluye todos los elementos importantes que veas, colores, objetos, personas, acciones, emociones, y cualquier detalle relevante que puedas observar."
                        
                        },
                        
                        {
                            
                            "type": "image_url",
                           
                            
                            "image_url": {
                                
                                "url": f"data:image/jpeg;base64,{imagen_base64}"
                            
                            }
                        
                        }
                    
                    ]
                
                }
            
            ],
           
            
            model="meta-llama/llama-4-scout-17b-16e-instruct",
           
            
            temperature=0.7,
           
            
            max_tokens=1000
       
        )
       
        
        return completado_chat.choices[0].message.content
       
    
    except Exception as e:
        
        print(f"Error al describir imagen con Groq: {e}")
       
        
        return None



@bot.message_handler(commands=['start'])



def enviar_bienvenida(mensaje):
   
    """Mensaje de bienvenida"""
   
   
    texto_bienvenida = """
¡Hola! 👋 Soy un bot que puede describir imágenes para ti.


🖼️ **¿Cómo funciono?**
Simplemente envíame una imagen y yo te daré una descripción detallada de lo que veo.


🤖 **Tecnología:**
Utilizo Groq AI para analizar las imágenes y generar descripciones precisas.


📸 **¡Pruébame!**
Envía cualquier imagen y verás lo que puedo hacer.


Para obtener ayuda, usa el comando /help
    # Cierra el string multi-línea y termina la asignación a texto_bienvenida
    """
   
    
    bot.reply_to(mensaje, texto_bienvenida)



@bot.message_handler(commands=['help'])



def enviar_ayuda(mensaje):
   
    """Mensaje de ayuda"""
   
   
    texto_ayuda = """
🔧 **Comandos disponibles:**


/start - Iniciar el bot
/help - Mostrar esta ayuda


📸 **¿Cómo usar el bot?**


1. Envía una imagen (foto, dibujo, captura, etc.)
2. Espera unos segundos mientras proceso la imagen
3. Recibirás una descripción detallada de lo que veo


💡 **Consejos:**
- Las imágenes más claras y nítidas generan mejores descripciones
- Puedo analizar fotos, dibujos, gráficos, capturas de pantalla, etc.
- Respondo en español siempre


❓ **¿Problemas?**
Si algo no funciona, intenta enviar la imagen de nuevo.
    # Cierra el string multi-línea
    """
   
   
    bot.reply_to(mensaje, texto_ayuda)



@bot.message_handler(content_types=['photo'])



def manejar_foto(mensaje):
    
    """Procesa las imágenes enviadas por el usuario"""
   
   
    try:

        bot.reply_to(mensaje, "📸 He recibido tu imagen. Analizándola... ⏳")
       
        
        foto = mensaje.photo[-1]
       
        
        info_archivo = bot.get_file(foto.file_id)
       
        
        archivo_descargado = bot.download_file(info_archivo.file_path)
       
        
        imagen_base64 = imagen_a_base64(archivo_descargado)
       
        
        if not imagen_base64:
            
            bot.reply_to(mensaje, "❌ Error al procesar la imagen. Intenta de nuevo.")
           
          
            return
       
        
        descripcion = describir_imagen_con_groq(imagen_base64)
       
        
        if descripcion:
            
            respuesta = f"🤖 **Descripción de la imagen:**\n\n{descripcion}"
           
            
            bot.reply_to(mensaje, respuesta, parse_mode='None')
       
       
        else:
           
            bot.reply_to(mensaje, "❌ No pude analizar la imagen. Por favor, intenta con otra imagen.")
   
  
    except Exception as e:
        
        print(f"Error al procesar la imagen: {e}")
       
        
        bot.reply_to(mensaje, "❌ Ocurrió un error al procesar tu imagen. Intenta de nuevo.")



@bot.message_handler(func=lambda mensaje: True)



def manejar_otros_mensajes(mensaje):
   
    """Maneja mensajes que no son comandos ni imágenes"""
   
    
    bot.reply_to(mensaje, """
📝 Solo puedo procesar imágenes por ahora.


📸 **Envía una imagen** y te daré una descripción detallada de ella.


💡 Usa /help para ver todos los comandos disponibles.
    # Cierra el string multi-línea y la llamada a reply_to
    """)



if __name__ == '__main__':
    
    print("🤖 Bot de descripción de imágenes iniciado...")
   
    
    print("📸 Esperando imágenes para describir...")
   
    
    try:
        
        bot.polling(none_stop=True)
   
   
    except Exception as e:
        
        print(f"Error al iniciar el bot: {e}")



# ============================================================================
# 5. MEMORIA CONTEXTUAL (JSON)
# ============================================================================

def cargar_memoria() -> Dict:
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}


def guardar_memoria(memoria: Dict):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memoria, f, ensure_ascii=False, indent=2)


def actualizar_memoria(user_id: int, sentimiento: str, recomendacion: str):
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
    # asegurar llaves existentes
    user_data.setdefault("total_interacciones", 0)
    user_data.setdefault("estadisticas", {"positivos": 0, "negativos": 0, "neutros": 0})

    user_data["ultima_interaccion"] = now
    user_data["total_interacciones"] += 1
    user_data["sentimiento_actual"] = sentimiento
    user_data["ultima_recomendacion"] = recomendacion

    if sentimiento == "POS":
        user_data["estadisticas"]["positivos"] += 1
    elif sentimiento == "NEG":
        user_data["estadisticas"]["negativos"] += 1
    else:
        user_data["estadisticas"]["neutros"] += 1

    memoria[user_key] = user_data
    guardar_memoria(memoria)
    print(f"💾 Memoria actualizada: {user_id} → {sentimiento}")


def obtener_memoria(user_id: int) -> Optional[Dict]:
    memoria = cargar_memoria()
    return memoria.get(str(user_id))

# ============================================================================
# 6. LOGS (JSON)
# ============================================================================

def agregar_log(user_id: int, mensaje: str, sentimiento: str, respuesta: str):
    try:
        if os.path.exists(LOGS_FILE):
            with open(LOGS_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append({
            "user_id": str(user_id),
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mensaje": mensaje[:100],
            "sentimiento": sentimiento,
            "respuesta": respuesta[:100] if isinstance(respuesta, str) else str(respuesta)[:100]
        })
        with open(LOGS_FILE, "w", encoding="utf-8") as f:
            json.dump(logs[-1000:], f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Error guardando log: {e}")

# ============================================================================
# 7. DASHBOARD (GRAFICOS + HTML)
# ============================================================================

def generate_dashboard_html(user_id: int) -> str:
    df = fetch_user_interactions(user_id)
    if df.empty:
        html_path = f"data/dashboard/{user_id}_dashboard.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(f"<html><body><h2>No hay datos para el usuario {user_id}</h2></body></html>")
        return html_path

    # convertir timestamps
    df['ts'] = pd.to_datetime(df['timestamp'])
    df.sort_values('ts', inplace=True)

    # Gráfico 1: Evolución del estado emocional (POS=1, NEU=0, NEG=-1)
    def map_sent(s):
        return 1 if s == 'POS' else (-1 if s == 'NEG' else 0)
    df['sent_val'] = df['sentimiento'].map(map_sent)
    mood_img = f"data/dashboard/{user_id}_mood.png"
    plt.figure(figsize=(8,3))
    plt.plot(df['ts'], df['sent_val'], marker='o')
    plt.title('Evolución del estado emocional')
    plt.xlabel('Fecha')
    plt.ylabel('Estado (POS=1, NEU=0, NEG=-1)')
    plt.tight_layout()
    plt.savefig(mood_img)
    plt.close()

    # Gráfico 2: Frecuencia de comidas saludables vs no saludables
    df_food = df[df['type']=='photo']
    eval_counts = df_food['evaluacion'].fillna('desconocida').value_counts()
    food_img = f"data/dashboard/{user_id}_food.png"
    plt.figure(figsize=(6,4))
    eval_counts.plot(kind='bar')
    plt.title('Frecuencia por evaluación de comidas')
    plt.xlabel('Evaluación')
    plt.ylabel('Veces')
    plt.tight_layout()
    plt.savefig(food_img)
    plt.close()

    # Gráfico 3: Recomendaciones más frecuentes
    top_recs = df['recomendacion'].fillna('sin_recomendacion')
    top_recs = top_recs.value_counts().head(10)
    recs_img = f"data/dashboard/{user_id}_recs.png"
    plt.figure(figsize=(8,3))
    top_recs.plot(kind='barh')
    plt.title('Recomendaciones más frecuentes')
    plt.tight_layout()
    plt.savefig(recs_img)
    plt.close()

    # Crear HTML
    html_path = f"data/dashboard/{user_id}_dashboard.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'><title>Dashboard Menta</title></head><body>")
        f.write(f"<h2>Dashboard - Usuario {user_id}</h2>")
        f.write("<h3>Evolución del estado emocional</h3>")
        f.write(f"<img src='{os.path.basename(mood_img)}' style='max-width:800px;'><br>")
        f.write("<h3>Frecuencia por evaluación de comidas</h3>")
        f.write(f"<img src='{os.path.basename(food_img)}' style='max-width:800px;'><br>")
        f.write("<h3>Recomendaciones más frecuentes</h3>")
        f.write(f"<img src='{os.path.basename(recs_img)}' style='max-width:800px;'><br>")
        f.write("</body></html>")

    # copiar imagenes a la misma carpeta (ya guardadas ahi), asegurarnos que html y imgs estén en data/dashboard
    return html_path

# ============================================================================
# 8. MANEJADORES DEL BOT
# ============================================================================

@bot.message_handler(commands=["start", "reset"])
def cmd_start(message: tlb.types.Message):
    user_id = message.from_user.id
    username = message.from_user.username or "Usuario"
    bienvenida = f"""🌱 *¡Hola {username}! Soy Menta, tu asistente de bienestar alimenticio.* 🧠🍎\n\nPodés interactuar conmigo de 3 formas:\n\n💬 *Texto:* Contame cómo te sentís\n🎤 *Audio:* Mandame un mensaje de voz\n📸 *Foto:* Enviame una imagen de tu comida\n\n_Soy un bot que sirve para analizar tus emociones y darte consejos personalizados._ ✨"""
    print(f"👤 Usuario conectado: {user_id} (@{username})")
    bot.reply_to(message, bienvenida, parse_mode="Markdown")


@bot.message_handler(commands=["ayuda", "help"])
def cmd_ayuda(message: tlb.types.Message):
    ayuda = """📚 *Comandos disponibles:*\n\n/start - Iniciar conversación\n/progreso - Ver tu evolución emocional\n/ayuda - Mostrar esta ayuda\n/dashboard - Generar y recibir tu dashboard en HTML\n\n🤖 *Funcionalidades de IA:*\n• Análisis de sentimientos (NLP)\n• Reconocimiento de voz (Whisper)\n• Análisis de imágenes (Vision AI)\n• Recomendaciones personalizadas\n\n💡 *Tip:* Hablame con naturalidad, entiendo español argentino perfectamente."""
    bot.reply_to(message, ayuda, parse_mode="Markdown")


@bot.message_handler(commands=["progreso"])
def cmd_progreso(message: tlb.types.Message):
    user_id = message.from_user.id
    memoria = obtener_memoria(user_id)
    if not memoria:
        bot.reply_to(message, "📊 Aún no tenés registros. Empezá a contarme cómo te sentís.")
        return
    stats = memoria.get('estadisticas', {"positivos":0,"negativos":0,"neutros":0})
    total = memoria.get('total_interacciones', 0)
    porcentaje = (stats.get('positivos',0) / total * 100) if total>0 else 0
    resumen = f"""📊 *Tu progreso emocional:*\n\n📈 Total de interacciones: *{total}*\n\n✅ Positivos: {stats.get('positivos',0)} ({porcentaje:.1f}%)\n⚠️ Negativos: {stats.get('negativos',0)}\n➖ Neutros: {stats.get('neutros',0)}\n\n"""
    if porcentaje >= 70:
        resumen += "🌟 *¡Excelente!* Tu estado emocional es muy positivo."
    elif porcentaje >= 50:
        resumen += "💪 *¡Muy bien!* Vas por buen camino."
    else:
        resumen += "🌱 Estoy acá para ayudarte. Juntos vamos a mejorar."
    bot.reply_to(message, resumen, parse_mode="Markdown")


@bot.message_handler(commands=["dashboard"])
def cmd_dashboard(message: tlb.types.Message):
    user_id = message.from_user.id
    bot.send_chat_action(message.chat.id, "upload_document")
    try:
        html_path = generate_dashboard_html(user_id)
        # enviar archivo HTML y las imagenes generadas
        folder = os.path.dirname(html_path)
        files_to_send = [html_path]
        # incluir imagenes png generados para el usuario
        for fname in os.listdir(folder):
            if fname.startswith(str(user_id)) and (fname.endswith('.png') or fname.endswith('.html')):
                files_to_send.append(os.path.join(folder, fname))
        # Enviar html como documento
        with open(html_path, 'rb') as f:
            bot.send_document(message.chat.id, f, caption='Dashboard generado (abrir en navegador)')
    except Exception as e:
        print(f"❌ Error generando dashboard: {e}")
        bot.send_message(message.chat.id, "⚠️ No se pudo generar el dashboard.")


@bot.message_handler(content_types=["text"])
def handle_text(message: tlb.types.Message):
    texto = message.text
    user_id = message.from_user.id
    if texto.startswith("/"):
        bot.reply_to(message, "❓ Comando no reconocido. Usá /ayuda")
        return
    bot.send_chat_action(message.chat.id, "typing")
    sentimiento = analizar_sentimiento(texto)
    respuesta = generar_recomendacion(texto, sentimiento)
    bot.reply_to(message, respuesta)
    actualizar_memoria(user_id, sentimiento, respuesta)
    agregar_log(user_id, texto, sentimiento, respuesta)
    save_interaction(user_id, 'text', texto, sentimiento, None, None, respuesta)


@bot.message_handler(content_types=["voice"])
def handle_voice(message: tlb.types.Message):
    user_id = message.from_user.id
    bot.send_chat_action(message.chat.id, "typing")
    bot.reply_to(message, "🎤 Escuchando tu audio...")
    try:
        file_info = bot.get_file(message.voice.file_id)
        audio_bytes = bot.download_file(file_info.file_path)
        texto = speech_to_text(audio_bytes)
        if not texto:
            bot.send_message(message.chat.id, "❌ No pude entender el audio. Probá de nuevo.")
            return
        bot.send_message(message.chat.id, f"📝 Escuché: _{texto}_", parse_mode="Markdown")
        sentimiento = analizar_sentimiento(texto)
        respuesta = generar_recomendacion(texto, sentimiento)
        bot.send_message(message.chat.id, respuesta)
        actualizar_memoria(user_id, sentimiento, respuesta)
        agregar_log(user_id, f"[VOZ] {texto}", sentimiento, respuesta)
        save_interaction(user_id, 'voice', texto, sentimiento, None, None, respuesta)
    except Exception as e:
        print(f"❌ Error procesando voz: {e}")
        bot.send_message(message.chat.id, "⚠️ Hubo un error con el audio.")


@bot.message_handler(content_types=["photo"])
def handle_photo(message: tlb.types.Message):
    user_id = message.from_user.id
    bot.send_chat_action(message.chat.id, "typing")
    bot.reply_to(message, "📸 Analizando tu comida con IA Vision...")
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        temp_path = f"data/temp/{user_id}_{int(time.time())}.jpg"
        with open(temp_path, "wb") as f:
            f.write(downloaded_file)
        analisis = analizar_imagen_comida(temp_path)
        feedback = formatear_analisis_imagen(analisis)
        bot.send_message(message.chat.id, feedback, parse_mode="Markdown")
        # Analisis de sentimiento del texto de recomendacion
        sentimiento_texto = analizar_sentimiento(analisis.get("recomendacion", ""))
        # Determinar sentimiento por evaluacion visual
        sentimiento = "NEU"
        evaluacion = (analisis.get("evaluacion") or "").lower()
        if evaluacion == "saludable":
            sentimiento = "POS"
        elif "poco" in evaluacion:
            sentimiento = "NEG"
        else:
            sentimiento = sentimiento_texto or "NEU"
        recomendacion_text = analisis.get("recomendacion", "")
        alimentos = ", ".join(analisis.get("alimentos", [])) if analisis.get("alimentos") else None
        agregar_log(user_id, f"[FOTO] {alimentos}", sentimiento, feedback[:100])
        actualizar_memoria(user_id, sentimiento, recomendacion_text)
        save_interaction(user_id, 'photo', '', sentimiento, alimentos, analisis.get('evaluacion'), recomendacion_text)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"✅ Imagen analizada para {user_id}")
    except Exception as e:
        print(f"❌ Error procesando imagen: {e}")
        bot.send_message(message.chat.id, "⚠️ Hubo un problema al analizar la imagen. Probá de nuevo con otra foto.")

# ============================================================================
# 9. INICIO DEL BOT
# ============================================================================

if __name__ == "__main__":
    init_db()
    print("\n" + "="*60)
    print("🤖 MENTA - Asistente de Bienestar Alimenticio con IA (Dashboard)")
    print("   Equipo: Guadalupe · Fabiola · Rocco")
    print("="*60 + "\n")
    try:
        bot.infinity_polling(timeout=30, long_polling_timeout=20)
    except KeyboardInterrupt:
        print("\n🛑 Bot detenido manualmente.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        time.sleep(5)

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
- Ejecutar: python bot_dashboard.py

"""
import os
import json
import io
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
import random



# Visualización
import matplotlib.pyplot as plt
import pandas as pd
from html2image import Html2Image

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

# Cargar el .env desde un nivel superior (fuera de src)
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))


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
            "Tratá de salir a trotar por 45', cuando vuelvas vas a sentir que comés con más conciencia 🏃‍♂️",
            "Este proceso enfocalo un día a la vez. No te exijas perfección 🌟",
            "A veces lo que necesitás no es comida, sino contención 💬",
            "Apoyate en una rutina que te dé calma: música, aire fresco, pausas ☀️",
            "Cuando la mente corre, el cuerpo acompaña. Movete un poco para liberar energía ✨"
        ],
        "estrés": [
            "Dale un descanso a tu mente. Una caminata corta puede ayudarte 🌿",
            "El estrés muchas veces se siente en el cuerpo. Hacé una pausa consciente 🧘‍♀️",
            "Respirá profundo y pensá: 'esto también va a pasar' 💨",
            "Dormí las horas correspondientes para que tu cuerpo y mente se recuperen bien 😴",
            "¿Sabés cuál es el mejor aliado para el estrés? El deporte regular 🏋️‍♂️",
            "Tu cuerpo no necesita más carga, necesita calma 💫",
            "Tomate un té, cerrá los ojos y volvé a vos ☕",
            "Desenchufate un rato del celular. A veces el silencio es la mejor medicina 📵"
        ],
        "frustración": [
            "Cada pequeño cambio cuenta. No busques perfección, buscá constancia 💪",
            "No todo tiene que salir bien para que estés avanzando 🚶‍♂️",
            "Comer no es fallar. Aprender también es parte del proceso 🌱",
            "El camino está lleno de obstáculos, pero cada paso te acerca a tu meta 🛤️",
            "Recordá por qué empezaste este camino. Eso te va a dar fuerzas para seguir adelante 🌟",
            "No te juzgues por tropezar, valorá que seguís intentando 💚",
            "Tu valor no se mide por lo que lográs, sino por lo que te animás a intentar 🌻"
        ],
        "motivación": [
            "¡Excelente! Aprovechá esa energía para preparar una comida nutritiva 🥗",
            "Seguí así, estás construyendo hábitos que te van a hacer sentir bien 🌞",
            "Motivarte hoy es cuidar de vos mañana 💫",
            "Me encanta verte tan comprometido con tu bienestar. ¡A seguir así! 🚀",
            "Está buenísimo que estés motivado, pero no te rijas solo por eso. La constancia es la clave 🔑",
            "Transformá esa motivación en acción, incluso si el paso es chiquito 👣",
            "Tu cuerpo es tu casa: cuidalo con amor y sin exigencias 🏡"
        ],
        "culpa": [
            "No te castigues por lo que comiste. Enfocate en cómo querés sentirte mañana 🌻",
            "Tu valor no se mide por una comida. Se mide por cómo te tratás 💛",
            "Perdonarte también es parte del bienestar 🕊️",
            "Centrate en la versión que querés ser, no en los errores del pasado 🌟",
            "Recordá: un día a la vez.",
            "No hay retroceso si aprendés del paso 💬",
            "Soltar la culpa abre espacio para el autocuidado 🌷"
        ],
        "tristeza": [
            "Está bien sentirse triste. No tenés que estar siempre perfecto 🌧️",
            "Cuidarte bien cuando estás triste es un acto de amor propio 💙",
            "Permitite sentir sin juzgarte. Esto también va a pasar 🌱",
            "Hoy podés descansar un poco más. Mañana vas a estar mejor 🌙",
            "Comé liviano y con calma, tu cuerpo también necesita contención 🍲"
        ],
        "aburrimiento": [
            "El aburrimiento puede ser una oportunidad para descubrir algo nuevo que te apasione 🎨",
            "A veces el cuerpo pide movimiento cuando la mente se aburre. Probá salir a caminar o estirarte un poco 🚶‍♀️",
            "Podés aprovechar este momento para probar una receta saludable o aprender algo distinto 🍲",
            "El aburrimiento no siempre es malo: puede ser una pausa que tu mente necesita para descansar 🌿",
            "¿Qué tal si convertís ese aburrimiento en un pequeño reto personal? Prepará una comida colorida o escribí cómo te sentís 💛",
            "Cuando sientas aburrimiento, hacé algo que te conecte con vos, aunque sea preparar una infusión rica ☕",
            "No busques llenar el aburrimiento con comida. Probá música, dibujo o movimiento 🎧🧘‍♀️",
            "A veces el aburrimiento es solo una señal de que necesitás un cambio de foco, no de comida 🔄",
            "Tu cuerpo no tiene hambre, tiene ganas de estímulo. Regalate una pausa consciente o algo que te inspire 🌸",
            "Transformá el aburrimiento en curiosidad: leé algo breve, salí al sol o anotá una idea que te motive ☀️"
        ],

        "hidratarse": [
            "Tomar agua es esencial para el bienestar físico y mental 💧",
            "Llevá siempre tu botella. A veces el cuerpo pide agua, no comida 🫗",
            "Hidratate bien, te va a ayudar a pensar con más claridad 🩵",
            "Un vaso de agua cada hora mantiene tu energía más estable ⏳"
        ],
        "descanso": [
            "Dormir bien regula el apetito y mejora tu estado de ánimo 😴",
            "El descanso también es parte de una vida saludable 🌙",
            "Si estás cansado, tu cuerpo te está pidiendo una pausa, no más esfuerzo 💤",
            "Un día productivo también puede incluir una siesta reparadora ☀️"
        ],
        "autoestima": [
            "Sos mucho más que lo que comés o pesás 💛",
            "Tu valor no depende del espejo, sino de cómo te tratás cada día 🌻",
            "Hablale a tu cuerpo como le hablarías a alguien que querés 💬",
            "Reconocé tus logros, aunque parezcan pequeños 🌿"
        ],
        "rutina": [
            "Organizá tus comidas del día, eso te da estructura y calma 📅",
            "Tener horarios regulares ayuda a tu cuerpo a sentirse seguro 🕒",
            "Una buena rutina no tiene que ser perfecta, solo constante 💪"
        ],
        "bajar_peso": [
            "No se trata de comer menos, sino de comer mejor 🍎",
            "Sumá más verduras y proteínas a tus comidas, y reducí los ultraprocesados 🥦",
            "Evitar los extremos: el equilibrio siempre gana 🌿",
            "Dormir bien es clave para regular el apetito y las hormonas del hambre 😴",
            "Tomá agua antes de cada comida, te ayuda a controlar la ansiedad y saciedad 💧",
            "Movete al menos 30 minutos por día, aunque sea caminando 🚶‍♀️",
            "Comé con atención plena: sin pantallas y escuchando a tu cuerpo 🍽️",
            "No te castigues si un día comés de más. Lo importante es volver al equilibrio 💚"
        ],
        "masa_muscular": [
            "Incluí una buena fuente de proteína en cada comida 🥚",
            "Dormí al menos 7-8 horas para que tus músculos se recuperen bien 💤",
            "Entrená con constancia, no con perfección 💪",
            "Comé más de lo que gastás, pero con alimentos reales y nutritivos 🍗",
            "Después de entrenar, sumá un combo de proteína + carbohidrato para recuperar energía 🍌",
            "Evitá saltear comidas: el cuerpo necesita combustible constante ⚡",
            "Hidratate bien, el agua es esencial para el crecimiento muscular 💧",
            "La paciencia también construye músculo. Los resultados llegan con el tiempo ⏳"
        ]
    },
    "respuestas_generales": [
        "Recordá que cada paso cuenta. Cuidarte también es escucharte 💛",
        "Estoy acá para acompañarte en este camino hacia un bienestar integral 🌱",
        "Tu relación con la comida puede mejorar. Confía en el proceso 🌻",
        "Combiná proteínas, fibras y carbohidratos complejos para mantener tu energía 🍎",
        "Agregar color a tu plato es sumar nutrientes 🌈",
        "Tu cuerpo te habla todo el tiempo, escuchalo con amabilidad 💬",
        "Cada comida es una oportunidad para nutrirte, no para exigirte 🍽️",
        "Cuidarte no es un castigo, es una forma de quererte 🌷",
        "Elegir con conciencia es un acto de amor propio 💚",
        "No te apures: los buenos hábitos crecen con paciencia ☀️",
        "Podés hacerlo a tu ritmo, no necesitás compararte con nadie 🌿",
        "El bienestar no es una meta, es una forma de vivir 🌞"
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
        posibles = ["ansiedad", "estrés", "frustración", "culpa", "tristeza", "aburrimiento"]
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
# DATASETS DE SALUDOS Y DESPEDIDAS
# ============================================================================

DATASET["saludos"] = {
    "patrones": [
        "hola", "buen día", "buenas", "buenas tardes", "buenas noches",
        "hey", "holis", "holiii", "qué tal", "como estas", "cómo va", "saludos"
    ],
    "respuestas": [
        "🌿 ¡Hola! Soy *MENTA*, tu consejera de bienestar emocional orientada a una alimentación consciente. ¿Cómo te sentís hoy?",
        "💚 ¡Buen día! Soy *MENTA*, especialista en bienestar y alimentación saludable. Estoy acá para acompañarte en tu proceso con empatía y equilibrio. ¿Cómo estás?",
        "🌸 ¡Hola! Te habla *MENTA*, tu guía para conectar emociones y hábitos saludables. Contame, ¿cómo viene tu día?",
        "☀️ ¡Hola! Soy *MENTA*, tu aliada en el camino hacia una relación más consciente con la comida y con vos misma. ¿Querés que hablemos un poco?",
        "🍀 ¡Hola! Soy *MENTA*, tu asistente de bienestar y alimentación equilibrada. Estoy lista para ayudarte a sentirte mejor. ¿Cómo estás hoy?"
    ]
}

DATASET["despedidas"] = {
    "patrones": ["chau", "adiós", "nos vemos", "hasta luego", "me voy", "hasta pronto", "bye", "nos hablamos"],
    "respuestas": [
        "🌷 ¡Hasta luego! Cuidate mucho 💚",
        "💤 ¡Nos vemos! Que descanses y te hidrates bien 💧",
        "🌿 ¡Adiós! Recordá escucharte y comer con calma 🍽️",
        "💫 ¡Hasta la próxima! Me encantó acompañarte hoy 🌻"
    ]
}


# ============================================================================
# FUNCIONES DE DETECCIÓN DE SALUDOS Y DESPEDIDAS
# ============================================================================

def detectar_saludo(texto: str) -> bool:
    texto_lower = texto.lower().strip()
    for patron in DATASET["saludos"]["patrones"]:
        if patron in texto_lower:
            palabras = texto_lower.split()
            if len(palabras) <= 5 or texto_lower.startswith(patron):
                return True
    return False

def detectar_despedida(texto: str) -> bool:
    texto_lower = texto.lower().strip()
    for patron in DATASET["despedidas"]["patrones"]:
        if patron in texto_lower:
            palabras = texto_lower.split()
            if len(palabras) <= 6 or texto_lower.startswith(patron):
                return True
    return False

def generar_saludo() -> str:
    return random.choice(DATASET["saludos"]["respuestas"])

def generar_despedida() -> str:
    return random.choice(DATASET["despedidas"]["respuestas"])

# ============================================================================
#  PALABRAS CLAVE PARA DETECCIÓN MANUAL DE EMOCIONES
# ============================================================================

KEYWORDS = {
    "ansiedad": ["ansiosa", "ansioso", "nerviosa", "nervioso", "me da ansiedad", "angustia"],
    "estrés": ["estresada", "estresado", "agotada", "agotado", "tensión", "presionada", "presionado", "mucho para estudiar", "mucho trabajo"],
    "frustración": ["frustrada", "frustrado", "desanimada", "desanimado", "no puedo", "me sale mal"],
    "culpa": ["culpa", "me siento mal por comer", "no debí", "me arrepiento"],
    "tristeza": ["triste", "bajón", "sin ganas", "mal día", "deprimida", "deprimido"],
    "motivación": ["motivado", "motivada", "con ganas", "feliz", "entusiasmado", "energía"],
    "aburrimiento": ["aburrida", "aburrido", "me aburro", "nada para hacer", "estoy embolada", "estoy embolado", "no tengo ganas", "todo me aburre"]
}

def detectar_emocion_por_palabras(texto: str) -> str:
    texto = texto.lower()
    for emocion, palabras in KEYWORDS.items():
        for palabra in palabras:
            if palabra in texto:
                return emocion
    return None


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

def analizar_imagen_comida(image_path: str) -> Dict[str, Any]:
    if not groq_client:
        return {"error": "Groq API key no configurada", "alimentos": [], "evaluacion": "error", "recomendacion": "Groq no disponible"}
    try:
        with open(image_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        prompt = """Sos un nutricionista argentino. Analizá esta comida y respondé en JSON:\n\n{\n  \"alimentos\": [\"alimento1\", \"alimento2\"],\n  \"evaluacion\": \"saludable\",\n  \"calorias_estimadas\": \"400-500 kcal\",\n  \"aspectos_positivos\": [\"aspecto1\"],\n  \"aspectos_mejorar\": [\"aspecto1\"],\n  \"recomendacion\": \"Consejo breve y amigable\"\n}\n\nevaluacion puede ser: \"saludable\", \"moderada\", o \"poco_saludable\"\nUsa lenguaje argentino: vos, te, podés"""
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ]
            }],
            temperature=0.7,
            max_tokens=1024
        )
        resultado = response.choices[0].message.content
        # intentar parsear JSON dentro del texto devuelto
        try:
            start = resultado.find('{')
            end = resultado.rfind('}') + 1
            if start != -1 and end > start:
                json_str = resultado[start:end]
                return json.loads(json_str)
        except Exception:
            pass
        return {"alimentos": ["Comida detectada"], "evaluacion": "detectada", "recomendacion": resultado[:300]}
    except Exception as e:
        print(f"❌ Error en análisis de imagen: {e}")
        return {"error": str(e), "alimentos": [], "evaluacion": "error", "recomendacion": "Hubo un problema al analizar la imagen."}


def formatear_analisis_imagen(analisis: Dict) -> str:
    if not analisis:
        return "❌ Error al analizar la imagen."
    if analisis.get("error"):
        return analisis.get("recomendacion", "❌ Error al analizar")
    msg = "🍽️ *Análisis de tu comida:*\n\n"
    if analisis.get("alimentos"):
        msg += f"📋 *Identificado:* {', '.join(analisis['alimentos'])}\n\n"
    if analisis.get("evaluacion"):
        emoji = {"saludable": "✅", "moderada": "⚖️", "poco_saludable": "⚠️"}.get(analisis["evaluacion"].lower().replace(" ", "_"), "🔍")
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

def generate_dashboard_html(user_id):
    """Genera un dashboard HTML y guarda los tres gráficos individuales."""
    import matplotlib.dates as mdates

    dashboard_dir = "data/dashboard"
    os.makedirs(dashboard_dir, exist_ok=True)
    output_path = os.path.join(dashboard_dir, f"{user_id}_dashboard.html")

    db_path = "data/menta.db"
    if not os.path.exists(db_path):
        print("⚠️ No hay base de datos. Generá interacciones antes de usar /dashboard.")
        return None

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM interactions WHERE user_id = ?", conn, params=(user_id,))
    conn.close()

    if df.empty:
        html = f"<h2>Dashboard - Usuario {user_id}</h2><p>No hay datos suficientes para generar el dashboard.</p>"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return output_path

    # ---------------------------
    # Función auxiliar
    # ---------------------------
    def fig_to_base64_and_save(fig, path):
        """Convierte figura a base64 y la guarda en archivo PNG."""
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        fig.savefig(path, bbox_inches="tight")
        buffer.seek(0)
        img_b64 = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close(fig)
        return img_b64

    df["fecha"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("fecha")

    # --- Gráfico 1: Evolución emocional ---
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    df["sentimiento_num"] = df["sentimiento"].map({"NEG": -1, "NEU": 0, "POS": 1})
    ax1.plot(df["fecha"], df["sentimiento_num"], marker="o", linewidth=2, color="#2a7c4e")
    ax1.set_title("Evolución del estado emocional")
    ax1.set_xlabel("Fecha")
    ax1.set_ylabel("Nivel de emoción (-1 Negativo / +1 Positivo)")
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    mood_b64 = fig_to_base64_and_save(fig1, os.path.join(dashboard_dir, f"{user_id}_fig1_emocional.png"))

    # --- Gráfico 2: Frecuencia por evaluación de comidas ---
    food_b64 = ""
    if "evaluacion" in df.columns and not df["evaluacion"].isna().all():
        evaluaciones = df["evaluacion"].value_counts()
        fig2, ax2 = plt.subplots()
        ax2.bar(evaluaciones.index, evaluaciones.values, color=["green", "orange", "red"])
        ax2.set_title("Frecuencia por evaluación de comidas")
        ax2.set_xlabel("Tipo de comida")
        ax2.set_ylabel("Cantidad")
        plt.tight_layout()
        food_b64 = fig_to_base64_and_save(fig2, os.path.join(dashboard_dir, f"{user_id}_fig2_comidas.png"))

    # --- Gráfico 3: Recomendaciones más frecuentes ---
    recs_b64 = ""
    if "recomendacion" in df.columns and not df["recomendacion"].isna().all():
        top_recs = df["recomendacion"].value_counts().head(10)
        fig3, ax3 = plt.subplots()
        ax3.barh(top_recs.index[::-1], top_recs.values[::-1], color="skyblue")
        ax3.set_title("Recomendaciones más frecuentes")
        ax3.set_xlabel("Cantidad de veces")
        plt.tight_layout()
        recs_b64 = fig_to_base64_and_save(fig3, os.path.join(dashboard_dir, f"{user_id}_fig3_recomendaciones.png"))

    # ---------------------------
    # HTML final
    # ---------------------------
    html = f"""
    <html lang="es">
    <head>
        <meta charset="utf-8">
        <title>Dashboard de Bienestar - Usuario {user_id}</title>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
        <style>
            body {{
                font-family: 'Poppins', sans-serif;
                margin: 0;
                padding: 0;
                background: linear-gradient(180deg, #f9fdfb 0%, #e6f1eb 100%);
                color: #2a2a2a;
            }}
            header {{
                background-color: #2a7c4e;
                color: white;
                padding: 20px 40px;
                text-align: center;
                box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            }}
            header h1 {{ margin: 0; font-size: 26px; letter-spacing: 1px; }}
            .container {{
                max-width: 960px; margin: 40px auto;
                background: white; border-radius: 16px;
                padding: 30px 50px; box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            }}
            h2 {{ color: #2a7c4e; margin-top: 0; }}
            section {{ margin-bottom: 50px; }}
            .chart {{ text-align: center; }}
            img {{
                display: block; margin: 0 auto 20px auto;
                max-width: 100%; border-radius: 10px;
                box-shadow: 0 3px 8px rgba(0,0,0,0.15);
            }}
            .no-data {{ color: #888; text-align: center; font-style: italic; }}
            footer {{ text-align: center; padding: 20px; font-size: 13px; color: #666; }}
        </style>
    </head>
    <body>
        <header><h1>🌱 Dashboard de Bienestar - Usuario {user_id}</h1></header>
        <div class="container">
            <section>
                <h2>Evolución emocional</h2>
                <div class="chart">
                    {'<img src="data:image/png;base64,' + mood_b64 + '">' if mood_b64 else '<p class="no-data">Sin datos.</p>'}
                </div>
            </section>
            <section>
                <h2>Evaluación de comidas</h2>
                <div class="chart">
                    {'<img src="data:image/png;base64,' + food_b64 + '">' if food_b64 else '<p class="no-data">Sin registros de comidas.</p>'}
                </div>
            </section>
            <section>
                <h2>Recomendaciones más frecuentes</h2>
                <div class="chart">
                    {'<img src="data:image/png;base64,' + recs_b64 + '">' if recs_b64 else '<p class="no-data">Sin recomendaciones.</p>'}
                </div>
            </section>
        </div>
        <footer>MentaBot 🌿 · Bienestar alimenticio y emocional · Generado automáticamente</footer>
    </body>
    </html>
    """

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Dashboard y gráficos guardados para usuario {user_id}")
    return output_path

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
    bot.send_chat_action(message.chat.id, "upload_photo")

    try:
        # 1️⃣ Generar HTML (para mantener consistencia, aunque no se use)
        html_path = generate_dashboard_html(user_id)
        if not html_path:
            bot.reply_to(message, "⚠️ No hay datos suficientes para generar el dashboard.")
            return

        # 2️⃣ Buscar los gráficos ya guardados en el directorio (fig1, fig2, fig3)
        dashboard_dir = "data/dashboard"
        user_files = [
            f for f in os.listdir(dashboard_dir)
            if f.startswith(str(user_id)) and f.endswith(".png")
        ]

        if not user_files:
            bot.reply_to(message, "⚠️ No se encontraron gráficos generados para este usuario.")
            return

        # 3️⃣ Enviar cada gráfico como una foto separada
        for file_name in sorted(user_files):
            file_path = os.path.join(dashboard_dir, file_name)
            caption = ""
            if "emotional" in file_name.lower() or "fig1" in file_name.lower():
                caption = "📈 Evolución emocional"
            elif "comida" in file_name.lower() or "fig2" in file_name.lower():
                caption = "🍽️ Evaluación de comidas"
            elif "recomendaciones" in file_name.lower() or "fig3" in file_name.lower():
                caption = "💬 Recomendaciones más frecuentes"

            with open(file_path, "rb") as img:
                bot.send_photo(message.chat.id, img, caption=caption)

        bot.reply_to(
            message,
            "✅ Dashboard completo enviado 🌿",
            parse_mode="Markdown"
        )

    except Exception as e:
        print(f"❌ Error generando dashboard: {e}")
        bot.send_message(message.chat.id, "⚠️ No se pudo generar el dashboard.")


@bot.message_handler(content_types=["text"])
def handle_text(message):
    user_id = message.from_user.id
    user_input = message.text.lower().strip()

    # --- 1️) Detectar saludos ---
    if detectar_saludo(user_input):
        respuesta = generar_saludo()
        bot.reply_to(message, respuesta, parse_mode="Markdown")
        actualizar_memoria(user_id, "POS", respuesta)
        save_interaction(user_id, 'text', user_input, "POS", None, None, respuesta)
        return

    # --- 2️) Detectar despedidas ---
    if detectar_despedida(user_input):
        respuesta = generar_despedida()
        bot.reply_to(message, respuesta, parse_mode="Markdown")
        actualizar_memoria(user_id, "NEU", respuesta)
        save_interaction(user_id, 'text', user_input, "NEU", None, None, respuesta)
        return

    # --- 3️) Detectar emoción mediante palabras clave ---
    emocion_detectada = detectar_emocion_por_palabras(user_input)
    if emocion_detectada:
        respuestas = DATASET["recomendaciones"].get(emocion_detectada, [])
        if respuestas:
            respuesta = random.choice(respuestas)
            bot.reply_to(
                message,
                f"🧠 Detecté que estás sintiendo *{emocion_detectada}*.\n\n{respuesta}",
                parse_mode="Markdown"
            )
            sentimiento = "NEG" if emocion_detectada in ["ansiedad", "estrés", "culpa", "frustración", "tristeza", "aburrimiento"] else "POS"
            actualizar_memoria(user_id, sentimiento, respuesta)
            agregar_log(user_id, f"[TEXTO] {user_input}", sentimiento, respuesta)
            save_interaction(user_id, 'text', user_input, sentimiento, None, None, respuesta)
            return

    # --- 4️) Detección de intenciones específicas (peso y músculo) ---
    if any(palabra in user_input for palabra in ["bajar de peso", "adelgazar", "perder grasa", "rebajar", "dietas", "definir"]):
        respuesta = random.choice(DATASET["recomendaciones"]["bajar_peso"])
        bot.reply_to(message, f"🍎 *Consejo para bajar de peso:*\n\n{respuesta}", parse_mode="Markdown")
        actualizar_memoria(user_id, "POS", respuesta)
        save_interaction(user_id, 'text', user_input, "POS", None, None, respuesta)
        return

    if any(palabra in user_input for palabra in ["ganar músculo", "masa muscular", "aumentar masa", "volumen", "subir de peso saludable"]):
        respuesta = random.choice(DATASET["recomendaciones"]["masa_muscular"])
        bot.reply_to(message, f"💪 *Consejo para aumentar masa muscular:*\n\n{respuesta}", parse_mode="Markdown")
        actualizar_memoria(user_id, "POS", respuesta)
        save_interaction(user_id, 'text', user_input, "POS", None, None, respuesta)
        return

    # --- 5️) Si no hay coincidencia, usar el modelo de sentimiento ---
    sentimiento = analizar_sentimiento(user_input)
    respuesta = generar_recomendacion(user_input, sentimiento)
    bot.reply_to(message, respuesta, parse_mode="Markdown")
    actualizar_memoria(user_id, sentimiento, respuesta)
    agregar_log(user_id, f"[TEXTO] {user_input}", sentimiento, respuesta)
    save_interaction(user_id, 'text', user_input, sentimiento, None, None, respuesta)



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
        bot.reply_to(message, feedback, parse_mode="HTML")
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
    print("🤖 MENTA - Asistente de Bienestar Alimenticio, todo empieza desde la consciencia.")
    print("   Equipo: Guadalupe · Fabiola · Rocco")
    print("="*60 + "\n")
    try:
        bot.infinity_polling(timeout=30, long_polling_timeout=20)
    except KeyboardInterrupt:
        print("\n🛑 Bot detenido manualmente.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        time.sleep(5)






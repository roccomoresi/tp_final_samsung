# 🌱 MENTA - Asistente de Bienestar Alimenticio

> *"Todo empieza desde la consciencia"*

**Equipo:** Guadalupe · Fabiola · Rocco

---

## 📋 Descripción

MENTA es un bot de Telegram que te acompaña en tu camino hacia un bienestar integral. Combina análisis de sentimientos y fotos contando calorías a la vez que te da consejos para cuidar de tu nutrición, reconocimiento de voz y recomendaciones personalizadas para ayudarte a desarrollar una relación más consciente con la alimentación y las emociones.

El bot cuenta con una base de datos SQLite que almacena toda tu información de forma segura y estructurada: cada mensaje, análisis de comida, estado emocional y recomendación queda registrada para construir tu historial completo. Esto permite hacer un recorrido por tu progreso a lo largo del tiempo y generar un dashboard personalizado - un panel visual interactivo en formato HTML con gráficos que muestran la evolución de tu estado emocional, tus patrones alimenticios, y las recomendaciones más frecuentes que recibiste. 

### ✨ Características principales

- 🧠 **Análisis de sentimientos** con NLP en español argentino
- 🎤 **Reconocimiento de voz** (Speech-to-Text con Whisper)
- 📸 **Análisis de comidas** con IA Vision
- 💾 **Memoria contextual** que aprende de tus interacciones
- 📊 **Dashboard interactivo** con gráficos de evolución
- 🎯 **Recomendaciones personalizadas** según tu estado emocional
- 📈 **Seguimiento de progreso** emocional y nutricional

---

## 🚀 Instalación

### Requisitos previos

- Python 3.10 o superior
- Cuenta de Telegram
- API Token de Telegram Bot (obtener de [@BotFather](https://t.me/botfather))
- API Key de Groq 

### 1. Clonar el repositorio

```bash
git clone <tu-repositorio>
cd menta-bot
```

### 2. Crear entorno virtual

```bash
python -m venv venv

# En Windows:
venv\Scripts\activate

# En Linux/Mac:
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```txt
python-dotenv==1.0.0
pyTelegramBotAPI==4.14.0
groq==0.4.2
transformers==4.36.0
torch==2.1.0
matplotlib==3.8.0
pandas==2.1.0
```

### 4. Configurar variables de entorno

Crear archivo `.env` en la raíz del proyecto:

```env
TELEGRAM_TOKEN=tu_token_de_telegram
GROQ_API_KEY=tu_api_key_de_groq
```

### 5. Ejecutar el bot

```bash
python BOT_final.py
```

---

## 🎮 Uso

### Comandos disponibles

| Comando | Descripción |
|---------|-------------|
| `/start` | Iniciar conversación con el bot |
| `/ayuda` | Mostrar lista de comandos y funcionalidades |
| `/progreso` | Ver tu evolución emocional y estadísticas |
| `/dashboard` | Generar dashboard HTML con gráficos detallados |
| `/reset` | Reiniciar la conversación |

### Formas de interactuar

1. **💬 Texto:** Escribe cómo te sientes o pregunta sobre alimentación
2. **🎤 Audio:** Envía un mensaje de voz y el bot lo transcribirá
3. **📸 Foto:** Envía una imagen de tu comida para análisis nutricional

### Ejemplos de uso

```
👤 Usuario: "Me siento ansioso y tengo ganas de comer"
🤖 MENTA: "Tomate unos minutos para respirar y tomar agua. 
          Evitá comer por impulso 🍵"

👤 Usuario: [Envía foto de ensalada]
🤖 MENTA: 
    🍽️ Análisis de tu comida:
    📋 Identificado: lechuga, tomate, zanahoria
    ✅ Evaluación: Saludable
    🔥 Calorías: 150-200 kcal
    💚 Lo bueno:
      • Rica en fibra y vitaminas
      • Bajo aporte calórico
    💡 Consejo: ¡Excelente elección! Podrías agregar 
       una porción de proteína para mayor saciedad
```

---

## 📁 Estructura del proyecto

```
tp_final_samsung/
├── BOT_final.py           #proyecto bot terminado
├── .env                  # Variables de entorno (no incluir en git)
├── requirements.txt      # Dependencias del proyecto
├── README.md            # Este archivo
└── src/
    ├── BOT_final.py      # Script del bot con menos emociones positivas
    ├── DASHBOARD         # Ejemplo de como se ve un dashboard otorgado en Telegram
    ├── + archivos        # muestra el archivo de como fuimos trabajando hasta llegar al archivo BOT_final.py
└── data/
    ├── user_memory.json # Memoria contextual de usuarios
    ├── user_logs.json   # Logs de interacciones
    ├── dataset.json     # Dataset de recomendaciones
└── utils/
    ├── audio_tools.py   # Archivo vacío
    ├── memory_manager.py # Gestiona la memoria temporal del bot por usuario.
    ├── progress_logger.py # Genera un historial de progreso

```

---

## 🛠️ Tecnologías utilizadas

- **[pyTelegramBotAPI](https://github.com/eternnoir/pyTelegramBotAPI)**: Framework para Telegram Bot
- **[Groq](https://groq.com/)**: API para Whisper (STT) y LLaMA Vision
- **[Transformers](https://huggingface.co/transformers/)**: Análisis de sentimientos con RoBERTuito
- **[SQLite](https://www.sqlite.org/)**: Base de datos local
- **[Matplotlib](https://matplotlib.org/)**: Generación de gráficos
- **[Pandas](https://pandas.pydata.org/)**: Análisis de datos

---

## 👥 Autores

- **Guadalupe Martellotta** - *Desarrollo*
- **Fabiola Yépez Oivero** - *Desarrollo*
- **Rocco Moresi** - *Desarrollo*

---
## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---


**🌱 MENTA - Porque el bienestar empieza desde la consciencia**

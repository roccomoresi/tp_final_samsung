"""
sentiment_analysis.py
---------------------
Analiza el sentimiento de un texto en español (positivo, negativo o neutro)
usando un modelo preentrenado de Hugging Face.

Modelo: pysentimiento/robertuito-sentiment-analysis
"""

from transformers import pipeline

# Inicializamos el modelo solo una vez
print("🧠 Cargando modelo de análisis de sentimientos...")
analizador = pipeline("sentiment-analysis", model="pysentimiento/robertuito-sentiment-analysis")

def analizar_sentimiento(texto: str):
    """
    Analiza el sentimiento de una frase en español.

    Parámetros:
        texto (str): Texto a analizar.

    Retorna:
        str: Etiqueta del sentimiento ('POS', 'NEG' o 'NEU').
    """
    try:
        if not texto or texto.strip() == "":
            return "NEU"

        resultado = analizador(texto[:512])[0]  # limitamos longitud por seguridad
        label = resultado["label"].upper()
        score = round(resultado["score"], 3)

        print(f"🩵 Sentimiento detectado: {label} ({score}) para texto: '{texto[:50]}...'")

        # Normalizamos nombres
        if "POS" in label:
            return "POS"
        elif "NEG" in label:
            return "NEG"
        else:
            return "NEU"

    except Exception as e:
        print(f"Error en analizar_sentimiento: {e}")
        return "NEU"

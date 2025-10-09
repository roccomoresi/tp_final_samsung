"""
sentiment_analysis.py
---------------------
Análisis de sentimientos en español usando modelos preentrenados de Transformers.
Detecta emociones básicas (positivo, negativo, neutro) a partir de texto.
"""

from transformers import pipeline

# Crear el pipeline de análisis de sentimientos (modelo multilingüe)
# Este modelo funciona bien para textos en español.
analizador = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

def analizar_sentimiento(texto: str) -> str:
    """
    Analiza el sentimiento de un texto y devuelve una etiqueta simple:
    "positivo", "negativo" o "neutro".
    """
    if not texto or not texto.strip():
        return "neutro"

    resultado = analizador(texto[:512])[0]  # recorte por límite del modelo
    label = resultado["label"].lower()
    score = resultado["score"]

    # Este modelo devuelve etiquetas tipo "1 star" a "5 stars"
    if "1" in label or "2" in label:
        sentimiento = "negativo"
    elif "3" in label:
        sentimiento = "neutro"
    else:
        sentimiento = "positivo"

    print(f"[DEBUG] Texto: {texto}")
    print(f"[DEBUG] Label: {label}, Score: {score:.2f}, Clasificado como: {sentimiento}")

    return sentimiento


if __name__ == "__main__":
    # Ejemplo de prueba
    frases = [
        "Estoy muy feliz con mis hábitos nuevos!",
        "No tengo energía para seguir con la dieta.",
        "Hoy me siento normal, nada especial."
    ]
    for f in frases:
        print(f"\nFrase: {f}")
        print(f"Sentimiento: {analizar_sentimiento(f)}")

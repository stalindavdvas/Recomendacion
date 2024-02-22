import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from flask_cors import CORS
carreras_url = "http://localhost:8181/api/v1/carreras/getAll"  # Corrige la URL
vocacion_url = "http://localhost:8181/api/v1/vocacion/getByEstudiante/"

def get_preguntas_respuestas(id_estudiante):
    try:
        url = vocacion_url + str(id_estudiante)
        data = pd.read_json(url)
        respuestas = " ".join(data["respuesta"])
        return respuestas
    except Exception as e:
        print(f"Error al obtener respuestas para el estudiante {id_estudiante}: {e}")
        return None

def get_similitud(preguntas_respuestas, carreras_df, vectorizer):
    vectores_carreras = vectorizer.fit_transform(carreras_df["nombre"])
    vector_estudiante = vectorizer.transform([preguntas_respuestas])
    similitudes = cosine_similarity(vector_estudiante, vectores_carreras)
    return similitudes.flatten()

app = Flask(__name__)
CORS(app)
@app.route("/recomendar/<id_estudiante>", methods=["GET"])
def recomendar(id_estudiante):
    carreras_df = pd.read_json(carreras_url)  # Mueve esta línea aquí
    vectorizer = TfidfVectorizer( stop_words='english',  # Puedes ajustar esto según tu idioma
    max_df=0.8,
    max_features=5000,
    strip_accents='unicode',
    analyzer='word',
    ngram_range=(1, 3))
    preguntas_respuestas = get_preguntas_respuestas(id_estudiante)
    if preguntas_respuestas is not None:
        similitudes = get_similitud(preguntas_respuestas, carreras_df, vectorizer)
        carreras_recomendadas = carreras_df.iloc[np.argsort(similitudes)[-5:]]
        return jsonify(carreras_recomendadas.to_dict("records"))
    else:
        # Manejar el caso en el que no hay respuestas para el estudiante
        return jsonify({"error": "No se encontraron respuestas para el estudiante"}), 404

if __name__ == "__main__":
    app.run(debug=True)

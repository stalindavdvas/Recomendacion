import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from flask import Flask, jsonify
from flask_cors import CORS

carreras_url = "http://localhost:8181/api/v1/carreras/getAll"
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


def obtener_embeddings(texto, modelo, tokenizer):
    tokens = tokenizer(texto, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = modelo(**tokens)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return embeddings


def get_similitud(embeddings_estudiante, embeddings_carreras):
    similitudes = cosine_similarity([embeddings_estudiante], embeddings_carreras)
    return similitudes.flatten()


app = Flask(__name__)
CORS(app)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
modelo = BertModel.from_pretrained("bert-base-uncased")
modelo.eval()


@app.route("/recomendar/<id_estudiante>", methods=["GET"])
def recomendar(id_estudiante):
    carreras_df = pd.read_json(carreras_url)
    preguntas_respuestas = get_preguntas_respuestas(id_estudiante)

    if preguntas_respuestas is not None:
        embeddings_estudiante = obtener_embeddings(preguntas_respuestas, modelo, tokenizer)
        embeddings_carreras = carreras_df["nombre"].apply(lambda x: obtener_embeddings(x, modelo, tokenizer))

        similitudes = get_similitud(embeddings_estudiante, np.stack(embeddings_carreras))
        carreras_recomendadas = carreras_df.iloc[np.argsort(similitudes)[-5:]]

        return jsonify(carreras_recomendadas.to_dict("records"))
    else:
        return jsonify({"error": "No se encontraron respuestas para el estudiante"}), 404


if __name__ == "__main__":
    app.run(debug=True)

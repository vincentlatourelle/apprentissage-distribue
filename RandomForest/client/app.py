"""Application Flask qui traite les demandes du master et les renvoies a la classe Client
    Chaque fonction est associe a une fonction du Client
"""

from joblib import dump, load
import io
import os
import sys

import numpy as np
import pandas as pd
from client import Client
from flask import Flask, jsonify, request, send_file

currentdir = os.path.dirname(os.path.realpath(__file__))
rootdir = os.path.join(currentdir, "../../")
sys.path.append(rootdir)

from RandomForest.randomForest import RandomForest
from RandomForest.node import Node


app = Flask(__name__)

c = Client()


@app.route('/rf/best-threshold')
def get_best_threshold():
    features = request.get_json()['features']
    threshold = request.get_json()['thresholds']
    current_tree = Node.deserialize(request.get_json()['current_tree'])

    feature, n_data = c.get_best_threshold(features, threshold, current_tree)
    # print(
    #     "Le client recoit l'arbre actuel et threshold choisis par le master et retourne le meilleur feature et son "
    #     "nombre de donnes",
    #     file=sys.stderr)
    return jsonify({"feature": feature, "n_data": n_data})


@app.route('/rf/leaf')
def get_leaf():
    current_tree = Node.deserialize(request.get_json()['current_tree'])
    labels = c.get_leaf(current_tree)

    # print("Le client recoit l'arbre actuel et renvoit les labels associes au noeud actuel", file=sys.stderr)

    return jsonify(labels.tolist())


@app.route('/rf/leaf-vote')
def get_leaf_vote():
    current_tree = Node.deserialize(request.get_json()['current_tree'])
    label, count = c.get_leaf_vote(current_tree)

    # print("Le client recoit l'arbre actuel et renvoit les labels associes au noeud actuel", file=sys.stderr)

    return jsonify({"label": label, "count": count})


@app.route('/rf/random-forest', methods=['POST'])
def set_new_forest():
    if request.get_json() is not None:
        random_forest = RandomForest()
        random_forest.deserialize(request.get_json()["forest"])
        c.set_new_forest(random_forest)
    else:
        file = request.files['file']
        model = load(io.BytesIO(file.read()))
        c.set_new_forest(model)

    return "", 200


@app.route('/rf/local-accuracy')
def get_local_accuracy():
    accuracy, n = c.get_local_accuracy()

    return jsonify({"accuracy": accuracy, "n": n})


@app.route('/rf/federated-accuracy')
def get_federated_accuracy():
    accuracy, n = c.get_federated_accuracy()

    return jsonify({"accuracy": accuracy, "n": n})


@app.route('/dataset', methods=['POST'])
def set_dataset():
    """Set le dataset du client (pour simplifier les tests)
    """

    dataset_dict = request.get_json()["dataset"]
    dataset_labels = np.array(list(request.get_json()["labels"].values()))
    dataset = pd.DataFrame.from_dict(dataset_dict)

    c.set_dataset(dataset, dataset_labels)

    return "", 200


@app.route('/rf/thresholds')
def get_thresholds():
    current_tree = Node.deserialize(request.get_json()['current_tree'])
    features = request.get_json()['features']

    # print("Le client recoit les features et l'arbre actuel et renvoit une valeur pour chaque feature", file=sys.stderr)

    values = c.get_thresholds(features, current_tree)

    return jsonify(values)


@app.route('/rf/features')
def get_features():
    features = c.get_features()

    return jsonify(features)


@app.route('/rf/local-model')
def get_local_model():
    model = c.get_local_model()
    bytes_io = io.BytesIO()
    dump(model, bytes_io)
    bytes_io.seek(0)

    return send_file(
        bytes_io,
        attachment_filename='model',
    )


@app.route('/rf/set-validation')
def set_validation():
    c.set_validation()
    return jsonify({})


@app.route('/rf/unset-validation')
def unset_validation():
    c.unset_validation()
    return jsonify({})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

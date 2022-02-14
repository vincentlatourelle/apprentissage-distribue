import json
import os, sys
import numpy as np

import pandas as pd
from flask import Flask, jsonify, request
from client import Client

currentdir = os.path.dirname(os.path.realpath(__file__))
rootdir = os.path.join(currentdir, "../../")
sys.path.append(rootdir)

from RandomForest.node import Node

app = Flask(__name__)

c = Client()


@app.route('/best-threshold')
def get_best_threshold():
    features = request.get_json()['features']
    threshold = request.get_json()['thresholds']
    current_tree = Node.deserialize(request.get_json()['current_tree'])

    feature, n_data = c.get_best_threshold(features, threshold, current_tree)

    return jsonify({"feature": feature, "n_data": n_data})


@app.route('/leaf')
def get_leaf():
    current_tree = Node.deserialize(request.get_json()['current_tree'])
    labels = c.get_leaf(current_tree)

    return jsonify(labels.tolist())


@app.route('/random-forest', methods=['POST'])
def set_new_forest(random_forest):
    random_forest = request.get_json()["random_forest"]
    c.set_new_forest(random_forest)

    return "", 200


@app.route('/local-accuracy')
def get_local_accuracy():
    dataset_dict = request.get_json()["dataset"]
    dataset_labels = np.array(list(request.get_json()["labels"].values()))
    dataset = pd.DataFrame.from_dict(dataset_dict)

    accuracy = c.get_local_accuracy(dataset, dataset_labels)

    return jsonify(accuracy)


@app.route('/dataset', methods=['POST'])
def set_dataset():
    """Set le dataset du client (pour simplifier les tests)
    """

    dataset_dict = request.get_json()["dataset"]
    dataset_labels = np.array(list(request.get_json()["labels"].values()))
    dataset = pd.DataFrame.from_dict(dataset_dict)
    c.dataset = dataset
    c.labels = dataset_labels

    print(dataset, file=sys.stderr)

    return "", 200


@app.route('/thresholds')
def get_thresholds():
    current_tree = Node.deserialize(request.get_json()['current_tree'])
    features = request.get_json()['features']
    values = c.get_thresholds(features, current_tree)
    return jsonify(values)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

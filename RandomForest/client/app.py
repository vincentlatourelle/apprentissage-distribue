import json
import os, sys

import pandas as pd
from flask import Flask, jsonify, request
from client import Client


currentdir = os.path.dirname(os.path.realpath(__file__))
rootdir = os.path.join(currentdir, "../../")
sys.path.append(rootdir)

from RandomForest.node import Node

app = Flask(__name__)

c = Client()
 
@app.route('/best-split')
def get_best_split():
    features = request.get_json()['features']
    splits = request.get_json()['splits']
    current_tree = Node.deserialize(request.get_json()['current_tree'])
    
    c.get_best_split(features,splits,current_tree)
    
    return jsonify('')

@app.route('/leaf')
def get_leaf():
    current_tree = request.get_json()["current_tree"]
    c.get_leaf(current_tree)
    
    return jsonify('')

@app.route('/random-forest', methods=['POST'])
def set_new_forest(random_forest):
    random_forest = request.get_json()["random_forest"]
    c.set_new_forest(random_forest)
    
    return "", 200

@app.route('/local-accuracy')
def get_local_accuracy():
    c.get_local_accuracy()
    
    return jsonify('')
    

@app.route('/dataset', methods=['POST'])
def set_dataset():
    """Set le dataset du client (pour simplifier les tests)
    """
    
    dataset_dict = request.get_json()["dataset"]
    dataset_labels = request.get_json()["labels"]
    dataset = pd.DataFrame.from_dict(dataset_dict)
    c.dataset = dataset
    
    print(dataset, file=sys.stderr)
    
    return "", 200
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

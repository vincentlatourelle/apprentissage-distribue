# Apprentissage distribué

## Prérequis
- Docker et docker-compose
- Python et pip
- Installer les requirements : ```pip install -r requirements.txt```

## Exécution

- Exécuter ```docker-compose up --build``` à partir de la racine du projet afin de démarrer les clients.
- Exécuter ```python network_creator.py .\BCWdata.csv 2 0.5 diagnosis``` :
    - ```.\BCWdata.csv``` est l'emplacement du fichier csv contenant.
    - ```2``` est le nombre de clients. Si cette valeur est mise à 0, un ensemble de tests est executé.
    - ```0.5``` est la répartition des données entre les clients, ce paramètre est pris en compte seulement quand 2 clients sont utilisés pour l'instant.
    - ```diagnosis``` est la colonne associée à la cible de prédiction dans le fichier de données.


## Test
Exécuter ```py test.py``` pour vérifier que tout est fonctionnel.
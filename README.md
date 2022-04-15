# Apprentissage distribué

## Prérequis
- Il est recommandé d'utiliser une distribution linux, car la création de plusieurs clients avec *Docker for Windows 4.6.1* est instable pour l'attribution des ports. Il serait nécessaire de modifier le fichier *docker-compose.yml* pour attribuer manuellement les ports à chacun des clients (voir exemple en commentaire dans docker-compose.yml). 
- Docker et docker-compose
- Python et pip
- Installer les requirements : ```pip install -r requirements.txt```

## Exécution

- Exécuter ```sudo docker-compose up --scale client={nb-clients}``` à partir de la racine du projet afin de démarrer les clients. ```{nb-clients}``` est le nombre maximum de clients utilisés (100 par defaut).
- Exécuter ```python3 network_creator.py BCWdata.csv 2 0.5 diagnosis 1``` :
    - ```.\BCWdata.csv``` est l'emplacement du fichier csv contenant.
    - ```2``` est le nombre de clients. Si cette valeur est mise à 0, un ensemble de tests est executé.
    - ```0.5``` est la répartition des données entre les clients, ce paramètre est pris en compte seulement quand 2 clients sont utilisés pour l'instant.
    - ```diagnosis``` est la colonne associée à la cible de prédiction dans le fichier de données.
    - ```1``` si on veut utiliser la cross-validation (0 si on ne la veut pas).


## Test
Exécuter ```python3 test.py``` pour vérifier que tout est fonctionnel.
---
title: Devops Ml
emoji: 🚀
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
---

# devops-ml

# Lancer le projet directement avec Docker
```bash
#Installer toutes les dépendances, le projet est lourd (7go) donc l'installation peut prendre du temps.
docker compose up --build

#Lancer les services en arrière plan
docker compose up -d

#le backend est disponible sur le port 8000
```

## utiliser les APIs sur le swagger
L'api de classification est api/classify. Il vous suffit de rentrer le destinataire, le sujet et le corps du mail et l'api retourne un json avec la classification associée du mail et sa probabilité. L'API fonctionne uniquement avec des emails en anglais. Vous pouvez tester directement via le swagger du backend : 
```
http://127.0.0.1:8000/docs
```
## visualisation de metriques
Une instance de grafana est disponible sur le port 3000 afin de visualiser les différentes métriques de base associé à l'appel de l'API de classification.
```bash
localhost:3000

#logs

admin
admin
```

## déploiment
Le backend a été déployé sur Huggingface space à l'adresse suivante
```
https://matyschampeyrol-devops-ml.hf.space

#pour acceder au swagger et tester l'api de classification de mail
https://matyschampeyrol-devops-ml.hf.space/docs
```

# dataset 
Dans api/dataset/dataset.py, un dataset de 100 email est généré à l'éxécution, labélisé, et envoyé à l'API de classification afin de pouvoir mesurer l'accuracy du modèle.
Le résultat de ces targets sont disponible sur mlflow au lancement du conteneur dataset-runner (automatique avec docker compose up), ou bien du script api/src/dataset/dataset.py

- lancer mlflow

```bash
##AVEC DOCKER
#accéder simplement au port 5000
localhost:5000

##OU

#dans le terminal
mlflow ui

#ensuite connectez-vous au port 5000
localhost:5000
```

# Host le projet en local sans Docker
## télécharger les dépendances
```bash
se mettre à la racine du projet

pip install --no-cache-dir -r api/requirements.txt
```

## lancer le serveur fastapi
le service est disponible sur le port 8000
```bash
uvicorn api.main:app --reload
```

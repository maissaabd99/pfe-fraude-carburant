# Utiliser l'image de base Python
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /Mise-en-place-dun-detecteur-de-fraude-de-carburant

# Copier tous les fichiers du répertoire actuel dans le répertoire de travail du conteneur Docker
COPY . /Mise-en-place-dun-detecteur-de-fraude-de-carburant

# Installer les dépendances
RUN pip install -r requirements.txt

# Exposer le port sur lequel Flask écoute les requêtes
EXPOSE 5000

# Commande par défaut pour exécuter l'application Flask
CMD ["python", "Flask_main.py"]


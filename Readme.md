F1 MACHINE LEARNING PROJECT

Ce projet de machine learning applique des techniques de classification et de régression à des données historiques de Formule 1 (1950 à 2024), ainsi qu'à une simulation complète de la saison 2025.

## Objectifs

Le projet est structuré en deux axes principaux :

### 1. Classification
Prédiction d'événements binaires au cours d'une course :
- Le pilote termine-t-il la course ?
- Monte-t-il sur le podium ?
- Bat-il son coéquipier ?
- Part-il dans le top 10 ?
- Gagne-t-il des places en course ?

### 2. Régression
Prédiction de variables continues liées aux performances :
- Nombre de points marqués
- Position finale
- Nombre de places gagnées
- Temps total de course
- Points marqués par une écurie

---

## Structure du projet

### data/
- `constructors.csv` : Données sur les écuries  
- `drivers.csv` : Infos générales des pilotes  
- `races.csv` : Informations des Grand Prix  
- `results.csv` : Résultats de chaque course  
- `status.csv` : Statuts de fin de course  
- `driver_standings.csv` : Classement championnat pilotes  
- `data_filter.csv` : Données nettoyées et enrichies  
- `data_filter_cleaned.csv` : Variante nettoyée (optionnelle)  

#### data/season2025/
- `data_2025.csv` : Données simulées pour la saison 2025  
- `drivers_2025.csv` : Line-up des pilotes 2025  
- `predictions_2025.csv` : Résultats prédits (grille + position finale)  
- `scores_pilotes.csv` : Moyenne des positions finales prédites par pilote  
- `scores_ecuries.csv` : Moyenne des positions finales par écurie  

### notebooks/
- `EDA.ipynb` : Analyse exploratoire  
- `classification.ipynb` : Modèles pour cibles binaires  
- `regression.ipynb` : Modèles pour cibles continues  
- `prediction_2025.ipynb` : Application des modèles sur la saison simulée  
- `model_regression_final_position.joblib` : Modèle sauvegardé (régression finale)  

### scripts/
- `prepare_filter_data.py` : Préparation des données historiques (fusion, enrichissement)  
- `prepare_data_2025.py` : Génération des données pour la saison 2025  

### Autres fichiers
- `environment.yml` : Dépendances Conda  
- `README.md` : Présentation du projet  
- `Plan.txt` : Planning ou structure de travail  
- `FriseChronoF1.png` : Illustration ou frise du projet  

---

## Installation et lancement

```bash
conda env create -f environment.yml
conda activate f1-ml
```

## Méthodologie Machine Learning

Le projet repose sur une approche rigoureuse basée sur les bonnes pratiques du machine learning supervisé.

### Étapes principales

- Création de pipelines avec `scikit-learn` (prétraitement + modèle)
- Séparation des données en jeu d'entraînement / validation
- Imputation des valeurs manquantes (médiane ou constante)
- Encodage des variables catégorielles (OneHotEncoder)
- Standardisation des variables numériques (StandardScaler)
- Sélection de features pertinentes selon la cible
- Validation croisée (cross-validation à 5 folds)
- Optimisation d’hyperparamètres avec `GridSearchCV`
- Évaluation des modèles avec des métriques adaptées

### Évaluation

- Classification : `accuracy`, `f1-score`, `confusion_matrix`
- Régression : `MAE`, `RMSE`, `R²`, analyse des résidus

### Modèles testés

#### Classification :
- `RandomForestClassifier` (méthode ensembliste)
- `AdaBoostClassifier` (méthode ensembliste)
- `SVC` (Support Vector Classifier)
- `KNeighborsClassifier` (non vu en cours)
- `MLPClassifier` (réseau de neurones)

#### Régression :
- `RandomForestRegressor` (méthode ensembliste)
- `AdaBoostRegressor` (méthode ensembliste)
- `SVR` (Support Vector Regression)
- `MLPRegressor` (réseau de neurones)
- `SGDRegressor` (descente de gradient stochastique)


## Auteur

**Raphaël Jouannet**  
Étudiant en 4e année à l’ESEO (Angers) 

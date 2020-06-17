Gini-PLS 
==============================

Un algorithme de classification supervisée basée sur la méthode Gini-PLS généralisée

Project structure
---------------------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data      
    │   └── processed      <- un petit ensemble de données pour tester que le code fonctionne (src/unittests/test_ginipls.py
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`so src can be imported
    └── ginipls                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to load, download or generate data
        │   └── make_dataset.py
        │   └── data_utils.py <- main
        │
        ├── experiments    <- Scripts to run experiments
        │   └── learning   <- fonction de sélection des hyperparamètres (le nu du Gini-PLS)
        │   └── api.py et classification.py     <-  appel des fonctions d'entrainement ou test des classifieurs
        │   └── scripts > evaluation.py  <- SCRIPT PRINCIPAL D'EVALUATION
        │
        └── models         <- Scripts to train models and then use trained models to make
            ├── ginipls.py <- Implémentation des variantes du Gini PLS
            ├── hyperparameters.py <- Implémentation du GridSearch pour déterminer de bonnes valeurs pour le nu et le nombre de composantes
        
Running
-------
`(py36)$> python -m ginipls --help`

TODO
----
### Préparation des données pour expérimentation (raw -> interim)
Pour chaque catégorie de demandes :
*  Créer un dossier pour cette catégorie 
*  A partir du tableau des annotations de demandes CASSANDRA.xls, filtrer les documents des décisions de justice qui n'ont qu'une seule demande en les répartissant en 2 sous-dossiers pour les 2 classes : **0-rejette** et **1-accepte**
```
python -m ginipls.data.make_dataset select-data taj-sens-resultat-data "amende civile" "32-1 code de procédure civile + 559 code de procédure civile : pour procédure abusive" data/raw/txt-all/acpa data/raw/CASSANDRA.tsv data/raw/txt-oneclaim/acpa
```
*  pré-traiter les données; 
```
python -m ginipls.data.make_dataset --logging preprocess taj-sens-resultat --language=fr --lowercase --lemmatizer=treetagger data/raw/taj_sens_resultat/acpa data/interim/taj-sens-resultat-pp/acpa.tsv
```
*  Répartir les données de chaque classe en 4 folds pour réaliser une validation croisée
```
python -m ginipls.data.make_dataset form-evaluation-data cv-traintest-from-dataset-file 4 data/interim/taj-sens-resultat-pp/acpa.tsv data/interim/taj-sens-resultat-cv
```
*  Pour chaque itération de la validation croisée, choisir 1 fold pour chaque classe pour former le jeu de test, et former le jeu d'entraînement avec les folds restants
```
python -m ginipls.data.make_dataset form-evaluation-data cv-traintest-from-dataset-file 4 data/interim/taj-sens-resultat-pp/acpa.tsv data/interim/taj-sens-resultat-cv
```
### Pré-traitement
Le pré-traitement a pour but de réduire le nombre de variantes des mots ou concept qui peuvent être parfois unique à un document. Ainsi, les variantes d'éléments propres à un label seront identiques. Pour la classification de textes, le pré-traitement consiste généralement à :
*  extraire la zone de texte à utiliser
*  lemmatiser le texte
*  éliminer les mots inutiles à la distinction de labels : les mots à un caractère, les ponctuations, les nombres, et les mots qui n'apparaissent que dans un seul document
*  mettre le texte en minuscule
### Représentation vectorielle (interim -> processed)
Pour chaque itération de validation croisée :
*  le jeu d'entrainement est utilisé pour apprendre le poids global (IDF ou DSIDF si VSM)
  * TFIDF
```
python -m ginipls.data.make_dataset --logging vectorize --vsm_scheme=tfidf --label_col=@label --index_col=@id --text_col=@text --ngram_nmax=2 data/interim/taj-sens-resultat-cv/acpa_cv0_train.tsv data/models/taj-sens-resultat/acpa_cv0.tfidf12 data/processed/taj-sens-resultat/acpa_cv0_train_tfidf12.tsv
python -m ginipls.data.make_dataset --logging vectorize --vsm_scheme=tfidf --label_col=@label --index_col=@id --text_col=@text --ngram_nmax=2 data/interim/taj-sens-resultat-cv/acpa_cv0_test.tsv data/models/taj-sens-resultat/acpa_cv0.tfidf12 data/processed/taj-sens-resultat/acpa_cv0_test_tfidf12.tsv
```
* TFCHI2
```
python -m ginipls.data.make_dataset --logging vectorize --vsm_scheme=tfchi2 --label_col=@label --index_col=@id --text_col=@text --ngram_nmax=2 data/interim/taj-sens-resultat-cv/acpa_cv0_train.tsv data/models/taj-sens-resultat/acpa_cv0.tfchi212 data/processed/taj-sens-resultat/acpa_cv0_train_tfchi212.tsv
python -m ginipls.data.make_dataset --logging vectorize --vsm_scheme=tfchi2 --label_col=@label --index_col=@id --text_col=@text --ngram_nmax=2 data/interim/taj-sens-resultat-cv/acpa_cv0_test.tsv data/models/taj-sens-resultat/acpa_cv0.tfchi212 data/processed/taj-sens-resultat/acpa_cv0_test_tfchi212.tsv
```
*  les jeux de test et train sont vectorisés pour constituer les deux matrices d'entraînement et de test
### Entraînement du classifieur (data/processed -> models)
Pour chaque itération de validation croisée :
*  employer la matrice d'entrainement pour entrainer le modèle de classification
  *  Si le modèle a des hyperparamètres, utiliser une validation croisée sur cette matrice pour les déterminer 
### Validation du classifieur (data/processed/trainmat ou testmat + models/... -> reports/predictions)
*  Pour chaque itération de validation croisée :
  *  appliquer le modèle entrainé sur les matrices d'entraînement et de test, et reporter les prédictions dans le seul fichier de prédictions de la CV   
*  appliquer le script d'évaluation sur le fichier de prédictions pour obtenir les valeurs des métrique de test (moyenne-macro des métriques précision, rappel et f1; erreur sur chaque label)


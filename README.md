Gini-PLS 
==============================

Un algorithme de classification supervisée basée sur la méthode Gini-PLS généralisée

Organisation du code source
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
        
   

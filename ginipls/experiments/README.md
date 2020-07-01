# Reproduction des expérimentations 

## Préparation des données

### Sélection des décisions à une seule demande pour chaque catégorie dans le dataset des demandes (taj java)

 ??? retrouver comment c'est fait

### Sectionnement (taj java)

Avec l'interface graphique de l'application java TAJ 

### Récupération des zones lemmatisées (taj java)

* récupérer les dossier section des différents sens du résultat dans les dossiers tmp > xml_wordposlemma : a la main
* nettoyer les xml en gardant les lemme > xml_lemma : taj\src\Main\TAJ.java args = new String[]{"-clean-xml", baseDir + "/xml_wordposlemma", "2", baseDir + "/xml-lemma"};
* générer 2 corpus sans elliminer les stopwords
  * full-lemma : toute la décision lemmatisées : taj\src\Main\TAJ.java args = new String[]{"-get-node-content", baseDir + "/xml-lemma", baseDir + "/arret_lemma", "arret"};
  * litige-motifs-dispositif_lemma : uniquement la combinaison litige-motifs-dispositif

### Générer des splits pour une k-fold cv (taj java)

* avec extraction.demande.evaluation.EvaluationConfigurator.generateDatasetGinipls()

### Vectorisation tf-idf et averagelocals-averageglobals (taj java)

* avec data_representation_pls.Vectorization.main() > 4folds/acpa

### Copier les vecteurs dans  le sous-dossier processed (os)

* cp 4folds/acpa/wd-1_1-tsv/*tsv > 4folds/processed

### Excuter le traintest (taj python)

python -m ginipls.experiments.cv_traintest_taj_sens_resultat acpa ...\taj\chap4\wd\litige-motifs-dispositif_lemma\4folds

### Excuter l'évaluation (taj python)

python -m ginipls.experiments.cv_eval_taj_sens_resultat acpa ...\taj\chap4\wd\litige-motifs-dispositif_lemma\4folds

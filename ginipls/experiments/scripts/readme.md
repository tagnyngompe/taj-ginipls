## Générer les folds et les trainset/testset
taj.pls.data.py > cv-xml

## Supprimer les marqueurs de normes et d'argent
```
bash 
cd cv-xml
find . -type f -exec sed -i 's/<[^>]*argent[^>]*>//g' {} +
find . -type f -exec sed -i 's/<[^>]*norme[^>]*>//g' {} +
```

## Prétraiter les documents de chaque fold et Extraire le texte
Avec java pour l'instant

cv-xml > cv-texte

## vectoriser les textes par gw*lw
Avec java pour l'instant

## vectoriser les textes par word-cluster
Avec Python: taj.scripts.pls.wordClusterVocIndex.py

## vectoriser les textes par agrégation de word embedding  
Avec Python: taj.embedding.doc_embedding.py
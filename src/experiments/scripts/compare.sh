#!/usr/bin/env bash
metric=$1
catDmd=$2
f="../resultat-xp/compare_zone_${catDmd}_${metric}.csv"
# echo $f
echo "categorie zone vectorization classifier meanScore worstScore worstCategory bestScore bestCategory distanceToBestConfig deltaBestWorst(Stability) rank" > $f
python bestconfig.py $metric $catDmd >> $f
# python bestconfig.py ../cv-demande_resultat_a_resultat_context/ $metric $catDmd >> $f
# python bestconfig.py ../cv-motifs/ $metric $catDmd >> $f
# python bestconfig.py ../cv-litige_motifs_dispositif/ $metric $catDmd >> $f

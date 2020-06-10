#!/usr/bin/env bash
function getLastElt(){
SEP="${2-/}"
#COLUMN="${2-/}"
while IFS="${SEP}" read -ra ADDR; do
#       for i in "${ADDR[@]}"; do
#           test
#       done
#    echo $SEP
    echo ${ADDR[-1]}
 done <<< $1
} 
vector_dir=$(echo $(getLastElt "${2}" /))
echo $vector_dir
python evaluation.py -c "$1" -o "@sens-resultat" -i "@id" -m "${2}" -r "../../../reports/resultat-${vector_dir}" -k 4 -b False -v ../../../data/processed/vectorizations-tfidf.txt
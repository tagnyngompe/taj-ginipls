entitylabels=("acpa" "concdel" "danais" "dcppc" "doris" "styx")
nrep=$3
for catDmd in ${entitylabels[@]}; do
    if [ $1 = "fasttext" ]; then
        source activate py36
        bash evalfasttext.sh "$2/${catDmd}_lemma_4_folds_cv" $nrep
        source deactivate
    elif [ $1 = "nbsvm" ]
    then
        source activate py27
        bash evalnbsvm.sh "$2/${catDmd}_lemma_4_folds_cv" ${catDmd}-accepte ${catDmd}-rejette  $nrep
        source deactivate
    else
    echo "Usage: runallstateofart fasttext|nbsvm <cvdir> <nb_train-test>"
    exit
    fi
    #break
done 

# Usage: bash runallstateofart.sh nbsvm ../cv-context/ 4
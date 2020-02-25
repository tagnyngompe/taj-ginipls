function normalize_text() {
	#echo $2
  echo $(awk '{print tolower($0);}' < $1 | sed -z 's/\n/ /g' | sed -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/"/ " /g' \
  -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' -e 's/\?/ \? /g' \
  -e 's/\;/ \; /g' -e 's/\:/ \: /g') >> $2
}

cvdir="$1"
class1=$2
class2=$3
nrep=$4
START=0
END=$(bc <<< "${nrep}-1")
nbsvmWd="$cvdir/nbsvm"
mkdir "$nbsvmWd"
datadir="$nbsvmWd/data"
mkdir "$datadir"
modeldir="$nbsvmWd/model"
mkdir "$modeldir"
predictiondir="$nbsvmWd/prediction"
mkdir "$predictiondir"

NBSVMHOME="/home/tagny/git/cassandra-python/nbsvm"
for k in $(seq $START $END); do    
    originaldatadir="$cvdir/$k"
    for j in train/$class1 train/$class2 test/$class1 test/$class2; do
        normf=$datadir/$(echo $j | sed -r "s/\//${k}_/g" | sed -e 's/\.//g')".txt"
        rm $normf
        #echo $datadir/$j
        for i in `ls $originaldatadir/$j`; do normalize_text $originaldatadir/$j/$i $normf ; done           
        echo $normf
    done

    predictionnameTmp="$predictiondir/test${k}.pred.tmp"
    predictionpath="$predictiondir/test${k}.pred"
    label1trainpath="$datadir/train${k}_$class1.txt"
    label2trainpath="$datadir/train${k}_$class2.txt"
    label1testpath="$datadir/test${k}_$class1.txt"
    label2testpath="$datadir/test${k}_$class2.txt"        
    # echo "BI-GRAM";
    # python $NBSVMHOME/nbsvm.py --liblinear $NBSVMHOME/liblinear-1.96 --ptrain $datadir/train_$class1.txt --ntrain $datadir/train_$class2.txt --ptest $datadir/test_$class1.txt --ntest $datadir/test_$class2.txt --ngram 12 --out $datadir/NBSVM-TEST-BIGRAM
    echo "TRI-GRAM";
    python $NBSVMHOME/nbsvm.py --liblinear $NBSVMHOME/liblinear-1.96 --ptrain $label1trainpath --ntrain $label2trainpath --ptest $label1testpath --ntest $label2testpath --ngram 123 --out $predictionnameTmp
    #echo  "$datadir/test_$class1.txt"
    python concatTruePrednbsvm.py $label1testpath $label2testpath $predictionnameTmp > $predictionpath    
    rm $predictionnameTmp
    # echo "4-GRAM";
    # python $NBSVMHOME/nbsvm.py --liblinear $NBSVMHOME/liblinear-1.96 --ptrain $datadir/train_$class1.txt --ntrain $datadir/train_$class2.txt --ptest $datadir/test_$class1.txt --ntest $datadir/test_$class2.txt --ngram 1234 --out $datadir/NBSVM-TEST-4GRAM
    #break
done

# USAGE:
# source activate py27
# bash evalnbsvm.sh ../cv-motifs/acpa_lemma_4_folds_cv acpa-accepte acpa-rejette  4

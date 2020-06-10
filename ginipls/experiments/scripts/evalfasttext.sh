cvdir=$1
fasttextWd="$cvdir/fasttext"
mkdir "$fasttextWd"
datadir="$fasttextWd/data"
mkdir "$datadir"
modeldir="$fasttextWd/model"
mkdir "$modeldir"
predictiondir="$fasttextWd/prediction"
mkdir "$predictiondir"
nrep=$2
START=0
END=$(bc <<< "${nrep}-1")
# format data for fasttext
python3 fastTextFormat.py "$cvdir" $nrep
# nrep-fold cross validation
for k in $(seq $START $END); do
    modelname="$modeldir/model$k"
    trainpath="$datadir/train$k.txt"
    testpath="$datadir/test$k.txt" 
    predictionpath="$predictiondir/test$k.pred"
    fasttext supervised -input $trainpath -output $modelname -lr 0.005 -epoch 50 -wordNgrams 3
    echo "-- Test on $testpath  --"
    nbTestSamples=$(wc -l $testpath)
    #fasttext test $modelname.bin $testpath $nbTestSamples
    fasttext predict $modelname.bin $testpath > "$predictionpath.tmp"
    python concatTruePredFastText.py $testpath "$predictionpath.tmp" > $predictionpath    
    rm "$predictionpath.tmp" 
#     break
done 
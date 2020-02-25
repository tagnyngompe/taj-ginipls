import os, re, sys

def readLabel(filename):
    labels  = []
    with open(filename, 'r') as fr:
        for line in fr:
            line = line.strip()
            if len(line) == 0:
                continue
            labels.append(line.split()[0])
    return labels

if __name__ == "__main__":
    Ytrue = readLabel(sys.argv[1])
    Ypred = readLabel(sys.argv[2])    
    for ytrue, ypred in zip(Ytrue, Ypred):
        print(ytrue+" "+ypred)
 
# USAGE: python concatTruePredFasttext.py ../cv-context/acpa_lemma_4_folds_cv/fasttext/data/test0.txt ../cv-context/acpa_lemma_4_folds_cv/fasttext/prediction/test0.pred

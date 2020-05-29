import os, re, sys

def countLines(filename):
    nblines=0
    with open(filename, 'r') as fr:
        for line in fr:
            line = line.strip()
            if len(line) == 0:
                continue
            nblines+=1
    return nblines

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
    #print("args: "+sys.argv[0])
    label1testfname = sys.argv[1]
    label2testfname = sys.argv[2]
    #print("concat \n"+label1testfname+"\n"+label2testfname)
    YtrueLabel1 = ["1"] * countLines(label1testfname)
    YtrueLabel2 = ["-1"] * countLines(label2testfname)
    Ytrue=YtrueLabel1+YtrueLabel2
    Ypred = readLabel(sys.argv[3])[1:]
    for ytrue, ypred in zip(Ytrue, Ypred):
        print(ytrue+" "+ypred)
 
# USAGE: python concatTruePrednbsvm.py ../cv-motifs/acpa_lemma_4_folds_cv/nbsvm/data/test_acpa-rejette.txt ../cv-motifs/acpa_lemma_4_folds_cv/nbsvm/data/test_acpa-accepte.txt ../cv-motifs/acpa_lemma_4_folds_cv/nbsvm/prediction/test0.pred.tmp

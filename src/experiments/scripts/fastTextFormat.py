import os, re, sys

def folder2file(textsFolderPath, fastTextInFilePath):
    if not os.path.exists(os.path.dirname(textsFolderPath)):
        return 
    textsFolderPath = os.path.abspath(textsFolderPath)
    if not os.path.exists(os.path.dirname(fastTextInFilePath)):
        os.makedirs(os.path.dirname(fastTextInFilePath))
    #textfnames = [os.path.join(textsFolderPath, name)
             #for root, dirs, files in os.walk(textsFolderPath)
             #for name in files
             ##if name.endswith((".html", ".htm"))
             #]
    textfnames = []
    for dirname, subdirnames, _ in os.walk(textsFolderPath):
        for subdirname in subdirnames:
            #print(subdirname)
            for root, _, filenames in os.walk(os.path.join(dirname, subdirname)):
                for fname in filenames:
                    textfnames += [os.path.join(root, fname)]
    pattern = re.compile(r'\s+')
    with open(fastTextInFilePath, 'w', encoding="utf-8") as fw:
        for fname in textfnames:
            #print(fname)            
            text = ""
            with open(os.path.join(dirname, fname), 'r', encoding="utf-8") as fr:
                for line in fr:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    text += line +" "
                text = re.sub(pattern, ' ', text.strip())+"."
                label = os.path.basename(os.path.dirname(fname))
                #print("__label__"+label, text[0:20])
                fw.write("__label__"+label+" "+text+"\n")
    
if __name__ == "__main__":
    #zonewd = "/media/tagny/Dell-USB-Portable-HDD/D/current-work/sens-resultat/cv-demande_resultat_a_resultat_context"
    #catDmd = "acpa"
    #nrep=4
    basePath = sys.argv[1]
    nrep=int(sys.argv[2])    
    
    for i in range(nrep):
        for dirname in {"train", "test"}:
            textsFolderPath = basePath + "/"+str(i)+"/"+dirname
            foldNum = os.path.basename(os.path.dirname(textsFolderPath))
            destPath = basePath+"/fasttext/data"
            datafname = os.path.basename(textsFolderPath)+foldNum+".txt"
            fastTextInFilePath = os.path.join(destPath, datafname)
            folder2file(textsFolderPath, fastTextInFilePath)
# python3 fastTextFormat.py /media/tagny/Dell-USB-Portable-HDD/D/current-work/sens-resultat/cv-demande_resultat_a_resultat_context/acpa_lemma_4_folds_cv 4 
# fasttext supervised -input fasttext/data/train0.txt -output fasttext/model/acpa0 -lr 0.005 -epoch 50 -wordNgrams 3
# fasttext test fasttext/model/acpa0.bin fasttext/data/test0.txt 7
# fasttext predict fasttext/model/acpa0.bin fasttext/data/test0.txt

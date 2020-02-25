import os, csv
from taj.embedding.preprocess import word_tokenize

class Cluster:
    def __init__(self, clusterId):
        self._id = clusterId
        self.size = 1
        self.nbOcc = 0
        self.docId2nbOcc = {}

    def __addNbOcc__(self, nbOcc):
        self.nbOcc += nbOcc

    def addNbOccInDoc(self, docId, nbOcc):
        if not docId in self.docId2nbOcc.keys():
            self.docId2nbOcc[docId] = 0
        self.docId2nbOcc[docId] += nbOcc
        self.__addNbOcc__(nbOcc)
    def toString(self):
        docId2nbOccStr = ":".join([str(docId)+"-"+str(self.docId2nbOcc[docId]) for docId in self.docId2nbOcc.keys()])
        return self._id+"\t"+str(self.size)+"\t"+str(self.nbOcc)+"\t"+str(len(self.docId2nbOcc.keys()))+"\t"+docId2nbOccStr

def getwordClusterVocIndex(corpusDirPath, word2clusterId, vocIndexDirPath):
    if not os.path.exists(vocIndexDirPath):
        os.makedirs(vocIndexDirPath)
    chunIndex = {} # 2chars(clusterId) fileId nbWords
    fileOfIndex = vocIndexDirPath+"/1" # word(clusterId) nbToken(=1) nbOcc nbDocs docId1-nbOccIndDocId1:docId2-nbOccIndDocId2
    clusterVocIndex = {}
    docpath2index = {}
    index2docpath = {}
    nbDocs = 0
    for dirname, subdirname, filenames in os.walk(corpusDirPath):
        for fname in filenames:
            text = ""
            docpath = os.path.join(dirname, fname)
            idDoc = nbDocs
            docpath2index[docpath] = idDoc
            index2docpath[idDoc] = docpath
            with open(docpath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    text += line + " "
                #print(text)
                words = word_tokenize(text, lowercase=False, tokenizer="simple")
                for word in words:
                    if word in word2clusterId.keys():
                        if not word2clusterId[word] in clusterVocIndex.keys():
                            clusterVocIndex[word2clusterId[word]] = Cluster(word2clusterId[word])
                        clusterVocIndex[word2clusterId[word]].addNbOccInDoc(idDoc, 1)
            nbDocs+=1

    with open(fileOfIndex, "w") as fw:
        for clusterId in sorted(clusterVocIndex.keys()):
            print(clusterVocIndex[clusterId].toString())
            fw.write(clusterVocIndex[clusterId].toString()+"\n")
    with open(vocIndexDirPath+"/chunk_index.tsv", 'w') as fw:
        fw.write("#\t1\t"+str(len(clusterVocIndex.keys())))
    with open(vocIndexDirPath + "/corpus_index.tsv", 'w') as fw:
        for k,v in index2docpath.items():
            print(str(k), v)
            fw.write(str(k)+"\t"+v+"\n")
    with open(vocIndexDirPath + "/info", 'w') as fw:
        fw.write("NB_SCANNED_WORDS="+str(len(clusterVocIndex.keys()))+"\n")
        fw.write("SIZE_VOC="+str(len(clusterVocIndex.keys()))+"\n")
        fw.write("NB_VALIDATED_SCANNED_WORDS="+str(len(clusterVocIndex.keys()))+"\n")
        fw.write("NB_FILES="+str(len(index2docpath.keys()))+"\n")


def getWordClusterDico(word2clusterPath):
    word2clusterId = {}
    with open(word2clusterPath, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
        next(reader, None)  # skip the headers
        for row in reader:
            if len(row) < 2:
                continue
            word2clusterId[str(row[0]).lower()] = row[1]
    return word2clusterId

if __name__ == '__main__':
    corpusDirPath = "/home/tagny/Documents/current-work/sens-resultat/styx_lemma_4_folds_cv/0/wd_3gram_avec_context/train/styx-accepte"
    word2clusterPath = "/run/media/tagny/Dell-USB-Portable-HDD/D/current-work/word-cluster/frJudg800k.glove.300d.w15.100iter/word1000clusters.out.txt"
    word2clusterId = getWordClusterDico(word2clusterPath)
    vocIndexDirPath = "/home/tagny/Documents/current-work/sens-resultat/test-wordclust-vocidx/vocIndex_"+os.path.basename(corpusDirPath)
    getwordClusterVocIndex(corpusDirPath, word2clusterId, vocIndexDirPath)
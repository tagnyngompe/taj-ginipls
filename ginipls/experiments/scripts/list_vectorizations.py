# generate the content for vectorizations.txt


localWeights = ['TP', 'TF', 'LOGTF', 'ATF', 'LOGAVE', 'AVERAGELocals']
globalWeights = ['CHI2', 'DBIDF', 'DELTADF', 'DSIDF', 'GSS', 'IDF', 'IG', 'KLD', 'MARASCUILO', 'NGL', 'RF', 'AVERAGEGlobals']

for gw in globalWeights:
    for lw in localWeights:
        print(lw+gw)


#print("max*embbedings")
#print("mean*embbedings")

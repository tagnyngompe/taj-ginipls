localWeights = ['TP', 'TF', 'LOGTF', 'ATF', 'LOGAVE']
globalWeights = ['CHI2', 'DBIDF', 'DELTADF', 'DSIDF', 'GSS', 'IDF', 'IG', 'KLD', 'MARASCUILO', 'NGL', 'RF'] 

for gw in globalWeights:
    for lw in localWeights:
        print(gw+"*"+lw)


#print("max*embbedings")
#print("mean*embbedings")

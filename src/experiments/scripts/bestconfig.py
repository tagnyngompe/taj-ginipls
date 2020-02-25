import sys, getopt, time, os, csv

colCatDmd = 0
colVectorization=1
colClassifier=3
colMetric = 10

config2avgScore = {}
config2minScore = {}
config2maxScore = {}
config2worstCat = {}
config2bestCat = {}

plsExtensions = ['OurStandardPLS', 'OurGiniPLS', 'OurLogitPLS', 'OurGiniLogitPLS']

def ff(f):
    return '%.3f' % float(f)

if __name__ == "__main__":
    # config maximal
    #zonepath = "/run/media/tagny/Dell-USB-Portable-HDD/D/current-work/sens-resultat/cv-context"
    basepath = "/home/tagny/Documents/current-work/sens-resultat"
    metric = sys.argv[1] if len(sys.argv) > 1 else 'f1-macro-avg'
    selectedCatDmd = sys.argv[2] if len(sys.argv) > 2 else None
    #print(metric)
    #print(selectedCatDmd)
    #if zonepath.endswith('/'):
        #zonepath = zonepath[0:len(zonepath)-1]
    categoriesDmd = ['acpa', 'concdel', 'danais', 'dcppc', 'doris', 'styx']
    readHeader = True
    for zone in ["context", "demande_resultat_a_resultat_context", "motifs", "litige_motifs_dispositif"]:
        #print(zone)
        zonepath = basepath + "/cv-"+zone
        for catDmd in categoriesDmd:
            if not selectedCatDmd is None and catDmd!=selectedCatDmd:
                #print("skip", catDmd)
                continue
            csv_fname = zonepath+"/"+catDmd+"_lemma_4_folds_cv/resultat-wd-3_2-tsv_LDA/metrics_"+catDmd+"_4_folds_cv_wd-3_2.tsv"
            with open(csv_fname, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile, delimiter='\t')
                row = next(reader, None)  # skip the headers
                colMetric = row.index(metric)
                #print(metric, str(colMetric))
                for row in reader:                
                    config = zone+"\t"+row[colVectorization]+"\t"+row[colClassifier]
                    configScore = float(row[colMetric])
                    if not config in config2avgScore.keys():
                        config2avgScore[config] = configScore
                        config2minScore[config] = configScore
                        config2worstCat[config] = catDmd
                        config2maxScore[config] = configScore
                        config2bestCat[config] = catDmd
                    else:              
                        config2avgScore[config] += configScore
                        if config2minScore[config] > configScore:
                            config2minScore[config] = configScore
                            config2worstCat[config] = catDmd
                        if config2maxScore[config] < configScore:
                            config2maxScore[config] = configScore
                            config2bestCat[config] = catDmd                    
    #sort dict by values
    i = 1
    plsExtFound = set()
    #print(configScore)
    # average
    if selectedCatDmd is None:
        for config in config2avgScore.keys():
            config2avgScore[config] = config2avgScore[config]/len(categoriesDmd)
    
    
    bestScore = max(config2avgScore.values())
    #bestScore = 0    
    for config in sorted(config2avgScore, key=config2avgScore.get, reverse=True): 
        classifier = config.split("\t")[2]        
        if i == 1:
            bestScore = config2avgScore[config]
        if i == 1 or (classifier in plsExtensions and not classifier in plsExtFound):
            configScore = config2avgScore[config]
            line = config+" "+ff(configScore)
            if selectedCatDmd is None:
                line += " "+str(ff(config2minScore[config]))+" "+config2worstCat[config]+" "+ str(ff(config2maxScore[config]))+" "+config2bestCat[config]+" "+ff(bestScore - configScore)+" "+ff(config2maxScore[config] - config2minScore[config])
                line = "Average " + line
            else:
                line = selectedCatDmd +" "+ line            
            print(line, str(i))
        if classifier in plsExtensions:
            plsExtFound.add(classifier)
            if len(plsExtFound) == len(plsExtensions):
                break
        i+=1
    #print()

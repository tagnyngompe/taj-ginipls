# encoding: utf-8
import traceback

import numpy as np
import pandas as pd
import math
from scipy.stats import rankdata ## equiv rankindx de GAUSS
from sklearn import linear_model
from sklearn.metrics import f1_score, accuracy_score
from ginipls.config import GLOBAL_LOGGER as logger

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
      expr -- input expression in which the error occurred
      msg  -- explanation of the error
    """
    def __init__(self, expr, msg):
        self.expr = expr
        self.msg = msg


from enum import Enum
class PLS_VARIANT(Enum):
    STANDARD = 1
    GINI = 2
    LOGIT = 3
    LOGIT_GINI=4

class PLS():
    def __init__(self, pls_type=PLS_VARIANT.STANDARD, n_components=10,
                 nu=2, centering_reduce=True, centering_reduce_rank=True,
                 use_VIP=False):
        """
        Parameters:
        ----------
        n_components: int
            nombres de composantes
        use_VIP: boolean
             elimination des variables
        """
        # print("\n", pls_type)
        # print('Paramètres: n_components=',n_components, 'nu=', nu)
        if not isinstance(pls_type, PLS_VARIANT):
            raise InputError(exp=pls_type, msg="the input PLS type is not implemented. Please use an instance of PLS_VARIANT Enum class (e.g. PLS_VARIANT.STANDARD)")
        self.pls_type = pls_type
        self.n_components = n_components
        self.w = None
        self.c = None
        self.nu = nu
        self.centering_reduce = centering_reduce # pour les données
        self.X_train_means = None
        self.X_train_stdevs = None
        self.y_train_mean = None
        self.y_train_stdev = None
        self.seuil_y_train_nvl_espace= None # seuil utiliser pour la classification
        self.X_train = None
        self.y_train = None
        self.y_train_cr = None # y_train centrée réduite
        self.k = 0 # nombre de colonne
        self.n_train = 0 # nombre d'exemples
        self.n0_train = 0 # nombre d'exemples y = 0
        self.n1_train = 0 # nombre d'exemples y = 1
        self.rank_train = None
        self.T = None
        self._constant_cols = []
        self.cols_to_keep = [] # significant variables
        self.centering_reduce_rank = centering_reduce_rank
        self.use_VIP = use_VIP

    def __identify_constant_columns(self, X_train):
        """
        Identify constant columns in the trainset pandas.dataframe (matrix)

        Parameters
        ----------
        data : numpy.matrix
        """
        self._constant_cols = []
        for i in range(X_train.shape[1]):
            vc=np.var(X_train[:,i])
            if vc == 0:
                self._constant_cols = self._constant_cols + [i]

    def __remove_columns(self, X, idx_cols_to_remove):
        """
        Remove columns identified in self._constant_cols
        """
        # Supression des colonnes de constantes
        return np.asmatrix(pd.DataFrame(X).drop(idx_cols_to_remove, 1))
    @staticmethod
    def __encode_y__(y):
        """Convert the 2 labels used in y (the expected output) into {0,1}"""
        labels = list(set(y))
        sorted(labels)
        #logger.debug("labels codes : %s" % str({labels[i]:i for i in range(len(labels))}))
        return np.asarray([labels.index(i) for i in y])

    def __init_variables(self, X_train, y_train):
        """
        Initialisation des variables :  k, n, y
        Identify constant columns in the trainset pandas.dataframe (matrix)
        compute the mean and stdev of X_train and y_train

        Parameters
        ----------

        Return
        ------
        X: np.matrix
          matrice de features
        y: np.array
          valeurs de la variable dépendante train
        k: int
          nombre de regresseurs (features)  =  colonnes de la matrice -1
          nombre de colonnes de la nouvelle matrice après suppression des colonnes
        n: int
          nombre d'instances de train (lignes)
        n1: int
          nombre de oui train (valeur 1)
        n2: int
          nombre de non train (valeur 0)
        """
        X = np.asmatrix(X_train)
        y = PLS.__encode_y__(y_train)
        n2 = 0 # nombre de oui train (valeur 1)
        n1 = 0 # nombre de non train (valeur 0)
        #print("ginipls.PLS.__init_variables y_train ", y_train)
        # convert labels to {0,1}
        y = np.transpose(np.asmatrix(y))
        #print("ginipls.PLS.__init_variables y ", y)
        for vc in y :
            if vc == 0:
                n1=n1+1
            else:
                n2=n2+1
        self.__identify_constant_columns(X)
        X = self.__remove_columns(X, self._constant_cols)
        n = X.shape[0] # nb rows
        k = X.shape[1] # nb cols
        self.X_train_means = self.__cols_mean(X)
        self.X_train_stdevs = self.__cols_stdev(X)
        self.y_train_mean = self.__cols_mean(y)
        self.y_train_stdev = self.__cols_stdev(y)
        self.cols_to_keep = [i for i in range(k)]
        self.n0_train = n1
        self.n1_train = n2
        return X, y, k, n, n1, n2

    def __cols_mean(self, x):
        return [np.mean(x[:,i]) for i in range(x.shape[1])]

    def __cols_stdev(self, x):
        return [math.sqrt((x.shape[0]/(x.shape[0]-1))*np.var(x[:,i])) for i in range(x.shape[1])]

    def __centre_reduit(self, x, means, stdevs):
        """
        centrer la variable aléatoire X: soustraire à chaque valeurs de X son espérance E[X]
        réduire X: diviser chaque valeur de X par l'écart-type de X
        """
        xc=np.asmatrix(np.zeros((x.shape[0],x.shape[1])))
        for i in range(x.shape[1]):
            #xc[:,i] = np.divide(x[:,i] - np.mean(x[:,i]),math.sqrt((n/(n-1))*np.var(x[:,i])))
            xc[:,i] = np.divide(x[:,i] - means[i], stdevs[i])
        return xc

    def __construit_poids_initiaux_w(self, x, y):
        """
        Construction des poids W

        Parameters
        ----------
        x: np.matrix
            matrice des features
        y: np.matrix
            matrice colonne de la var dépendantes
        Return
        ------
        w: scalar
            poids du ginipls (covariance entre y et les features)
        v:
            Matrice rangs  de X
        """
        w = np.zeros((x.shape[1], self.n_components))
        init = np.zeros((x.shape[1],1))
        rank = None
        if self.pls_type == PLS_VARIANT.STANDARD:
            for i in range(x.shape[1]):
                init[i,:] = np.cov(x[:,i].T, y[:,0].T)[0][1]
            init /= np.sqrt(np.sum(np.square(init)))
            w[:,0] = init[:, 0]
        elif self.pls_type == PLS_VARIANT.LOGIT:
            for i in range(x.shape[1]):
                lm = linear_model.LogisticRegression(solver="lbfgs")
                lm.fit(x[:,i], self.y_train) # y_train doit être le vecteur y de départ avec des 0 et 1
                init[i,:] = lm.coef_
            init /= np.sqrt(np.sum(np.square(init)))
            w[:,0] = init[:, 0]
        elif self.pls_type == PLS_VARIANT.LOGIT_GINI:
            rank = np.asmatrix(np.zeros((x.shape[0], x.shape[1])))
            for i in range(x.shape[1]):
                lm = linear_model.LogisticRegression(solver="lbfgs")
                rank[:,i] = np.asmatrix((x.shape[0]+1-rankdata(x[:,i], method='average'))**(self.nu-1)).T
                lm.fit(rank[:,i], self.y_train) # y_train doit être le vecteur y de départ avec des 0 et 1
                init[i,:] = lm.coef_
            init /= np.sqrt(np.sum(np.square(init)))
            w[:,0] = init[:, 0]
        elif self.pls_type == PLS_VARIANT.GINI:
            rank = np.asmatrix(np.zeros((x.shape[0], x.shape[1])))
            for i in range(x.shape[1]):
                rank[:,i] = np.asmatrix((x.shape[0]+1-rankdata(x[:,i], method='average'))**(self.nu-1)).T
                init[i,:] = -self.nu*(np.cov(rank[:,i].T, y[:,0].T)[0][1])
            sum_init = np.sum(np.abs(init))
            if sum_init != 0:
                init /= sum_init
            w[:, 0] = init[:, 0]
        return np.asmatrix(w), rank

    def __calcule_composantes_principales(self, x, y, w, rank):
        """
        Calcul des composantes principales t_j

        Parameters
        ----------
        x: np.matrix
            matrice des features
        y: np.matrix
            matrice colonne de la var dépendantes
        w:  np.matrix
            matrice colonne des poids (construite par self.construction_poids_w)
        centering_reduce_rank: centrer et réduire les rank (valide uniquement pour la variante gini)
        Return
        ------
        """
        k = x.shape[1]
        n = x.shape[0]
        #/***********************************************************************/
        #/*      Création des matrices pour les résultats du Gini-PLS            */
        #/***********************************************************************/
        c=np.asmatrix(np.zeros((self.n_components+1,2))) # nombre de paramètres estimés GPLS ?? à quoi correspond le 2 (?? # classes ??)
        Q=np.asmatrix(np.zeros((self.n_components,1))) # Les stat Q pour la validation croisée GPLS
        pc=np.asmatrix(np.zeros((k,self.n_components)))
        #/***********************************************************************/
        #/*     Matrice des compsoantes principales t_j et matrices d'info RSS  */
        #/***********************************************************************/
        xr = np.asmatrix(np.copy(x)) # residu de x
        T=np.matrix(np.zeros((n,self.n_components)))
        eps=np.matrix(np.zeros((n,self.n_components)))
        eps=np.asmatrix(eps)
        RSS = np.matrix(np.zeros((self.n_components,1)))    #/*** vecteur des RSS ***//
        RSS0 = np.asscalar((1 / (n-1)) * np.sum(np.square(y), axis=0))  # RSS t1
        #print('RSS0',RSS0)
        #/***********************************************************************/
        #/*              Calcul des compsoantes principales t_j                 */
        #/***********************************************************************/
        for j in range(self.n_components):
            #print('j',j)
            j1=j+1
            RES=np.asmatrix(np.zeros((n,k)))
            T[:,j] = np.dot(xr, w[:,j])
            Tj = np.matrix(np.zeros((n,j1)))
            Tj = T[:,0:j1]
            Tjp = np.transpose(Tj)
            c[0:j1,0] = np.dot(np.linalg.pinv(np.dot(Tjp,Tj)), np.dot(Tjp, y)) # utilisation de la pseudo inverse pour gérer les cas de matrices singulières
            eps[:,j] = y - np.dot(T[:,0:j1],c[0:j1,0])
            RSS[j,:] = np.sum(np.square(eps[:,j]), axis=0)    #  /* SCR */
            Q1 = np.sum(np.square(np.dot((T[:,0:j1]),c[0:j1,0])), axis=0)
            Q2 = np.asscalar(RSS[j,:])
            Q3 = Q1+Q2
            c[j,1] = Q1 / Q3 #/* R carré du modèle Gini-PLS : c[j,2] = Q1/Q3 = (Q3'Q3)^-1Q3'Q1*/
            # Régressions partielles + poids prochaine étape
            #v = np.zeros((x.shape[1], n_components))
            if self.pls_type == PLS_VARIANT.STANDARD:
                init = np.asmatrix(np.zeros((xr.shape[1], 1)))
                for i in range(k):
                    b = np.dot(np.linalg.pinv(np.dot(Tjp,Tj)), np.dot(Tjp,x[:,i]))
                    RES[:,i] = x[:,i]-np.dot(Tj, b) # Les résidus des régressions partielles sont mis en colonne
                    pc[i,0:j1] = np.transpose(b)
                    init[i,:] = np.cov(RES[:,i].T, eps[:,j].T)[0][1]
                    #print('init[i,:]', init[i,:])
                init /= np.sqrt(np.sum(np.square(init)))
            elif self.pls_type == PLS_VARIANT.LOGIT:
                #poids = np.asmatrix(np.zeros((xr.shape[1],1)))
                init = np.asmatrix(np.zeros((xr.shape[1],2)))
                for i in range(k):
                    b = np.dot(np.linalg.pinv(np.dot(Tjp,Tj)), np.dot(Tjp,x[:,i]))
                    RES[:,i] = x[:,i]-np.dot(Tj, b) #
                    pc[i,0:j1] = np.transpose(b)
                    concat = np.concatenate((RES[:,i], Tj), axis=1)
                    #print("y", y.T.tolist())
                    lm = linear_model.LogisticRegression(solver="lbfgs")
                    #print("y_train", y_train)
                    lm.fit(concat, self.y_train) # y_train doit être le vecteur y de départ avec des 0 et 1
                    #print("init",init)
                    #print("init[i,:]",init[i,:].shape)
                    #print("lm.coef_", lm.coef_)
                    init[i,:] = lm.coef_[:,0]# lm.coef_[0,:] le nombre de coefficient de la regression logistic peut varier (2 ou 3)
                init /= np.sqrt(np.sum(np.square(init)))
            elif self.pls_type == PLS_VARIANT.LOGIT_GINI:
                #poids = np.asmatrix(np.zeros((xr.shape[1],1)))
                init = np.asmatrix(np.zeros((xr.shape[1],2)))
                for i in range(k):
                    if self.centering_reduce_rank:
                        rank[:,i] = rank[:,i]-np.mean(rank[:,i])
                    #b = np.dot(np.linalg.pinv(np.dot(Tjp,Tj)), np.dot(Tjp,x[:,i]))
                    b = np.dot(np.linalg.pinv(np.dot(Tjp,Tj)), np.dot(Tjp,rank[:,i]))
                    RES[:,i] = rank[:,i]-np.dot(Tj, b) #
                    pc[i,0:j1] = np.transpose(b)
                    concat = np.concatenate((RES[:,i], Tj), axis=1)
                    #print("y", y.T.tolist())
                    lm = linear_model.LogisticRegression(solver="lbfgs")
                    #print("y_train", y_train)
                    lm.fit(concat, self.y_train) # y_train doit être le vecteur y de départ avec des 0 et 1
                    #print("init",init)
                    #print("init[i,:]",init[i,:].shape)
                    #print("__calcule_composantes_principales lm.coef_[:,0]", lm.coef_[:,0])
                    try:
                        init[i,:] = lm.coef_[0,:init[i,:].shape[1]]# lm.coef_[0,:] le nombre de coefficient de la regression logistic peut varier (2 ou 3)
                    except Exception:
                        tb = traceback.format_exc()
                        print(tb)
                        print("__calcule_composantes_principales lm.coef_[0,:init[i,:].shape[1]]",
                              lm.coef_[0, :init[i, :].shape[1]])
                #print("__calcule_composantes_principales init =", init)
                init /= np.sqrt(np.sum(np.square(init)))
                #print("__calcule_composantes_principales init /= np.sqrt(np.sum(np.square(init))) =", init)
            elif self.pls_type == PLS_VARIANT.GINI:
                init = np.asmatrix(np.zeros((xr.shape[1], 1)))
                v = np.asmatrix(np.zeros((xr.shape[0], k)))
                for i in range(k):
                    if self.centering_reduce_rank:
                        rank[:,i] = rank[:,i]-np.mean(rank[:,i])
                    b = np.dot(np.linalg.pinv(np.dot(Tjp,Tj)), np.dot(Tjp,rank[:,i]))
                    RES[:,i] = rank[:,i]-np.dot(Tj, b) # Les résidus des régressions partielles sont mis en colonne
                    pc[i,0:j1] = np.transpose(b)
                    v[:,i] = np.asmatrix((xr.shape[0]+1-rankdata(RES[:,i], method='average'))**(self.nu-1)).T
                    init[i,:] = -self.nu*(np.cov(v[:,i].T, eps[:,j].T))[0][1]
                    #print('init[i,:]', init[i,:])
                sum_init = np.sum(np.abs(init))
                if sum_init != 0:
                    init /= sum_init
            if j < self.n_components-1:
                w[:,j+1] = init[:, 0]
            #print('w',w)
            # Nouvelles matrices x pour la prochaine étape
            #xresidu=np.asmatrix(np.zeros((n,k)))
            xr=RES
            # Validation Croisée
            PRESS=np.asmatrix(np.zeros((n,1)))
            for i in range(n):
                Tv = np.delete(T, i, 0) # Suppression de la ligne i de T
                yv = np.delete(y, i, 0)
                Tvj = Tv[:,0:j1]
                Tvjp = np.transpose(Tvj)
                b = np.dot(np.linalg.pinv(np.dot(Tvjp,Tvj)), np.dot(Tvjp,yv))
                PRESS[i,:]=np.square(y[i] - np.dot(T[i,0:j1], b))
            if j==0:
                Q[j,:]=1- np.asscalar(np.sum(PRESS, axis=0)/(RSS0*(n-1)))
            else:
                Q[j,:]=1 - np.asscalar(np.sum(PRESS, axis=0)/RSS[j,:])
        self.w = w
        self.c = c
        self.T = T
        # calcul du seuil du y dans le nouvel espace pour la classification
        yh = np.asmatrix(np.zeros((n,1)))
        for i in range(self.n_components):
            yh = yh + self.c[i,0] * self.T[:,i]
        self.seuil_y_train_nvl_espace = np.mean(yh, axis=0)

    #    def __vip0(self):
    #        """
    #        elimination des variables non-significatives VIP
    #        """
    #        k = len(self.cols_to_keep) # nb features
    #        n = self.n_train # nb exemples
    #        M3 = np.asmatrix(np.zeros((1,self.n_components)))
    #        y = self.y_train_cr
    #        for h in range(self.n_components):
    #            Th = self.T[:,h]
    #            Thp = np.transpose(Th)
    #            b = np.asscalar(np.dot(np.linalg.pinv(np.dot(Thp,Th)), np.dot(Thp,y)))
    #            yh2 = np.square(b * self.T[:,h]) / np.sum(np.square(y),axis=0)
    #            M3[:,h] = np.sum(yh2[0:n,:],axis=0)
    #        VIP=np.asmatrix(np.zeros((k,self.n_components)))
    #        M2 = np.asmatrix(np.zeros((1,self.n_components)))
    #        for s in range(self.n_components):
    #            cx = np.corrcoef(self.T[:,s], y, 0) # corrélation linéaire (pearson) entre T[.,s] et y
    #            M2[:,s] = np.square(cx[0,1])
    #            RdY = np.sum(M2, axis=1) / k
    #            for i in range(k):
    #                RdYY = np.asmatrix(np.zeros((1,self.n_components)))
    #                for l in range(self.n_components):
    #                    RdYY[0,l] = np.sum(M3[:,l], axis=0) * np.square(self.w[i,l])
    #                VIP[i,s]=(1 / RdY) * np.sum(RdYY[0,0:(s+1)], axis=1)
    #        return VIP

    def __vip(self):
        """ VIP : elimination des variables GPLS """
        p = self.n_components
        M30 = np.asmatrix(np.zeros((1,p)))
        M31 = np.asmatrix(np.zeros((1,p)))
        y = self.y_train_cr
        T = self.T
        n1 = self.n0_train
        #n2 = self.n1_train
        n = self.n_train
        k = len(self.cols_to_keep)
        h=0
        while h < p :
            #A=np.asmatrix(np.zeros((n,2))) # A n'est pas utilisé dans la boucle => oui bien vu !
            Th = T[:,h]
            Thp = np.transpose(Th)
            b = np.asscalar(np.dot(np.linalg.pinv(np.dot(Thp,Th)), np.dot(Thp,y)))
            yh2 = np.square(b * T[:,h]) / np.sum(np.square(y),axis=0)
            M30[:,h] = np.sum(yh2[0:n1,:],axis=0)
            M31[:,h] = np.sum(yh2[n1:n,:],axis=0)
            h+=1
        #import scipy
        VIP=np.asmatrix(np.zeros((k,p)))
        VIP0=np.asmatrix(np.zeros((k,p)))
        VIP1=np.asmatrix(np.zeros((k,p)))
        M2 = np.asmatrix(np.zeros((1,p)))
        s=0
        while s < p: # un peu long
            #cx = np.asmatrix(scipy.stats.pearsonr(T[:,s], y))
            cx = np.corrcoef(T[:,s], y, 0) # corrélation linéaire (pearson) entre T[.,s] et y
            M2[:,s] = np.square(cx[0,1])
            RdY = np.sum(M2, axis=1) / k
            i=0
            while i < k:
                RdYY1 = np.asmatrix(np.zeros((1,p)))
                RdYY0 = np.asmatrix(np.zeros((1,p)))
                l=0
                while l < p:
                    RdYY0[0,l] = np.sum(M30[:,l], axis=0) * np.square(self.w[i,l])
                    RdYY1[0,l] = np.sum(M31[:,l], axis=0) * np.square(self.w[i,l])
                    l+=1
                VIP0[i,s]=(1 / RdY) * np.sum(RdYY0[0,0:(s+1)], axis=1)
                VIP1[i,s]=(1 / RdY) * np.sum(RdYY1[0,0:(s+1)], axis=1)
                i+=1
            s+=1
        VIP=VIP0+VIP1
        verif = np.sum(VIP0+VIP1, axis=0)
        # verif : si la somme des stat VIP = k, code ok
        #print("verif", verif, "k", k)
        return VIP

    def __get_vip_based_significative_cols(self, vip_threshold=1):
        """retourne les colonnes de X_train à supprimer"""
        VIP = self.__vip()
        sum_VIP = np.sum(VIP, axis=1)
        sum_VIP = [sum_VIP[i, 0] for i in range(sum_VIP.shape[0])]
        #print("sum_VIP", len(sum_VIP))
        #k = len(self.cols_to_keep)
        #print("np.sum(VIP, axis=0)", np.sum(VIP, axis=0),  "# k=",k)
        new_cols_to_keep = []
        for j in range(len(self.cols_to_keep)):
            if sum_VIP[j] >= vip_threshold:
                new_cols_to_keep += [self.cols_to_keep[j]]
        return new_cols_to_keep

    def fit(self, X_train, y_train):
        """
        Gini-PLS/PLS training

        Parameters:
        x: np.matrix or list(), not ndarray
            matrice des features
        y: np.matrix or list(), not ndarray
            matrice colonne de la var dépendantes
        """
        #print("fiting...")
        #/***********************************************************************/
        #/*            Création des matrices X et y : partie train              */
        #/***********************************************************************/
        X, y, k, n, n1, n2 = self.__init_variables(X_train, y_train)
        self.k = k
        self.n_train = n
        self.y_train = y_train
        #/***********************************************************************/
        #/*                      Centrage - Réduction                            */
        #/***********************************************************************/
        if self.centering_reduce:
            X = self.__centre_reduit(X, self.X_train_means, self.X_train_stdevs)
            y = self.__centre_reduit(y, self.y_train_mean, self.y_train_stdev)
        self.y_train_cr = y
        self.X_train = np.asmatrix(np.copy(X))
        while True:  ## ajout de la VIP
            #/***********************************************************************/
            #/*                    Construction des poids W                         */
            #/***********************************************************************/
            w, self.rank_train = self.__construit_poids_initiaux_w(X, y)
            #print("w",w.shape)
            #print("w",w)
            #print("self.rank_train",self.rank_train)
            #/***********************************************************************/
            #/*              Calcul des compsoantes principales t_j                 */
            #/***********************************************************************/
            self.__calcule_composantes_principales(X, y, w, self.rank_train)
            if self.use_VIP == True:
                #print("self.cols_to_keep", self.cols_to_keep)
                new_cols_to_keep = self.__get_vip_based_significative_cols()
                cols_to_delete = [j for j in range(len(self.cols_to_keep)) if self.cols_to_keep[j] not in new_cols_to_keep]
                #print('new_cols_to_keep', new_cols_to_keep)
                #print("cols_to_delete", cols_to_delete)
                if len(new_cols_to_keep) < self.n_components or len(new_cols_to_keep) == len(self.cols_to_keep):
                    break
                X = self.__remove_columns(X, cols_to_delete)
                self.X_train = self.__remove_columns(self.X_train, cols_to_delete)
                #print('new X.cols', X.shape[1])
                self.cols_to_keep = new_cols_to_keep
            else:
                break

    def predict(self, X_test):
        """
        prédit les classes des lignes de X (variable dépendante)

        Parameters:
        -----------
        X_test: matrix
            Matrice dont les lignes sont des vecteurs d'instances

        Return
        ------
        y_pred:
            liste des classes prédites pour chaque instance/ligne de X
        """
        X_test = np.asmatrix(X_test)
        TT = self.transform(X_test)
        #print("TT.shape", TT.shape)
        nt = X_test.shape[0] # nb instances test
        y_pred = np.asmatrix(np.zeros((nt,1)))
        yh = np.asmatrix(np.zeros((nt,1)))
        for i in range(self.n_components):
            yh = yh + self.c[i,0] * TT[:,i]
        for i in range(nt):
            if yh[i] >= self.seuil_y_train_nvl_espace : # ??? à généraliser pour les cas où on a plus de 2 classes
                y_pred[i] = 1
            else :
                y_pred[i] = 0
        return [int(y) for y in np.asarray(y_pred.T)[0,:]]

    def transform(self, X_test):
        """
        Projecte X dans le nouvel espace réduit

        Parameters:
        -----------
        X_test: matrix
            Matrice dont les lignes sont des vecteurs d'instances

        Return
        ------
        TT: matrix
            Matrice représentant la projection des lignes de X dans le nouvel espace construit par fit()
        """
        X_test = self.__remove_columns(X_test,self._constant_cols)
        if self.centering_reduce:
            X_test = self.__centre_reduit(X_test, self.X_train_means, self.X_train_stdevs)
        idx_cols = [i for i in range(X_test.shape[1])]
        # supprimer ls variable non significatives de la base de test
        cols_to_delete = [col_id for col_id in idx_cols if col_id not in self.cols_to_keep]
        X_test = self.__remove_columns(X_test, cols_to_delete)
        #print('X_test', X_test)
        n = self.X_train.shape[0] # nb lignes train
        nt = X_test.shape[0] # nb instances/lignes test
        k =  X_test.shape[1] # nb features
        TT = np.asmatrix(np.zeros((nt,self.n_components)))
        TT[:,0] = X_test * np.asmatrix(self.w[:,0])
        if self.pls_type == PLS_VARIANT.STANDARD:
            for j in range(self.n_components-1) :
                #print('j',j)
                RES = np.asmatrix(np.zeros((nt,k)))
                Tj1 = self.T[:,0:(j+1)]
                Tj1p = np.transpose(Tj1)
                for i in range(k) :
                    #print('Tj1p', Tj1p.shape)
                    #print('Tj1', Tj1.shape)
                    #print('self.X_train[:,i]', self.X_train[:,i].shape)
                    BETA = np.dot(np.linalg.pinv(np.dot(Tj1p,Tj1)), np.dot(Tj1p,self.X_train[:,i]))
                    RES[:,i] = X_test[:,i] - TT[:,0:(j+1)]*BETA
                TT[:,(j+1)] = RES * self.w[:,(j+1)]
        elif self.pls_type == PLS_VARIANT.LOGIT:
            for j in range(self.n_components-1) :
                #print('j',j)
                RES = np.asmatrix(np.zeros((nt,k)))
                Tj1 = self.T[:,0:(j+1)]
                Tj1p = np.transpose(Tj1)
                for i in range(k) :
                    # print('Tj1p', Tj1p.shape)
                    # print('Tj1', Tj1.shape)
                    # print('self.X_train[:,i]', self.X_train[:,i].shape)
                    BETA = np.dot(np.linalg.pinv(np.dot(Tj1p,Tj1)), np.dot(Tj1p,self.X_train[:,i]))
                    RES[:,i] = X_test[:,i] - TT[:,0:(j+1)]*BETA
                TT[:,(j+1)] = RES * self.w[:,(j+1)]
        elif self.pls_type == PLS_VARIANT.GINI or self.pls_type == PLS_VARIANT.LOGIT_GINI:
            Rang = np.asmatrix(np.zeros((nt,k)))
            for i in range(nt):
                rank2 = np.asmatrix(np.zeros(((n+1),k)))
                #print('X_test', X_test.shape)
                #print('self.X_train', self.X_train.shape)
                fact = np.concatenate((X_test[i,:],self.X_train), axis=0)
                for j in range(k):
                    r = np.matrix(rankdata(fact[:,j],method='average')).T
                    rank2[:,j] = np.matrix(np.ones(((n+1),1))) * (n+2) - r
                    rank2[:,j] = np.power(rank2[:,j], self.nu-1)
                    if self.centering_reduce_rank:
                        rank2[:,j] = rank2[:,j]-np.mean(rank2[:,j])
                Rang[i,:] = rank2[0,:]
            for j in range(self.n_components-1):
                RES = np.asmatrix(np.zeros((nt,k)))
                Tj = self.T[:,0:(j+1)]
                Tjp = np.transpose(Tj)
                for i in range(k) :
                    b1 = np.dot(np.linalg.pinv(np.dot(Tjp,Tj)), np.dot(Tjp, self.rank_train[:,i]))
                    RES[:,i] = Rang[:,i] - TT[:,0:(j+1)]*b1
                TT[:,(j+1)] = RES*self.w[:,(j+1)]
        return TT
    #/*****************************************************************************************************************************************/
    #/*                                      For compatibility with sklearn GridSearch.                                                       */
    #/*https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV.score */
    #/*****************************************************************************************************************************************/
    def get_params(self, deep=True):
        """ Returns a dict of the attribute of the model with their name as key and their value as value.
        """
        return self.__dict__
    def set_params(self, **parameters):
        """ Set the value of some attribute from a dict. parameters.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    def score(self, X, y):
        """ predict the labels ypred for X and score the prediction with regard to the expected labels y
        """
        y = PLS.__encode_y__(y)
        ypred = self.predict(X)
        #print("ypred", ypred)
        #print("y", y)
        return f1_score(y, ypred, labels=[0,1], average='macro')

if __name__ == "__main__":
    cls = PLS(pls_type = PLS_VARIANT.GINI)
    params_ginipls = {'nu':1.1}
    cls.set_params(**params_ginipls)
    print(cls.get_params())

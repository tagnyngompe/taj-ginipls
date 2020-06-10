from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from src.models.ginipls import PLS, PLS_VARIANT
iris = datasets.load_iris()
nu_min = 1.3
nu_max = 3
nu_step = 0.1
parameters = {'nu':[i*nu_step for i in range(int(nu_min/nu_step),int(nu_max/nu_step))]}
print(parameters)
ginipls = PLS(pls_type = PLS_VARIANT.GINI)
clf = GridSearchCV(ginipls, parameters)
clf.fit(iris.data, iris.target)
print(clf.cv_results_)
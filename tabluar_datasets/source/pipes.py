#Data Handling
import numpy as np

#Pipelines
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack
from scipy import sparse
from scipy.sparse import csr_matrix
import scipy
from sklearn.model_selection import GridSearchCV

#Transformation
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

#Imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer

#Decomposition
from sklearn.decomposition import TruncatedSVD

#Storing estimators
import pickle

def applypreprocessing(X, nompipe, numpipe, idnumerical = None, idnominal = None):
    if idnumerical==None:
        idnumerical = [i for i, e in enumerate(X.dtypes) if e == 'float64']
    if idnominal==None:
        idnominal = list(np.setdiff1d(list(range(0,len(X.columns))),idnumerical))
    
    numerical = list(X.iloc[:,idnumerical].columns)
    nominal = list(X.iloc[:,idnominal].columns)
    
    #Identifying numerical and nominal variables
    X_nom = X.loc[:,nominal]
    
    #Numerical
    X_num = X.loc[:,numerical]
    
    #Apply trained pipes
    if nompipe==None:
        X_num = numpipe.transform(X_num)
        X_final = X_num
    elif numpipe==None:
        X_nom = nompipe.transform(X_nom)
        if not(scipy.sparse.issparse(X_nom)):
            X_nom = sparse.csr_matrix(X_nom)
        X_final = X_nom
    else:
        X_nom = nompipe.transform(X_nom)
        if not(scipy.sparse.issparse(X_nom)):
            X_nom = sparse.csr_matrix(X_nom)
        X_num = numpipe.transform(X_num)
        X_final = hstack((X_num, X_nom))   
    
    return X_final

def preprocessing(X, idnumerical=None, idnominal=None, imputation=True, encode = 'one-hot', normalization = True ):
    #Return a sparse matrix using X as a train dataset for fitting estimators.
    #Additionally it is returned the fitted pipelines related to numerical and nominal features.
    
    #Identifying numerical and nominal variables
    if idnumerical==None:
        idnumerical = [i for i, e in enumerate(X.dtypes) if e == 'float64']
    if idnominal==None:
        idnominal = list(np.setdiff1d(list(range(0,len(X.columns))),idnumerical))
        
    numerical = list(X.iloc[:,idnumerical].columns)
    nominal = list(X.iloc[:,idnominal].columns)
    nom_num = [y for x in [numerical, nominal] for y in x] 
    
    X_nom = X.loc[:,nominal]
    X_num = X.loc[:,numerical]
    

    #Applying estimators for nominal an numerical
    #nominal
    pipe_nom = None
    if len(nominal)>0:
        estimators = []
        imp_nom = SimpleImputer(strategy='most_frequent')
        scale = StandardScaler(with_mean=True)
        if encode == 'one-hot':
            enc = OneHotEncoder(drop='first')
        elif encode == 'label':
            enc = OrdinalEncoder()

        if imputation == True:
            estimators.append(('imputation', imp_nom))
        if encode != None:
            estimators.append(('encoding', enc))
        if normalization:
            estimators.append(('standardscale', scale))

        pipe_nom = Pipeline(estimators)
        pipe_nom.fit(X_nom)
    
    #numerical
    pipe_num = None
    if len(numerical)>0:
        estimators = []
        imp_num = IterativeImputer(max_iter=100, random_state=1)
        scale = StandardScaler(with_mean=True)

        if imputation == True:
            estimators.append(('imputation', imp_num))

        if normalization:
            estimators.append(('standardscale', scale))

        pipe_num = Pipeline(estimators)
        pipe_num.fit(X_num)
        
    #Merge both transformations
    if len(nominal)<1:
        X_num = pipe_num.transform(X_num)
        X_final = X_num
    elif len(numerical)<1:
        X_nom = pipe_nom.transform(X_nom)
        if not(scipy.sparse.issparse(X_nom)):
            X_nom = sparse.csr_matrix(X_nom)
        X_final = X_nom
    else:
        X_nom = pipe_nom.transform(X_nom)
        if not(scipy.sparse.issparse(X_nom)):
            X_nom = sparse.csr_matrix(X_nom)
        X_num = pipe_num.transform(X_num)
        X_final = hstack((X_num, X_nom))    
    
    hot_encoder = pipe_nom['encoding']
    if encode == 'one-hot':
        nom_features = hot_encoder.get_feature_names(nominal)
    else:
        nom_features = nominal
        
    return X_final, pipe_nom, pipe_num, numerical, nom_features

def import_pickle(directory):
    with open(directory, 'rb') as f:
        p = pickle.load(f)

    return p


def get_grid(X, y, parameters, model, model_name, scoring=['f1'], refit = 'f1'):
    pipe_model_train = Pipeline([(model_name, model)])
    
    grid = GridSearchCV(pipe_model_train,param_grid=parameters, cv=5, scoring = scoring, refit=refit) 
    #alternatives: 'accuracy', 'roc_auc' and 'f1' (the las two only accept binary variables)
    
    fit = grid.fit(X,y)
    
    return fit
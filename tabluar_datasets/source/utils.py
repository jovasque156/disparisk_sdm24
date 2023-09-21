import os
import random
import math
import numpy as np
import pickle
from itertools import product

#fairness metrics
from source.fairness import fpr, recall, selection_rate

#Pipelines
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack
from scipy import sparse
import scipy

#Transformation
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, Normalizer

# Dimensionality reduction
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import get_scorer

#Imputation
#Do not remove enable_iterative_imputer, it is needed to import IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

DIR_DATA = {
    'dutch_census': 'data/dutch_census/',
    'census_income':'data/census_income/',
    'compas': 'data/compas/',
    }

import ipdb

def get_datasets(args):
    '''
    Load the datasets for all scenarios

    Returns
    -------
    sets : dict
        Dictionary containing the datasets for all scenarios
    '''
    sets = {}

    for data in args.datasets:
        with open (DIR_DATA[data]+data+'.pkl', 'rb') as f:
            dic = pickle.load(f)
        sets[data] = dic
    
    return sets

def apply_preprocessing(X, nompipe, numpipe, pipe_normalize, idnumerical = None, idnominal = None):
    '''
    Apply transformer pipelines to X and returned the transformer dataset.

    Inputs:
    X: pandas (n,m), representing the dataset with n samples and m features to transform.
    nompipe: pipeline, representing the pipeline with transformer to apply to nominal features in X.
    numpipe: pipeline, representing the pipeline with transformer to apply to numerical features in X.
    pipe_normalize: pipeline, representing the pipeline with transformer to apply to normalize the features in X.
    idnumerical: numpy, representing the id of numerical features in X to transform using numpipe.
    idnominal: numpy, representing the id of nominal features in X to transform using nompipe.

    Outputs:
    X_final: csr_matrix, sparse matrix with the transformed X. 
    '''
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
    
    if pipe_normalize!=None:
        X_final = pipe_normalize.transform(X_final)

    return X_final

def train_preprocessing(X, idnumerical=None, idnominal=None, imputation=True, encode = 'one-hot', standardscale=True, normalize = True ):
    '''
    Train transformer pipelines to X and returned the transformer dataset.

    Inputs:
    X: pandas (n,m), representing the dataset with n samples and m features to transform.
    idnumerical: numpy, representing the id of numerical features in X to transform using numpipe.
    idnominal: numpy, representing the id of nominal features in X to transform using nompipe.
    imputation: boolean, representing if imputation should be applied to X.
    encode: string, representing the type of encoding to apply to nominal features in X.
    standardscale: boolean, representing if standardization should be applied to X.
    normalize: boolean, representing if normalization should be applied to X.

    Outputs:
    X_final: csr_matrix, sparse matrix with the transformed X.
    pipe_nom: pipeline, representing the pipeline with transformer to apply to nominal features in X.
    pipe_num: pipeline, representing the pipeline with transformer to apply to numerical features in X.
    pipe_normalize: pipeline, representing the pipeline with transformer to apply to normalize the features in X.
    numerical: numpy, representing the name of numerical features
    nominal: numpy, representing the name of nominal features
    '''

    #Identifying numerical and nominal variables
    if idnumerical==None:
        idnumerical = [i for i, e in enumerate(X.dtypes) if e == 'float64']
    if idnominal==None:
        idnominal = list(np.setdiff1d(list(range(0,len(X.columns))),idnumerical))
        
    numerical = list(X.iloc[:,idnumerical].columns)
    nominal = list(X.iloc[:,idnominal].columns)
        
    X_nom = X.loc[:,nominal]
    X_num = X.loc[:,numerical]
    

    #Applying estimators for nominal an numerical
    #nominal
    pipe_nom = None
    if len(nominal)>0:
        estimators = []
        if imputation == True:
            imp_nom = SimpleImputer(strategy='most_frequent')
            estimators.append(('imputation', imp_nom))
        if encode != None:
            enc = OneHotEncoder(drop='first') if encode=='one-hot' else OrdinalEncoder()
            estimators.append(('encoding', enc))
        if standardscale:
            scaler = StandardScaler(with_mean=True) if encode=='label' else StandardScaler(with_mean=False)
            estimators.append(('standardscale', scaler))

        pipe_nom = Pipeline(estimators)
        pipe_nom.fit(X_nom)
    
    #numerical
    pipe_num = None
    if len(numerical)>0:
        estimators = []
        if imputation == True:
            imp_num = IterativeImputer(max_iter=100, random_state=1)
            estimators.append(('imputation', imp_num))
        if standardscale:
            scale = StandardScaler(with_mean=True)
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
    
    # Add Normalize pipeline
    pipe_normalize = None
    if normalize:
        estimators = []
        estimators.append(('normalize', Normalizer()))
        pipe_normalize = Pipeline(estimators)
        pipe_normalize.fit(X_final)
        X_final = pipe_normalize.transform(X_final)

    hot_encoder = pipe_nom['encoding']
    if encode == 'one-hot':
        nom_features = hot_encoder.get_feature_names_out(nominal)
    else:
        nom_features = nominal
    

    return (X_final, pipe_num, pipe_nom , pipe_normalize, numerical, nom_features)

def rangeQuery(db, dist_f, point, e):
    '''
    Return the N neighbors of point in db within a dist e
    
    Input
    db: a numpy (n,m), representing the n samples with m features.
    dist_f: func, representing the distance function to compute the distance between each sample in db and point
    point: numpy (m,), representing the point with m features
    e: float, representing the epsilon to find neighbors
    
    Output
    N: a numpy (r,m), representing the r neighbors with m features within distance e
    '''
    #Get distance
    distance = dist_f(db, point)
    
    #Get index of those whose distance is below e
    N = np.where(distance<=e)[0]
    
    return N

def sensitiveQuery(S, sensitive_attr, y):
    '''
    Return the count and rate of values equal and not equal to sensitive_attr
    
    Input
    S: numpy len=m, representing the sensitive attributes
    sensitive_attr: object, representing the value to compute the number and rate
    y: numpy len(y)=m, representing the outcome
    
    Output (tuple)
    total_S: int, total of values equal to sensitive_attr
    total-total_S: int, total of values not equal to sensitive_attr
    total_S/total: float, ratio of values equal to sensitive_attr
    1-(total_S/total): float, ratio of values not equal to sensitive_attr
    balance: float, min(total_S/(1-total_S), (1-total_S)/total_S)
    '''
    unique, unique_counts = np.unique(S, return_counts=True)
    total = sum(unique_counts)
    mi = None
    
    if len(unique)==1:
        balance = 0
        total_S = unique_counts[np.where(unique==sensitive_attr)]
        if len(total_S)==0: total_S=0
        
    else:
        total = sum(unique_counts)
        total_S = unique_counts[np.where(unique==sensitive_attr)]
        
        balance = min(total_S/(total-total_S),(total-total_S)/total_S)
    
    if y is not None:
        mi = mutual_info(S, y)
    
    return total_S, total-total_S, total_S/total, 1-(total_S/total), balance, mi

def euclidean(X, Y):
    '''
    Return the euclidean distance between X and Y, both with the same number of features
    
    Input
    X: numpy (n,m), where each row m is a sample and each column is a feature
    Y: numpy (r,m), where each row r is a sample and each column is a feature
    
    Output
    distance: matrix (m,n) with pairwise distance between each sample of X and Y.
    '''
    #Get the number of features
    if len(X.shape)>1: m = X.shape[1]
    elif len(Y.shape)>1: m = Y.shape[1]
    else: m = X.shape[0]
    
    #Reshape to assure the matrix computations
    X = np.array(X).reshape(-1,m)
    Y = np.array(Y).reshape(-1,m)
    
    #Compute the element of (a-b)^2=a^2+b^2-2ab
    X_dots = (X*X).sum(axis=1).reshape(-1,1)*np.ones((X.shape[0],Y.shape[0]))
    Y_dots = (Y*Y).sum(axis=1)*np.ones((X.shape[0],Y.shape[0]))
    two_XY = 2*X@Y.T
    
    #Compute distance
    distance = X_dots + Y_dots - two_XY
    
    #Squared
    less_zero = np.where(distance<=0.0)
    distance[less_zero] = 0
    distance = np.sqrt(distance)
    
    return distance

def local_pmi(x1, x2, p_x1_x2, base = 'e'):
    '''
    Return the pointwise mutual information by using the given distributions of 
    p_x1 and p_x2
    
    Inputs:
    x1  : numpy (n,1), representing the observed values of x1
    x2  : numpy (n,1), representing the observed values of x2
    p_x1_x2: panda, representing the distribution of x1,x2. Domain of x1 is in the first column and the next contains the x2's domain.
                    note that p_x1_x2 = p(x1,x2)/p(x1)p(x2)
    base: int or string, representing the base of logarithmic computation
    
    Output
    local_pmi: float, local pointwise mutual information
    '''
    local_pmi = 0
    n = x1.shape[0]
    
    x1_v, _ = np.unique(x1, return_counts=True)
#     px1 = x1_f/sum(x1_f)
    
    x2_v, _ = np.unique(x2, return_counts=True)
#     px2 = x2_f/sum(x2_f)
    
    for v1 in x1_v:
        for v2 in x2_v:
            #Note that p_x1_x2 = P(x1,x2)/P(x1)P(x2)
            local_pmi = local_pmi + (sum((x1==v1) & (x2==v2))/n)*np.log(p_x1_x2[str(v1)+'_'+str(v2)]) if p_x1_x2[str(v1)+'_'+str(v2)]>0 else local_pmi
    
    return local_pmi

def mutual_info(x1, x2, base = 'e'):
    '''
    Return the mutual information of x1 and x2
    '''
    x1_entropy = entropy(x1, base=base)
    cond_entropy = conditional_entropy(x1,x2,base)
    
    return x1_entropy - cond_entropy

def conditional_entropy(x1, x2, base = 'e'):
    '''
    Return the conditional entropy H(x1|x2)
    
    Inputs:
    x1: numpy (n,1), representing the observed values of x1
    x2: numpy (n,1), representing the observed values of x2
    base: int or string, representing the base of logarithmic computation
    
    Output
    c_e: float, conditional entropy
    '''
    
    return entropy(x1,x2,base)-entropy(x2,base)

def entropy(x1, x2=None, base = 'e'):
    '''
    Return the entropy of x1, x2, or only x1 in case x2 is None
    
    Inputs:
    x1: numpy (n,1), representing the observed values of x1
    x2: numpy (n,1), representing the observed values of x2
    base: int or string, representing the base of logarithmic computation
    
    Output
    e: float, entropy
    '''
    #Compute pbb of each value
    v_x1, f_x1 = np.unique(x1, return_counts=True)
    px1 = f_x1/sum(f_x1)
    
    if x2 is None:
        if base == 'e':
            e = -1*np.sum(px1 * np.log(px1))
        elif base == 2:
            e = -1*np.sum(px1 * np.log2(px1))
        else:
            e = -1*np.sum(px1 * np.log10(px1))
    else:
        v_x2, f_x2 = np.unique(x2, return_counts=True)
        px2 = f_x2/sum(f_x2)

        p_x1_x2 = np.zeros((v_x1.shape[0], v_x2.shape[0]))
        for v1 in range(len(v_x1)):
            for v2 in range(len(v_x2)):
                p_x1_x2[v1,v2] = sum((x1==v_x1[v1]) & (x2==v_x2[v2]))

        p_x1_x2 /= p_x1_x2.sum()

        p_x1_x2 = p_x1_x2[p_x1_x2>0]
        if base=='e':
            e = -1*np.sum(p_x1_x2 * np.log(p_x1_x2))
        elif base ==2:
            e = -1*np.sum(p_x1_x2 * np.log2(p_x1_x2))
        else:
            e = -1*np.sum(p_x1_x2 * np.log10(p_x1_x2))
    
    return e

def sigmoid(X, dim=0):
    '''
    Returns the sigmoid function of X

    Input
    X: numpy (n,m), representing the dataset of n observations with m dimensions.
    dim: int, representing the dimension of X to apply the sigmoid function.

    Output
    sig: numpy (n,m), representing the sigmoid function of X
    '''
    sig = 1/(1+np.exp(-X))
    return sig

def preprocess(X, nominal=None, numerical=None):
    '''
    Returns a preprocessed data X by applying OneHot encoding to columns in
    nominal and Scale in numerical.
    
    Input
    X: panda (n,m), representing the dataset of n observations and m columns
    nominal: array (l, type: int), representing the position of l<=m columns to transform into OneHot encoding.
    numerical: array (r, type: int), representing the position r<=r columns to Scale.
    
    Output:
    encoded_X: (n,P), representing the data keeping the n observations but with P>=m encoded columns.
    col_names: array (P), representing the names of the P columns w.r.t. the order in encoded_X.
    one_hot_encoder: preprocessing.OneHotEncoder(), representing the encoder that preprocess the nominal
                    columns into one-hot codifications.
    scaler: preprocessing.StandardScaler(), representing the encoder that preprocess the numerical
            columns into standard scale.
    '''
    one_hot_encoder = None
    scaler = None
    
    n = X.shape[0]
    if nominal is not(None):
        nom = X.iloc[:,nominal]
        #Preprocess nominal
        one_hot_encoder = OneHotEncoder().fit(nom)
        X_nom = one_hot_encoder.transform(nom).toarray()
        col_nom = np.array([i.replace(" ", "") for i in one_hot_encoder.get_feature_names_out()], dtype=str)
    
    if numerical is not(None):
        num = X.iloc[:,numerical]
        #Preprocess numerical
        scaler = StandardScaler().fit(num)
        X_num = scaler.transform(num)
        col_num = np.array([i.replace(" ", "") for i in scaler.get_feature_names_out()], dtype=str)
    
    if nominal is not(None) and numerical is not(None):
        #Merge results
        encoded_X = np.concatenate((X_nom.reshape(n,-1), X_num.reshape(n,-1)), axis = 1)
        col_names = np.concatenate((col_nom, col_num))
    else:
        if nominal is not(None) and numerical is None:
            encoded_X = X_nom.reshape(n,-1)
            col_names = col_nom
        else:
            encoded_X = X_num.reshape(n,-1)
            col_names = col_num
            
    return encoded_X, col_names, one_hot_encoder, scaler
    
    
#used in create_syntehtic_dataset
def circle_max_distance(center, max_radius):
    '''
    Returns a point between the circle whose center is placed at center position
    and with radius=max_radius
    
    Input
    center: int-tuple (x,y), representing the position of center
    max_radius: float, representing the length of radius of circle
    
    Output
    (x,y): float-tuple, representing the position of sampling point
    '''
        
    c_x, c_y = center
    theta = random.uniform(0, 2 * math.pi)
    r = math.sqrt(random.uniform(0, max_radius*max_radius))
    return [r* math.cos(theta)+c_x, r*math.sin(theta)+c_y]

def epsilon_max_distance(center, vert_max_radius, hor_max_radius):
    '''
    Returns a point between the epsilon whose center is placed at center position
    and with radius={vert_max_radius, hor_max_radius}
    
    Input
    center: int-tuple (x,y), representing the position of center
    vert_max_radius: float, representing the length of vertical radius of epsilon
    hor_max_radius: float, representing the length of horizontal radius of epsilon
    
    Output
    (x,y): float-tuple, representing the position of sampling point
    '''
    c_x, c_y = center
    theta = random.uniform(0, 2 * math.pi)
    r_hor = math.sqrt(random.uniform(0, hor_max_radius*hor_max_radius))
    r_ver = math.sqrt(random.uniform(0, vert_max_radius*vert_max_radius))
    
    return [r_hor* math.cos(theta)+c_x, r_ver*math.sin(theta)+c_y]

#used in create_synthetic_dataset
def square_less_circle(center, max_radius, a):
    '''
    Returns a point between an edge of square of size a and the circle of
    radius max_radius and center at center
    
    Input
    center: int-tuple (x,y), representing the position of center
    max_radius: float, representing the length of radius of circle
    a: float (x,), representing the size of the edge of square
    
    Output
    (x,y): float-tuple, representing the position of sampling point
    '''
        
    c_x, c_y = center
    
    while True:
        x = random.uniform(c_x-a/2, c_x+a/2)
        y = random.uniform(c_y-a/2, c_y+a/2)
        if euclidean(np.array([x, y]), np.array([c_x, c_y]))>max_radius:
            return x,y

#used in create_synthetic_dataset
def sample_sensitive(n, gamma=None):
    '''
    Return n samples of sensitive attribute, assuming the sensitive
    attribute \in {0,1}
    
    Input
    n: int, representing the number of points to sample
    gamma: numpy (n), representing the probability that s of sample i would be 0
    
    Output
    s: numpy (n), representing the sampling sensitive attributes
    
    '''
    
    #if gamma==None then s~U(0,1)
    if gamma is None:
        gamma = np.ones((n))*0.5
    
    #Sampling sensitive attributes
    s = np.ones((n))
    for i in range(n):
        s[i] = np.random.choice([1,0], size=1, p = [gamma[i],1-gamma[i]])
        
    return s

#used in create_synthetic_dataset
def sample_target(n, gamma=None):
    '''
    Return n samples of sensitive attribute, assuming the sensitive
    attribute \in {0,1}
    
    Input
    n: int, representing the number of points to sample
    s: numpy (n,), representing the sensitive attribute
    gamma: numpy (n), representing the probability that y of sample i would be 0
    
    Output
    s: numpy (n), representing the sampling sensitive attributes
    
    '''
    
    #if gamma==None then s~U(0,1)
    if gamma is None:
        gamma = np.ones((n))*0.5
    
    #Sampling sensitive attributes
    s = np.ones((n))
    for i in range(n):
        s[i] = np.random.choice([1,0], size=1, p = [gamma[i],1-gamma[i]])
        
    return s


def reweighing(S, protected_group, Y, positive_class):
    '''
    Returns a dataset with weightings to take them into account in the training of 
    a classifier
    
    Inputs:
    D: numpy (n,m), representing the n samples with m dimensions
    S: numpy (n,), representing the sensitive attributes for the n samples
    protected_group: object, representing the value in S related to the protected group
    Y: numpy (n,), representing the target variable for the n samples.
    positive_class: object, representing the value for positive class in Y
    
    Outputs:
    D_US: numpy (n,m), representing the uniform sampled D
    final_Y: numpy (n), representing the uniform sampled Y
    final_S: numpy (n), representing the uniform sampled S    
    '''
    preproc_S = np.array(1*(S==protected_group))
    preproc_Y = np.array(1*(Y==positive_class))
    W = {}
    
    for s in [0,1]:
        for y in [0, 1]:
            W[str(s)+'_'+str(y)] = sum(preproc_S==s)*sum(preproc_Y==y)/(preproc_S.shape[0]*sum((preproc_Y==y) & (preproc_S==s)))
    
    D_w = np.zeros((preproc_S.shape[0],1))
    
    for i in range(D_w.shape[0]):
        D_w[i] = W[str(preproc_S[i])+'_'+str(preproc_Y[i])]
        
    return D_w.reshape(-1)

def uniform_sampling(D, S, protected_group, Y, positive_class):
    '''
    Return a resampling from D s.t. a trained classifier on this dataset is
    mitigated the discrimination
    
    Input:
    D: numpy (n,m), representing the n samples with m dimensions
    S: numpy (n,), representing the sensitive attributes for the n samples
    protected_group: object, representing the value in S related to the protected group
    Y: numpy (n,), representing the target variable for the n samples.
    positive_class: object, representing the value for positive class in Y
    
    Output:
    D_US: numpy (n,m), representing the uniform sampled D
    final_Y: numpy (n), representing the uniform sampled Y
    final_S: numpy (n), representing the uniform sampled S
    '''
    preproc_S = np.array(1*(S==protected_group))
    preproc_Y = np.array(1*(Y==positive_class))
    W = {}
    
    for s in [0,1]:
        for y in [0, 1]:
            W[str(s)+'_'+str(y)] = sum(preproc_S==s)*sum(preproc_Y==y)/(D.shape[0]*sum((preproc_Y==y) & (preproc_S==s)))
    
    DP = D[np.where((preproc_S==1) & (preproc_Y==1))]
    DP = np.array(random.choices(DP, k=int(round(W['1_1']*len(DP),0))))
    final_Y = np.ones(DP.shape[0])
    final_S = np.ones(DP.shape[0])
    
    FP = D[np.where((preproc_S==0) & (preproc_Y==1))]
    FP = np.array(random.choices(FP, k=int(round(W['0_1']*len(FP),0))))
    final_Y = np.concatenate((final_Y, np.ones(FP.shape[0])))
    final_S = np.concatenate((final_S, np.zeros(FP.shape[0])))
    
    DN = D[np.where((preproc_S==1) & (preproc_Y==0))]
    DN = np.array(random.choices(DN, k=int(round(W['1_0']*len(DN),0))))
    final_Y = np.concatenate((final_Y, np.zeros(DN.shape[0])))
    final_S = np.concatenate((final_S, np.ones(DN.shape[0])))
    
    FN = D[np.where((preproc_S==0) & (preproc_Y==0))]
    FN = np.array(random.choices(FN, k=int(round(W['0_0']*len(FN),0))))
    final_Y = np.concatenate((final_Y, np.zeros(FN.shape[0])))
    final_S = np.concatenate((final_S, np.zeros(FN.shape[0])))
    
    D_US = np.concatenate((DP, FP, DN, FN))
    
    return D_US, final_Y, final_S

def preferential_sampling(D, S, protected_group, Y, positive_class, ranker):
    '''
    Returns a preferential sampled D using a ranker
    
    Inputs:
    D: numpy (n,m), representing the n samples with m dimensions
    S: numpy (n,), representing the sensitive attributes for the n samples
    protected_group: object, representing the value in S related to the protected group
    Y: numpy (n,), representing the target variable for the n samples.
    positive_class: object, representing the value for positive class in Y
    ranker: model, representing the ranker to be fitted in D. The ranker must have
            the method fit.
    
    Outputs:
    D_US: numpy (n,m), representing the uniform sampled D
    final_Y: numpy (n), representing the uniform sampled Y
    final_S: numpy (n), representing the uniform sampled S    
    '''
    preproc_S = np.array(1*(S==protected_group))
    preproc_Y = np.array(1*(Y==positive_class))
    W = {}
    
    for s in [0,1]:
        for y in [0, 1]:
            W[str(s)+'_'+str(y)] = sum(preproc_S==s)*sum(preproc_Y==y)/(preproc_S.shape[0]*sum((preproc_Y==y) & (preproc_S==s)))
    
    ranker.fit(D, preproc_Y)
    
    
    D_PS = np.empty((0,D.shape[1]+2))
    
    
    D = np.concatenate((D, preproc_S.reshape(-1,1), preproc_Y.reshape(-1,1)), axis=1)
    
    DP = D[((preproc_S==1) & (preproc_Y==1)),:]
    FP = D[((preproc_S==0) & (preproc_Y==1)),:]
    DN = D[((preproc_S==1) & (preproc_Y==0)),:]
    FN = D[((preproc_S==0) & (preproc_Y==0)),:]
    
    rank_DP = ranker.predict_proba(DP[:,:-2])[:,1].argsort()
    rank_FP = ranker.predict_proba(FP[:,:-2])[:,1].argsort()
    
    rank_DN = (-ranker.predict_proba(DN[:,:-2])[:,1]).argsort()
    rank_FN = (-ranker.predict_proba(FN[:,:-2])[:,1]).argsort()
    
    for i in range(0,math.floor(W['1_1'])):
        D_PS = np.concatenate((D_PS, DP))
    
    ranked_el = math.floor((W['1_1']-math.floor(W['1_1']))*DP.shape[0])
    if ranked_el>0:
        D_PS = np.concatenate((D_PS, DP[(rank_DP<=ranked_el),:]))
        
    ranked_el = math.floor(W['1_0']*DN.shape[0])
    if ranked_el>0:
        D_PS = np.concatenate((D_PS, DN[(rank_DN<=ranked_el),:]))
        
    ranked_el = math.floor(W['0_1']*FP.shape[0])
    if ranked_el>0:
        D_PS = np.concatenate((D_PS, FP[(rank_FP>=(max(rank_FP)-ranked_el)),:]))
    
    for i in range(0,math.floor(W['0_0'])):
        D_PS = np.concatenate((D_PS, FN))
    
    ranked_el = math.floor((W['0_0']-math.floor(W['1_0']))*FP.shape[0])
    if ranked_el>0:
        D_PS = np.concatenate((D_PS, FN[(rank_FN>=(max(rank_FN)-ranked_el)),:]))
    
    final_S = D_PS[:,-2]
    final_Y = D_PS[:,-1]
    D_PS = D_PS[:,:-2]

    return D_PS, final_Y, final_S

def compute_results(classifiers, X, y_true, pos_class, S, prot_group, metric_func=None):
    '''
    Print the results for the given prediction and ground truth by using the 
    performance measure function
    
    Inputs:
    classifiers: dic, representing the classifiers. The keys must be
                the name of the models and the value the prediction as (n,) numpy.
    y_true: numpy (n,), representing the ground truth of the target variable.
    pos_class: value, representing the positive class.
    S: numpy (n,), representing the sensitive attribute for retrieving the group membership.
    prot_group: value, representing the membership value for the protected group.
    metric_func: func, representing the metric function to compute the performance.
    '''
    if metric_func is None:
        metric_func = get_scorer('f1_macro')
    
    print('PERFORMANCE')
    print('Overall')
    for m in classifiers:
        print(f'{m}: {round(metric_func(classifiers[m], X, y_true),3)}')
    print()
    print('Protected Group')
    for m in classifiers:
        print(f'{m}: {round(metric_func(classifiers[m], X[S==prot_group], y_true[S==prot_group]),3)}')
    print()
    print('Unprotected Group')
    for m in classifiers:
        print(f'{m}: {round(metric_func(classifiers[m], X[S!=prot_group], y_true[S!=prot_group]),3)}')
    print()    
    print()
    print('SELECTION RATES')
    print('Overall')
    for m in classifiers:
        overall, _, _ = selection_rate(classifiers[m], S, unpriv_class = prot_group)
        print(f'{m}: {round(overall,3)}')
    print()
    print('Protected Group')
    for m in classifiers:
        _, unpr, _ = selection_rate(classifiers[m], S, unpriv_class = prot_group)
        print(f'{m}: {round(unpr,3)}')
    print()
    print('Unprotected Group')
    for m in classifiers:
        _, _, priv = selection_rate(classifiers[m], S, unpriv_class = prot_group)
        print(f'{m}: {round(priv,3)}')
            
    return

def compute_disparities(y_pred, y_true, pos_class, S, protected_group):
    '''
    Returns a dictionary with several disparity measures:
    - dem_p = demographic parity
    - eq_opp = equalized opportunity
    - eq_odd = equalized odd
    
    Inputs:
    y_pred: numpy (n,), representing the prediction of a classifier of the target y
    y_true: numpy (n,), representing the ground truth of target y
    pos_class: value, representing the value related to the positive class.
    S: numpy (n,), representing the sensitive attribute
    protected_group: value, representing the value in S associated to the protected group
    
    Outputs:
    dic_disp: dictionary of disparities computed from inputs.
    '''
    disc_disp = {}
    
    prot_g = (S==protected_group)
    unpr_g = (S!=protected_group)
    
    dem_p = sum(y_pred[prot_g])/sum(prot_g)-sum(y_pred[unpr_g])/sum(unpr_g)
    eq_opp = (recall(y_pred[prot_g], y_true[prot_g], pos_class)
                - recall(y_pred[unpr_g], y_true[unpr_g], pos_class))
    eq_odd = 0.5*(abs((recall(y_pred[prot_g], y_true[prot_g], pos_class)- recall(y_pred[unpr_g], y_true[unpr_g], pos_class)))
                    +abs((fpr(y_pred[prot_g], y_true[prot_g], pos_class)- fpr(y_pred[unpr_g], y_true[unpr_g], pos_class))))
    
    disc_disp['dem_parity'] = dem_p
    disc_disp['eq_opp'] = eq_opp
    disc_disp['eq_odd'] = eq_odd
    
    return disc_disp

##Synthetic based on ex-ante risk.
#Define functions to make replications
def makeCopies(X, y, s, gamma, n_copies):
    X_final = X.copy()
    s_final = s.copy()
    y_final = y.copy()
    gamma_final = gamma.copy()
    
    #copies with original s
    for i in range(n_copies-1):
        X_final = np.concatenate((X_final, X.copy()), axis= 0)
        s_final = np.concatenate((s_final, s), axis = 0)
        y_final = np.concatenate((y_final, y.copy()), axis=0)
        gamma_final = np.concatenate((gamma_final, gamma.copy()), axis=0)
    
    #copies with S flipped
    for i in range(n_copies):
        X_final = np.concatenate((X_final, X.copy()), axis= 0)
        s_final = np.concatenate((s_final, 1-s), axis = 0)
        y_final = np.concatenate((y_final, y.copy()), axis=0)
        gamma_final = np.concatenate((gamma_final, gamma.copy()), axis=0)
    
    return X_final, s_final, y_final, gamma_final
    
def createSynthetic(X, y, s, gamma, n_copies):
    X_copy, s_copy, y_copy, gamma_final = makeCopies(X,y,s,gamma, n_copies)
    mid = int((X_copy.shape[0]/2))
    
    for i in range(mid, X_copy.shape[0]):
        y_copy[i] = np.random.choice([0,1], size=1, p = [gamma_final[i],1-gamma_final[i]])
        
    return X_copy, s_copy, y_copy

def computeRisk(X, S, Y):
    risk = np.zeros((X.shape[0]))
    for x in range(X.shape[0]):
        s_x = S[x]
        dim1 = X[x,0]
        dim2 = X[x,1]
        risk[x] = abs(Y[(S==s_x) & (X[:,0]==dim1) & (X[:,1]==dim2)].mean()-
                    Y[(S==1-s_x) & (X[:,0]==dim1) & (X[:,1]==dim2)].mean())
    return risk

def getCombinationParameters(dic_parameters):
    settings = {}

    for model in dic_parameters.keys():
        temp = list(dic_parameters[model].keys())
        res = dict()
        cnt = 0

        for combs in product (*dic_parameters[model].values()):
            res[cnt] = {ele: cnt for ele, cnt in zip(dic_parameters[model], combs)}
            cnt += 1
        
        settings[model] = res

    return settings

def save_file(file, dir_path, name_file):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    if dir_path[-1] != '/': dir_path +='/'

    with open(dir_path+name_file, 'wb') as f:
        pickle.dump(file, f, protocol = pickle.HIGHEST_PROTOCOL)

def compute_results():
    return
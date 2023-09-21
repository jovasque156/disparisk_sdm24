from sklearn import metrics
import numpy as np

#Defining the Group Fair metrics
def log_loss(y_true, y_pred_prob):
    return metrics.log_loss(y_true, y_pred_prob)

def accuracy(y_true, y_pred):
    return metrics.accuracy_score(y_true, y_pred)

def auc(y_true, y_pred):
    return metrics.roc_auc_score(y_true, y_pred)

def f1score(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred)

def recall(y_true, y_pred, pos_class=1):
    #it returns the recall/TPR
    #it is assumed that positive class is equal to 1

    return y_pred[y_true==pos_class].sum()/y_true.sum() if y_true.sum()>0 else 0

def fpr(y_true, y_pred, pos_class):
    #It returns the False Positive Rate
    #it is assumed that positive class is equal to 1

    return y_pred[y_true!=pos_class].sum()/sum(y_true!=pos_class) if sum(y_true!=pos_class)>0 else 0

def precision(y_true, y_pred):
    #It returns the precision

    return y_true[y_pred==1].sum()/y_pred.sum() if y_pred.sum()>0 else 0

def selection_rate(y_pred, protected_attr, unpriv_class, priv_class=None):
    #It returns the Selection Ratio for privileged and unprivileged group
    #The positive class must be equal to 1, which is used for 'select' the individual
    #Pr(h=1|priv_class=a)

    overall = y_pred.sum()/len(y_pred)
    if priv_class == None:
        priv=y_pred[protected_attr!=unpriv_class].sum()/len(y_pred[protected_attr!=unpriv_class]) if len(y_pred[protected_attr!=unpriv_class])>0 else 0
    else:
        priv=y_pred[protected_attr==priv_class].sum()/len(y_pred[protected_attr==priv_class]) if len(y_pred[protected_attr==priv_class])>0 else 0
        
    unpr=y_pred[protected_attr==unpriv_class].sum()/len(y_pred[protected_attr==unpriv_class]) if len(y_pred[protected_attr==unpriv_class])>0 else 0

    return overall, unpr, priv

def demographic_parity_dif(y_pred, protected_attr, unpriv_class, priv_class=None):
    #It returns the Statistical Parity Difference considering the prediction
    #It is assumed that positive class is equal to 1
    #Pr(h=1|priv_class=unprivileged) - Pr(h=1|priv_class=privileged)

    _, unpr, priv = selection_rate(y_pred, protected_attr, unpriv_class, priv_class)

    return priv-unpr

def disparate_impact_rate(y_pred, protected_attr, unpriv_class, priv_class=None):
    '''
    It returns the Disparate Impact Ratio
    It is assumed that positive class is equal to 1
    Pr(h=1|priv_class=unprivileged)/Pr(h=1|priv_class=privileged)
    Note that when Disparate Impact Ratio<1, it is considered a negative impact to unprivileged class
    This ratio can be compared to a threshold t (most of the time 0.8 or 1.2) in order to identify the presence
    of disparate treatment.

    Inputs:
    y_pred: numpy (n,), representing the predictions.
    protected_attr: numpy (n,), representing the sensitive attribute of n samples.
                                It assumed binary membership.
    priv_class: value, representing the value in protected_attr associated to the privileged group.
    unpriv_class: value, representing the value in protected_attr associated to the unprivileged group.

    Outputs: disparate impact rate
    '''

    _, unpr, priv = selection_rate(y_pred, protected_attr, priv_class, unpriv_class)

    return unpr/priv


def equal_opp_dif(y_true, y_pred, protected_attr, unpriv_class, priv_class=None, weight=False):
    #It returns the Equal Opportunity Difference between the priv and unpriv group
    #This is obtained by subtracting the recall/TPR of the priv group to the recall/TPR of the unpriv group

    if priv_class == None:
        tpr_priv = recall(y_true[protected_attr!=unpriv_class], y_pred[protected_attr!=unpriv_class]) if len(y_true[protected_attr!=unpriv_class])>0 and len(y_pred[protected_attr!=unpriv_class])>0 else 0
    else:
        tpr_priv = recall(y_true[protected_attr==priv_class], y_pred[protected_attr==priv_class]) if len(y_true[protected_attr==priv_class])>0 and len(y_pred[protected_attr==priv_class])>0 else 0
    
    tpr_unpriv = recall(y_true[protected_attr==unpriv_class], y_pred[protected_attr==unpriv_class]) if len(y_true[protected_attr==unpriv_class])>0 and len(y_pred[protected_attr==unpriv_class])>0 else 0
    

    return tpr_priv-tpr_unpriv

def equal_odd_dif(y_true, y_pred, protected_attr, unpriv_class, priv_class = None):
    pos_unpriv = y_pred[(y_true == 1) & (protected_attr == unpriv_class)].mean()
    pos_priv = y_pred[(y_true == 1) & (protected_attr != unpriv_class)].mean() if priv_class is None else y_pred[(y_true == 1) & (protected_attr == priv_class)].mean()
    neg_unpriv = y_pred[(y_true == 0) & (protected_attr == unpriv_class)].mean()
    neg_priv = y_pred[(y_true == 0) & (protected_attr != unpriv_class)].mean() if priv_class is None else y_pred[(y_true == 0) & (protected_attr == priv_class)].mean()

    pos_unpriv = 0 if np.isnan(pos_unpriv) else pos_unpriv
    pos_priv = 0 if np.isnan(pos_priv) else pos_priv
    neg_unpriv = 0 if np.isnan(neg_unpriv) else neg_unpriv
    neg_priv = 0 if np.isnan(neg_priv) else neg_priv

    pos = np.abs(pos_priv - pos_unpriv)
    neg = np.abs(neg_priv - neg_unpriv)

    return (pos + neg)*0.5

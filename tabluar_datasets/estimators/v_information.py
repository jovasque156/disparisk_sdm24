import numpy as np
from copy import deepcopy
from sklearn.calibration import CalibratedClassifierCV

# Assumption
# if C is none, then use the original model
# We assume that model_v is already fitted on the entire dataset X_train.

class VInformationEstimator(object):
    def __init__(self, 
                model_v, 
                X_train,
                Y_train):
        '''
        Initiate the class v_information for the predictive v-family.

        Input:
        model_v: sklearn model, representing the function from predictive v-family. Must contain predict_proba
        '''
        super()
        assert hasattr(model_v, 'predict_proba'), 'model_v does not have predict_proba method, use CalibratedClassifierCV'
        self.model_v = model_v
        self.X_train = X_train
        self.Y_train = Y_train

        # Finetuned model on the conditioned variables
        self.model_v_C = None
        self.last_C = None

    def fit_on_C(self, C):
        
        B = C * self.X_train if C is not None else self.X_train

        model_v_C = deepcopy(self.model_v)

        # Fit if the C is different to None and the selected features is a subset of original feature 
        if C is not None and sum(C)<self.X_train.shape[1]:
            model_v_C.fit(B, self.Y_train) 

        self.model_v_C = model_v_C

        self.last_C = C

    def estimate_v_entropy(self, Y, X, C=None):
        '''
        Compute v-entropy of Y for the given data X, Y, C
        Note that if C a list of zeros, this is similar to:

            H(Y|\varnothing) = inf E[-log f[\varnothing](y)]

        Otherwise, conditional entropy is computed:
            
            H(Y|C)

        Input:
        X: numpy array, shape (n_samples, n_features)
        Y: numpy array, shape (n_samples,)
        C: numpy array, shape (n_features,), representing what features are conditioned
            If C is not given (None), all variables are considered.

        Output:
        H_v: float, representing the v-entropy of the given data
        '''
        if C is not None: assert len(C) == X.shape[1], f'len(C)={len(C)} mismatches with X.shape[1]={X.shape[1]}'

        # FIt again if C is different than last fitting and is not None
        if C is not None and not(np.array_equal(C, self.last_C)):
            self.fit_on_C(C)

        total_pve = self.estimate_pve(Y, X, C)

        H_v = sum(total_pve)/total_pve.shape[0]

        return H_v

    def estimate_pve(self, Y, X, C=None):
        '''
        Compute pointwise v-entropy for every point at X
        '''
        if C is not None: assert len(C) == X.shape[1], f'len(C)={len(C)} mismatches with X.shape[1]={X.shape[1]}'

        # FIt again if C is different than last fitting and is not None
        if C is not None and not(np.array_equal(C, self.last_C)):
            self.fit_on_C(C)
            
        B = C * X if C is not None else X
        
        pve = np.zeros((X.shape[0]))

        for i in range(pve.shape[0]):
            prob = self.model_v_C.predict_proba(B[i,:].reshape(1,-1))[0,Y[i]] if C is not None else self.model_v.predict_proba(B[i,:].reshape(1,-1))[0,Y[i]]
            #TODO: look for another solution when prob is 0, since it can be not reflecting that
            # 1: the first term of pvi is bigger (so, more uncertainty from the conditioned)
            # 2: the second term of pvi is bigger (so, less uncertainty from the conditioned)
            pve[i] = -np.log(prob) if prob>0 else 0.0

        return pve

    def estimate_v_information(self, Y, X, C=None):

        Hv_Y = self.estimate_v_entropy(Y, X, C)
        Hv_YX = self.estimate_v_entropy(Y, X)

        I_v = Hv_Y - Hv_YX

        return I_v

    def estimate_pvi(self, Y, X, C=None):

        pve_y = self.estimate_pve(Y, X, C)
        pve_yx = self.estimate_pve(Y, X)

        pve = pve_y - pve_yx

        return pve
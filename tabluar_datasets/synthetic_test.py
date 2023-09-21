#Standard
import numpy as np
import pandas as pd

import warnings
import pickle
warnings.filterwarnings('ignore')

# Own modules and libraries
import source.fairness as fm

DIR_DATA = {
    'dutch_census': 'data/dutch_census/',
    'census_income':'data/census_income/',
    'compas': 'data/compas/',
}

def load_data():
    data_sets = {}

    for data in DIR_DATA:
        with open (DIR_DATA[data]+data+'.pkl', 'rb') as f:
            dic = pickle.load(f)
        
        data_sets[data] = dic
    return data_sets

#Sampling based on weights of classes
def sample_weights(X, S, Y, weights):
    n_samples = X.shape[0]
    n_classes = len(weights)
    n_samples_per_class = np.round(weights*n_samples).astype(int)
    n_samples_per_class[-1] = n_samples - np.sum(n_samples_per_class[:-1])
    X_s, S_s, Y_s = [], [], []
    for i in range(n_classes):
        idx = np.where(Y==i)[0]
        idx = np.random.choice(idx, n_samples_per_class[i], replace=True)
        X_s.append(X[idx])
        S_s.append(S[idx])
        Y_s.append(Y[idx])
    return np.concatenate(X_s), np.concatenate(S_s), np.concatenate(Y_s)

def compute_toussaint(V):
    left = np.log((2+V)/(2-V))-(2*V)/(2+V)
    right = (V**2)/2 + (V**4)/36 + (V**6)/288

    return max(left,right)

if __name__ == '__main__':
    results = pd.DataFrame()
    
    print('Loading data sets')
    data_sets = load_data()

    for ds in ['census_income', 'dutch_census', 'compas']:
        print(f'Computing results for {ds}')
        count=1
        for weight in range(95, 4, -2):
            print(f'Computing results for weight {count} of {len(range(95,4,-2))}')
            X_train, S_train, Y_train = data_sets[ds]['train']
            X_test, S_test, Y_test = data_sets[ds]['test']

            X_train = X_train.toarray()
            X_test = X_test.toarray()

            S_mean = S_train.mean()
            S_std = S_train.std()

            Y_mean = Y_train.mean()
            Y_std = Y_train.std()

            X_s_1, S_s_1, Y_s_1 = X_train[S_train==1], S_train[S_train==1], Y_train[S_train==1]
            X_s_0, S_s_0, Y_s_0 = sample_weights(X_train[S_train==0], S_train[S_train==0], Y_train[S_train==0], np.array([weight/100, 1.-weight/100]))
            X_tr = np.concatenate((X_s_0, X_s_1), axis=0)
            S_tr = np.concatenate((S_s_0, S_s_1), axis=0)
            Y_tr = np.concatenate((Y_s_0, Y_s_1), axis=0)

            X_s_1, S_s_1, Y_s_1 = X_test[S_test==1], S_test[S_test==1], Y_test[S_test==1]
            X_s_0, S_s_0, Y_s_0 = sample_weights(X_test[S_test==0], S_test[S_test==0], Y_test[S_test==0], np.array([weight/100, 1.-weight/100]))
            X_te = np.concatenate((X_s_0, X_s_1), axis=0)
            S_te = np.concatenate((S_s_0, S_s_1), axis=0)
            Y_te = np.concatenate((Y_s_0, Y_s_1), axis=0)
            
            for m in ['sgd_lr', 'mlp_one_layer', 'mlp_two_layer']:
                print(f'Computing results for {m}   ', end='\r')

                #Load estimator
                estimator_s = pickle.load(open(f'v_info_scores/{m}/{ds}_s_v_info_estimator.pkl', 'rb'))
                model_s_v_info_estimator = estimator_s['estimator']
    
                #Compute prediction
                model_s_v_info_estimator.X_train = np.concatenate((X_tr, ((S_tr-S_mean)/S_std).reshape(-1,1)), axis=1)
                model_s_v_info_estimator.model_v_C.fit(X_tr, Y_tr)
    
                Y_pred = model_s_v_info_estimator.model_v_C.predict(X_te)
                dem_p = fm.demographic_parity_dif(Y_pred, S_te, 1)
    
                #Compute PVI(x->s)
                model_s_v_info_estimator.X_train = np.concatenate((X_tr, ((Y_tr-Y_mean)/Y_std).reshape(-1,1)), axis=1)
                model_s_v_info_estimator.Y_train = S_tr
                model_s_v_info_estimator.model_v_C.fit(np.concatenate((X_tr, ((Y_tr-Y_mean)/Y_std).reshape(-1,1)), axis=1), S_tr)
    
                C = np.zeros(X_train.shape[1]+1)
                pve_s = model_s_v_info_estimator.estimate_pve(S_te,
                                                        np.concatenate((X_te, ((Y_te-Y_mean)/Y_std).reshape(-1,1)), axis=1),
                                                        C)
    
                C = np.ones(X_train.shape[1]+1)
                C[-1] = 0
                pve_s_from_x = model_s_v_info_estimator.estimate_pve(S_te,
                                                                np.concatenate((X_te, ((Y_te-Y_mean)/Y_std).reshape(-1,1)), axis=1),
                                                                C)
    
                pvi_s_from_x = pve_s - pve_s_from_x
                I_s_from_x = pvi_s_from_x.mean()
    
                g = (1-S_te.mean())*compute_toussaint(S_te.mean()*abs(dem_p))+S_te.mean()*compute_toussaint((1-S_te.mean())*abs(dem_p))
    
                results = results.append({'dataset': ds, 
                                            'model': m,
                                            'pr(Y=1|S=0)': 1-weight/100,
                                            'DP': abs(dem_p),
                                            't(P(S=1), DP)': g,
                                            'I_v(X_to_S)': I_s_from_x}, ignore_index=True)

            count+=1
                
            print('Saving results so far                                                        ')
            results.to_csv('lower_bounded_results.csv', index=False)

        print()

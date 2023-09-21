import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

data_sets = {}

DIR_DATA = {
    'dutch_census': 'data/dutch_census/',
    'census_income':'data/census_income/',
    'compas': 'data/compas/',
}


for data in DIR_DATA:
    with open (DIR_DATA[data]+data+'.pkl', 'rb') as f:
        dic = pickle.load(f)
    
    data_sets[data] = dic

results = pd.DataFrame(columns = ['dataset', 'model', 'scenario', 'portion', 'I_s_from_x'])

for ds in ['dutch_census', 'census_income', 'compas']:
    print(f'STARTING FOR {ds}')
    X_train, S_train, Y_train = data_sets[ds]['train']
    X_test, S_test, Y_test = data_sets[ds]['train']

    X_train = X_train.toarray()
    X_test = X_test.toarray()
    
    S_mean = S_train.mean()
    S_std = S_train.std()
    
    Y_mean = Y_train.mean()
    Y_std = Y_train.std()
    
    for scenario in range(0,4):
        for portion in range(10, 101, 10):
            print(f'...portion: {portion/100}')
            idx = np.random.choice(X_train.shape[0], int(X_train.shape[0]*portion/100), replace=True)
            X_train_selected = X_train[idx,:]
            S_train_selected = S_train[idx]
            Y_train_selected = Y_train[idx]
            
            for m in ['sgd_lr', 'mlp_one_layer', 'mlp_two_layer']:
                print(f'......model: {m}')
                estimator_s = pickle.load(open(f'v_info_scores/{m}/{ds}_s_v_info_estimator.pkl', 'rb'))
                model_s_v_info_estimator = estimator_s['estimator']

                model_s_v_info_estimator.Y_train = S_train_selected
                model_s_v_info_estimator.X_train = np.concatenate((X_train_selected, ((Y_train_selected-Y_mean)/Y_std).reshape(-1,1)), axis=1)
                model_s_v_info_estimator.fit_on_C(np.ones(X_train_selected.shape[1]+1))

                C = np.zeros(X_train.shape[1]+1)
                pve_s = model_s_v_info_estimator.estimate_pve(S_test,
                                                        np.concatenate((X_test, ((Y_test-Y_mean)/Y_std).reshape(-1,1)), axis=1),
                                                        C)

                C = np.ones(X_train.shape[1]+1)
                C[-1] = 0
                pve_s_from_x = model_s_v_info_estimator.estimate_pve(S_test,
                                                                np.concatenate((X_test, ((Y_test-Y_mean)/Y_std).reshape(-1,1)), axis=1),
                                                                C)

                pvi_s_from_x = pve_s - pve_s_from_x
                I_s_from_x = pvi_s_from_x.mean()
                
                new = {'dataset': ds, 
                        'model': m,
                        'scenario': scenario+1,
                        'portion': portion, 
                        'I_s_from_x': I_s_from_x}
                results = pd.concat([results, pd.DataFrame(new, index=[0])], axis=0, ignore_index=True)

                results.to_csv('size_training.csv', index=False)

    print()
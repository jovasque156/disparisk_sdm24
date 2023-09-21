import os
import pickle
import argparse
import pandas as pd
import numpy as np

import source.utils as ut

import warnings
warnings.filterwarnings("ignore")

import pickle
from copy import deepcopy

from sklearn.calibration import CalibratedClassifierCV
import estimators.v_information as v_information

import ipdb

DIR_DATA = {
    'dutch_census': 'data/dutch_census/',
    'census_income':'data/census_income/',
    'compas': 'data/compas/',
    }

MODELS = ['sgd_lr', 'mlp_one_layer', 'mlp_two_layer']

VERBALIZE = True

def get_dataset(args):
    '''
    Retrieve the data from the pickle files.

    Outputs:
    data_sets: dictionary, containing the data for each dataset.

    '''
    data_sets = {}

    for data in args.datasets:
        assert data in DIR_DATA.keys(), f'Data given ({data}) is not in the possible lists of datasets: f{DIR_DATA.keys()}'
        
        if args.verbalize: print(f'Loading {data}...', end='')
        with open (DIR_DATA[data]+data+'.pkl', 'rb') as f:
            dic = pickle.load(f)
        if args.verbalize: print('Done')

        data_sets[data] = dic
    
    return data_sets

def get_best_model(results, args):
    '''
    Retrieve the best model for each dataset.

    Inputs:
    results: pandas dataframe, containing the performances for each scenario.
    args: arguments from the command line.

    Outputs:
    best_models: list, containing the best model for each dataset.

    '''
    best_perfo = results[args.best_perfo].min() if 'log_loss' in args.best_perfo else results[args.best_perfo].max()
    candidates = results[results[args.best_perfo]==best_perfo]
    ids = np.unique(candidates['id_scenario'])

    best_models = ids

    return best_models

def compute_pve_given_data_model(data, model, s_v_family, args):
    '''
    Compute the pve for a given model and data.

    Inputs:
    data: dictionary, containing the data for each dataset.
    model: string, name of the model.
    args: arguments from the command line.

    Outputs:
    pves: dictionary, containing the pve for each dataset.

    '''
    for d in data.keys():
        if s_v_family:
            X_train, Y_train, S_train = data[d]['train']
            X_test, Y_test, S_test = data[d]['test']
        else:
            X_train, S_train, Y_train = data[d]['train']
            X_test, S_test, Y_test = data[d]['test']

        features_list = data[d]['features']
        features_list = features_list[0]+features_list[1]

        X_model_train = np.concatenate((X_train.toarray(), ((S_train-S_train.mean())/S_train.std()).reshape(-1,1)), axis=1)
        X_model_test = np.concatenate((X_test.toarray(), ((S_test-S_test.mean())/S_test.std()).reshape(-1,1)), axis=1)

        #Obtain best models:
        # best_models = get_best_model(pickle.load(open(f'results/{model}/{d}_results_v_family.pkl', 'rb')), args)
        ipdb.set_trace()
        if s_v_family:
            best_models = get_best_model(pd.read_csv(f'results/{model}/{d}_results_s_v_family.txt'), args)
        else:
            best_models = get_best_model(pd.read_csv(f'results/{model}/{d}_results_y_v_family.txt'), args)
        # id_best = best_models[d][0]
        id_best = best_models[0]
        
        if s_v_family:
            model_set = pickle.load(open(f'models/{m}/{d}_s_v_family_set.pkl', 'rb'))
        else:
            model_set = pickle.load(open(f'models/{m}/{d}_y_v_family_set.pkl', 'rb'))
        model_example = model_set[id_best]

        if not(hasattr(model_example, 'predict_proba')):
            model_example = CalibratedClassifierCV(model_example, cv=10)
            model_example.fit(X_model_train, Y_train)

        #First we create the class the compute v_info and everything else
        v_info = v_information.VInformationEstimator(model_example, X_model_train, Y_train)

        # To compute entropy of Y, we need to use X_train and using the entire model and put into C all variables
        pve_y = v_info.estimate_pve(Y_test, X_model_test, C = np.zeros(X_model_test.shape[1]))
        pve_y_pos = v_info.estimate_pve(np.ones_like(Y_test), X_model_test, C = np.zeros(X_model_test.shape[1]))
        pve_y_neg = v_info.estimate_pve(np.zeros_like(Y_test), X_model_test, C = np.zeros(X_model_test.shape[1]))
        prediction_y = v_info.model_v_C.predict_proba(X_model_test*np.zeros(X_model_test.shape[1]))
        
        # To compute entropy of Y, we need to use X_train and using the entire model and put into C all variables
        pve_y_all = v_info.estimate_pve(Y_test, X_model_test)
        pve_y_all_pos = v_info.estimate_pve(np.ones_like(Y_test), X_model_test)
        pve_y_all_neg = v_info.estimate_pve(np.zeros_like(Y_test), X_model_test)
        prediction_y_all = v_info.model_v.predict_proba(X_model_test) 

        # Let compute now the entropy conditioned to the sensitive attribute
        C = np.zeros(X_model_test.shape[1])
        C[-1] = 1
        pve_y_s = v_info.estimate_pve(Y_test, X_model_test, C = C)
        pve_y_s_pos = v_info.estimate_pve(np.ones_like(Y_test), X_model_test, C = C)
        pve_y_s_neg = v_info.estimate_pve(np.zeros_like(Y_test), X_model_test, C = C)
        prediction_y_s = v_info.model_v_C.predict_proba(X_model_test*C)
        
        # Let compute now the entropy conditioned to the X w/o sensitive attribute
        C = np.ones(X_model_test.shape[1])
        C[-1] = 0
        pve_y_x = v_info.estimate_pve(Y_test, X_model_test, C = C)
        pve_y_x_pos = v_info.estimate_pve(np.ones_like(Y_test), X_model_test, C = C)
        pve_y_x_neg = v_info.estimate_pve(np.zeros_like(Y_test), X_model_test, C = C)
        prediction_y_x = v_info.model_v_C.predict_proba(X_model_test*C)
        
        v_pve_y_features_models_features = {}
        v_pve_y_features_models_features_sensitive = {}
        v_pve_y_features_models_wo_feature = {}
        v_pve_y_features_models_wo_feature_sensitive = {}  
        
        estimator = {
            'id_scenario': id_best,
            'estimator': v_info
        }
        if s_v_family:
            ut.save_file(estimator, f'v_info_scores/{model}', f'{d}_s_v_info_estimator.pkl')
        else:
            ut.save_file(estimator, f'v_info_scores/{model}', f'{d}_y_v_info_estimator.pkl')

        for i, f in enumerate(features_list):
            if args.verbalize: print(f'......{d}......feature {i}/{len(features_list)}', end='\r')
            C = np.zeros(X_model_test.shape[1])
            C[i] = 1
            prev_y_feature = v_info.estimate_pve(Y_test, X_model_test, C = C)
            prev_y_feature_pos = v_info.estimate_pve(np.ones_like(Y_test), X_model_test, C = C)
            prev_y_feature_neg = v_info.estimate_pve(np.zeros_like(Y_test), X_model_test, C = C)
            prediction_y_feature = v_info.model_v_C.predict_proba(X_model_test*C)
            v_pve_y_features_models_features[f] = (prev_y_feature, prev_y_feature_pos, prev_y_feature_neg, prediction_y_feature)

            C = np.zeros(X_model_test.shape[1])
            C[i] = 1
            C[-1] = 1
            prev_y_x_feature = v_info.estimate_pve(Y_test, X_model_test, C = C)
            prev_y_x_feature_pos = v_info.estimate_pve(np.ones_like(Y_test), X_model_test, C = C)
            prev_y_x_feature_neg = v_info.estimate_pve(np.zeros_like(Y_test), X_model_test, C = C)
            prediction_y_x_feature = v_info.model_v_C.predict_proba(X_model_test*C)
            v_pve_y_features_models_features_sensitive[f] = (prev_y_x_feature, prev_y_x_feature_pos, prev_y_x_feature_neg, prediction_y_x_feature)
            
            C = np.ones(X_model_test.shape[1])
            C[i] = 0
            C[-1] = 0
            prev_y_wo_feature = v_info.estimate_pve(Y_test, X_model_test, C = C)
            prev_y_wo_feature_pos = v_info.estimate_pve(np.ones_like(Y_test), X_model_test, C = C)
            prev_y_wo_feature_neg = v_info.estimate_pve(np.zeros_like(Y_test), X_model_test, C = C)
            prediction_y_wo_feature = v_info.model_v_C.predict_proba(X_model_test*C)
            v_pve_y_features_models_wo_feature[f] = (prev_y_wo_feature, prev_y_wo_feature_pos, prev_y_wo_feature_neg, prediction_y_wo_feature)

            C = np.ones(X_model_test.shape[1])
            C[i] = 0
            prev_y_wo_feature = v_info.estimate_pve(Y_test, X_model_test, C = C)
            prev_y_x_wo_feature_pos = v_info.estimate_pve(np.ones_like(Y_test), X_model_test, C = C)
            prev_y_x_wo_feature_neg = v_info.estimate_pve(np.zeros_like(Y_test), X_model_test, C = C)
            prediction_y_x_wo_feature = v_info.model_v_C.predict_proba(X_model_test*C)
            v_pve_y_features_models_wo_feature_sensitive[f] = (prev_y_wo_feature, prev_y_x_wo_feature_pos, prev_y_x_wo_feature_neg, prediction_y_x_wo_feature)

            if s_v_family:
                results = { 
                        'id_scenario': id_best,
                        'pve_s': (pve_y, pve_y_pos, pve_y_neg, prediction_y),
                        'pve_s_all': (pve_y_all, pve_y_all_pos, pve_y_all_neg, prediction_y_all),
                        'pve_s_y': (pve_y_s, pve_y_s_pos, pve_y_s_neg, prediction_y_s),
                        'pve_s_x': (pve_y_x, pve_y_x_pos, pve_y_x_neg, prediction_y_x),
                        'v_pve_s_with_feature_models_features': v_pve_y_features_models_features,
                        'v_pve_s_with_feature_models_features_target': v_pve_y_features_models_features_sensitive,
                        'v_pve_s_wo_feature_model_features': v_pve_y_features_models_wo_feature,
                        'v_pve_s_wo_feature_model_features_target': v_pve_y_features_models_wo_feature_sensitive,
                }
            else:
                results = { 
                        'id_scenario': id_best,
                        'pve_y': (pve_y, pve_y_pos, pve_y_neg, prediction_y),
                        'pve_y_all': (pve_y_all, pve_y_all_pos, pve_y_all_neg, prediction_y_all),
                        'pve_y_s': (pve_y_s, pve_y_s_pos, pve_y_s_neg, prediction_y_s),
                        'pve_y_x': (pve_y_x, pve_y_x_pos, pve_y_x_neg, prediction_y_x),
                        'v_pve_y_with_feature_models_features': v_pve_y_features_models_features,
                        'v_pve_y_with_feature_models_features_sensitive': v_pve_y_features_models_features_sensitive,
                        'v_pve_y_wo_feature_models_features': v_pve_y_features_models_wo_feature,
                        'v_pve_y_wo_feature_models_features_sensitive': v_pve_y_features_models_wo_feature_sensitive
            }
            
            if s_v_family:
                ut.save_file(results, f'v_info_scores/{model}', f'{d}_s_v_info_checkpoint.pkl')
            else:
                ut.save_file(results, f'v_info_scores/{model}', f'{d}_y_v_info_checkpoint.pkl')
        
        if args.verbalize: print(f'......{d}......feature {len(features_list)}/{len(features_list)}......Done.')
        
        if s_v_family:
            os.remove(f'v_info_scores/{model}/{d}_s_v_info_checkpoint.pkl')
            ut.save_file(results, f'v_info_scores/{model}', f'{d}_s_v_info.pkl')
        else:
            os.remove(f'v_info_scores/{model}/{d}_y_v_info_checkpoint.pkl')
            ut.save_file(results, f'v_info_scores/{model}', f'{d}_v_info.pkl')
    
    if args.verbalize: print('......Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbalize', action='store_true', help='Use verbalization')
    parser.add_argument('--models', nargs='+', type=str, default= 'sgd_test' ,help='v-models to compute results')
    parser.add_argument('--datasets', nargs='+', type=str, default='compas', help='Datasets to compute results')
    parser.add_argument('--best_perfo', type=str, default='log_loss_train_mean', help='Performance metric to pick the best model for v-family')
    parser.add_argument('--compute_y_pve_scores', action='store_true', help='Use to compute pve scores for predicting y.')
    parser.add_argument('--compute_s_pve_scores', action='store_true', help='Use to compute pve scores for predicting s.')
    args = parser.parse_args()
    
    print('=================')
    print('RUNNING EXPERIMENTS FOR')
    print(f'Models: {args.models}')
    print(f'Datasets: {args.datasets}')
    print(f'Best perfo: {args.best_perfo}')
    print('=================')
    print()

    if not isinstance(args.datasets, list): args.datasets = [args.datasets]
    if not isinstance(args.models, list): args.models = [args.models]

    if args.verbalize: print('Loading datasets')
    data = get_dataset(args)
    if args.verbalize: print()

    if args.compute_y_pve_scores:
        if args.verbalize: print('Computing pve scores')
        for m in args.models:
            if args.verbalize: print(f'...Model: {m}')
            compute_pve_given_data_model(data, m, False, args)
            if args.verbalize: print()

    if args.compute_s_pve_scores:
        if args.verbalize: print('Computing pve scores from s')
        for m in args.models:    
            if args.verbalize: print(f'...Model: {m}')
            compute_pve_given_data_model(data, m, True, args)
            if args.verbalize: print()

    if args.verbalize: print()
    if args.verbalize: print('Process complete!')
    if args.verbalize: print('=================')
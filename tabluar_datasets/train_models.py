#Standard
import os
import pickle
import numpy as np
import pandas as pd
import argparse
import warnings
import statistics as st
warnings.filterwarnings('ignore')

# Own modules and libraries
import source.utils as ut
import source.fairness as fm

#Models
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.calibration import CalibratedClassifierCV

from copy import deepcopy

import ipdb

HYPERPARAMETERS = {
    'sgd_lr': {'learning_rate': ['constant', 'optimal'],
            'eta0': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1], 
            'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
            'loss': ['log_loss'],
            'penalty':['l1', 'l2'], 
            'n_jobs':[-1],
            'tol': [1e-4]},
    'mlp_one_layer': {'hidden_layer_sizes': [(10,), (100,), (500,)],
            'learning_rate_init': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
            'max_iter': [1000],
            'tol': [1e-4]},
    'mlp_two_layer': {'hidden_layer_sizes':
                                [(10, 10), (10, 100), (10, 500),
                                (100, 10), (100, 100), (100, 500),
                                (500, 10), (500, 100), (500, 500)],
            'learning_rate_init': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
            'max_iter': [1000],
            'tol': [1e-4]}
}

VERBALIZE = True

def trainModels(models_settings, datasets, model, y_v_family, s_v_family, verbalize):
    '''
    Trains a model for each scenario in the dataset.
    Returns a dictionary with the trained models.

    models_settings: dictionary with the settings for each model
    datasets: dictionary with the datasets
    model: string with the model name
    y_v_family: boolean to train for I_v((X,S)->Y)
    s_v_family: boolean to train for I_v((X,Y)->S)
    verbalize: boolean to print the progress

    '''
    total_scenarios = len(models_settings.keys())

    for ds in dataset.keys():
        if verbalize: print(f'Dataset {ds}                     ')
        if y_v_family or s_v_family:
            X, S, Y = datasets[ds]['train']
        else:
            X, _, Y = datasets[ds]['train']

        if not(isinstance(X, (np.ndarray, np.generic))):
            X = X.toarray()

        if y_v_family:
            X = np.concatenate((X, ((S-S.mean())/S.std()).reshape(-1,1)), axis=1)
        elif s_v_family:
            X = np.concatenate((X, ((Y-Y.mean())/Y.std()).reshape(-1,1)), axis=1)
            Y = S.copy()

        settings = {}
        for setting in models_settings.keys():
            if verbalize: print(f'...Scenario {setting+1}/{total_scenarios}', end='\r')
            
            kwargs = models_settings[setting]            

            if 'mlp' in model:
                m = MLPClassifier(random_state = 1, **kwargs)
            elif 'sgd' in model:
                m = SGDClassifier(random_state = 1, **kwargs)
            else:
                raise Exception('Model not found')

            try:
                m.fit(X, Y)
            except:
                continue

            settings[setting] = deepcopy(m)

            if y_v_family:
                ut.save_file(settings, f'models/{model}', f'{ds}_y_v_family_set_checkpoint.pkl')
            elif s_v_family:
                ut.save_file(settings, f'models/{model}', f'{ds}_s_v_family_set_checkpoint.pkl')
            else:
                ut.save_file(settings, f'models/{model}', f'{ds}_set_checkpoint.pkl')
        
        if y_v_family:
            os.remove(f'models/{model}/{ds}_y_v_family_set_checkpoint.pkl')
            ut.save_file(settings, f'models/{model}', f'{ds}_y_v_family_set.pkl')
        elif s_v_family:
            os.remove(f'models/{model}/{ds}_s_v_family_set_checkpoint.pkl')
            ut.save_file(settings, f'models/{model}', f'{ds}_s_v_family_set.pkl')
        else:
            os.remove(f'models/{model}/{ds}_set_checkpoint.pkl')
            ut.save_file(settings, f'models/{model}', f'{ds}_set.pkl')

    if verbalize:
        print()
        print('Done!')


def compute_model_performances(datasets, args, y_v_family, s_v_family):
    '''
    Compute the performances of the models for each scenario.

    Inputs:
    datasets: dictionary, containing the data for each dataset.
    args: arguments from the command line.

    Outputs:
    results: pandas dataframe, containing the performances for each scenario.

    '''
    cols = ['id_scenario', 
            'acc_train_mean',
            'acc_train_sd', 
            'acc_test',
            'f1_train_mean',
            'f1_train_sd',
            'f1_test',
            'log_loss_train_mean',
            'log_loss_train_sd',
            'log_loss_test',
            'demp', 
            'eqopp',
            'eqodd']

    new = {}

    for model in args.models:
        if args.not_verbalize: print(f'Starting with {model} model')
        
        for ds in datasets.keys():
            results = pd.DataFrame(columns = cols)
            
            count = 0
            
            if y_v_family:
                model_sets = pickle.load(open(f'models/{model}/{ds}_y_v_family_set.pkl', 'rb'))
            elif s_v_family:
                model_sets = pickle.load(open(f'models/{model}/{ds}_s_v_family_set.pkl', 'rb'))
            else:
                model_sets = pickle.load(open(f'models/{model}/{ds}_set.pkl', 'rb'))
            if args.not_verbalize: print(f'Computing for {ds} dataset')

            for id_scenario in model_sets.keys():
                count+=1
                if args.not_verbalize: print(f'...setting {count}/{len(model_sets.keys())}', end='\r')
                
                new['id_scenario'] = id_scenario

                m = model_sets[id_scenario]

                if s_v_family:
                    X_train, Y_train, S_train = datasets[ds]['train']
                    X_test, Y_test, S_test = datasets[ds]['test']
                else:
                    X_train, S_train, Y_train = datasets[ds]['train']
                    X_test, S_test, Y_test = datasets[ds]['test']

                if s_v_family or y_v_family:
                    S_train_final = (S_train-S_train.mean())/S_train.std()
                    S_test_final = (S_test-S_test.mean())/S_test.std()

                X_model_train = np.concatenate((X_train.toarray(), S_train_final.reshape(-1,1)), axis=1) if (y_v_family or s_v_family) else X_train
                X_model_test = np.concatenate((X_test.toarray(), S_test_final.reshape(-1,1)), axis=1) if (y_v_family or s_v_family) else X_test

                if not(hasattr(m, 'predict_proba')):
                    m = CalibratedClassifierCV(m, cv=10)
                    m.fit(X_model_train, Y_train)

                Y_pred = m.predict(X_model_test)
                Y_pred_prob = m.predict_proba(X_model_test)
                acc_test = fm.accuracy(Y_test, Y_pred)
                f1_test = fm.f1score(Y_test, Y_pred)
                log_loss_test = fm.log_loss(Y_test, Y_pred_prob)
                demp = fm.demographic_parity_dif(Y_pred, S_test, 1)
                eqopp = fm.equal_opp_dif(Y_test, Y_pred, S_test, 1)
                eqodd = fm.equal_odd_dif(Y_test, Y_pred, S_test, 1)

                new['acc_test'] = acc_test
                new['f1_test'] = f1_test
                new['log_loss_test'] = log_loss_test
                new['demp'] = demp
                new['eqopp'] = eqopp
                new['eqodd'] = eqodd

                scores = cross_validate(estimator=m, 
                                        X=X_model_train, 
                                        y=Y_train,
                                        scoring=('accuracy', 'f1', 'neg_log_loss'))
                
                new['acc_train_mean'], new['acc_train_sd'] = st.mean(scores['test_accuracy']), st.stdev(scores['test_accuracy'])
                new['f1_train_mean'], new['f1_train_sd'] = st.mean(scores['test_f1']), st.stdev(scores['test_f1'])
                new['log_loss_train_mean'], new['log_loss_train_sd'] = -1*st.mean(scores['test_neg_log_loss']), st.stdev(scores['test_neg_log_loss'])
                                
                results = pd.concat([results, pd.DataFrame(new, index=[0])], axis=0, ignore_index=True)
                if not os.path.exists(f'results/{model}/'):
                    os.makedirs(f'results/{model}/')
                
            if y_v_family:
                results.to_csv(f'results/{model}/{ds}_results_y_v_family.txt', index=False)
            elif s_v_family:
                results.to_csv(f'results/{model}/{ds}_results_s_v_family.txt', index=False)
            else:
                results.to_csv(f'results/{model}/{ds}_results.txt', index=False)
                
        if args.not_verbalize: print()

def compute_predictions(datasets, model, y_v_family, s_v_family, verbalize):
    predictions = {}
    for ds in datasets.keys():
        if verbalize: print(f'...Dataset {ds}                           ')
        
        if y_v_family:
            models = pickle.load(open(f'models/{model}/{ds}_y_v_family_set.pkl', 'rb'))
        elif s_v_family:
            models = pickle.load(open(f'models/{model}/{ds}_s_v_family_set.pkl', 'rb'))
        else:
            models = pickle.load(open(f'models/{model}/{ds}_set.pkl', 'rb'))

        predictions = {}
        for setting in models.keys():
            if verbalize: print(f'Setting {setting+1} of {len(models.keys())}', end='\r')

            if y_v_family:
                X_train, S_train, _ = datasets[ds]['train']
                X_test, S_test, _ = datasets[ds]['test']
            elif s_v_family:
                # the third element of the tuple, Y, is considered as an input feature to predict S
                X_train, _, S_train = datasets[ds]['train']
                X_test, _, S_test = datasets[ds]['test']
            else:
                X_train, _, _ = datasets[ds]['train']
                X_test, _, _ = datasets[ds]['test']

            X_train = X_train.toarray() if not(isinstance(X_train, (np.ndarray, np.generic))) else X_train
            X_test = X_test.toarray() if not(isinstance(X_test, (np.ndarray, np.generic))) else X_test

            if y_v_family or s_v_family:
                S_test = (S_test-S_test.mean())/S_test.std()
                S_train = (S_train-S_train.mean())/S_train.std()
                X_train = np.concatenate((X_train, S_train.reshape(-1,1)), axis=1)
                X_test = np.concatenate((X_test, S_test.reshape(-1,1)), axis=1)

            m = models[setting]
            
            Y_train_pred = m.predict(X_train)
            Y_test_pred = m.predict(X_test)

            predictions[setting] = {'train': Y_train_pred, 'test': Y_test_pred}

            if y_v_family:
                ut.save_file(predictions, f'predictions/{model}', f'{ds}_y_v_family_predictions_checkpoint.pkl')
            elif s_v_family:
                ut.save_file(predictions, f'predictions/{model}', f'{ds}_s_v_family_predictions_checkpoint.pkl')
            else:
                ut.save_file(predictions, f'predictions/{model}', f'{ds}_predictions_checkpoint.pkl')
        
        if y_v_family:
            os.remove(f'predictions/{model}/{ds}_y_v_family_predictions_checkpoint.pkl')
            ut.save_file(predictions, f'predictions/{model}', f'{ds}_v_family_predictions.pkl')
        elif s_v_family:
            os.remove(f'predictions/{model}/{ds}_s_v_family_predictions_checkpoint.pkl')
            ut.save_file(predictions, f'predictions/{model}', f'{ds}_s_v_family_predictions.pkl')
        else:
            os.remove(f'predictions/{model}/{ds}_predictions_checkpoint.pkl')
            ut.save_file(predictions, f'predictions/{model}', f'{ds}_predictions.pkl')
            
    if verbalize: print()
    if verbalize: print('Done!')

if __name__=='__main__':
    #Argparse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', type=str, default='compas', help='Datasets to compute results')
    parser.add_argument('--models', nargs='+', type=str, default='sgd_test', help='Models to compute results')

    parser.add_argument('--train_models', action='store_true', help='Train models to predict y from x without s.' )
    parser.add_argument('--predict_models', action='store_true', help='Predict outputs Y of trained models.')
    
    parser.add_argument('--train_y_v_family', action='store_true', help='Train v-family to predict y from (x,s).')
    parser.add_argument('--predict_y_v_family', action='store_true', help='Predict outputs predictive V-families for predict y.')
    
    parser.add_argument('--train_s_v_family', action='store_true', help='Train v-family to predict s from (x,y)')
    parser.add_argument('--predict_s_v_family', action='store_true', help='Predict outputs predictive V-families for predict s.')

    parser.add_argument('--compute_y_v_family_perfo', action='store_true', help='Use to compute v-family performances for predicting y.')
    parser.add_argument('--compute_s_v_family_perfo', action='store_true', help='Use to compute v-family performances for predicting s.')
    parser.add_argument('--compute_models_perfo', action='store_true', help='Use to compute models performances.')
    
    parser.add_argument('--not_verbalize', action='store_false', help='Deactivate verbalization')

    args = parser.parse_args()

    if not isinstance(args.datasets, list): args.datasets = [args.datasets]
    if not isinstance(args.models, list): args.models = [args.models]

    print('RUNNING EXPERIMENTS')
    print(f'{args}')

    if args.not_verbalize: print('Loading datasets')
    dataset = ut.get_datasets(args)
    print()

    model_settings = ut.getCombinationParameters(HYPERPARAMETERS)

    # For models training
    if args.train_models:
        if args.not_verbalize: print('Training models.')
        for model in args.models:
            if args.not_verbalize: print(f'Running {model} settings')
            trainModels(model_settings[model], dataset, model, False, False, args.not_verbalize)
            print()
    if args.predict_models:
        if args.not_verbalize: print('Predicting target variables.')
        for model in args.models:
            if args.not_verbalize: print(f'Running {model} settings')
            compute_predictions(dataset, model, False, False, args.not_verbalize)
            print()
    if args.compute_models_perfo:
        if args.not_verbalize: print('Computing models performance')
        _ = compute_model_performances(dataset, args, False, False)
        if args.not_verbalize: print()

    # For v-family models for y prediction
    if args.train_y_v_family:
        if args.not_verbalize: print('Training v-family models for y prediction.')
        for model in args.models:
            if args.not_verbalize: print(f'Running {model} settings')
            trainModels(model_settings[model], dataset, model, args.train_y_v_family, False, args.not_verbalize)
            if args.not_verbalize: print()
    if args.predict_y_v_family:
        if args.not_verbalize: print('Predicting target variables v-families.')
        for model in args.models:
            if args.not_verbalize: print(f'Running {model} settings')
            compute_predictions(dataset, model, args.predict_y_v_family, False, args.not_verbalize)
            if args.not_verbalize: print()
    if args.compute_y_v_family_perfo:
        if args.not_verbalize: print('Computing v-models performance for predicting y')
        _ = compute_model_performances(dataset, args, True, False)
        if args.not_verbalize: print()

    # For v-family models for s prediction
    if args.train_s_v_family:
        if args.not_verbalize: print('Training v-family models for s prediction.')
        for model in args.models:
            if args.not_verbalize: print(f'Running {model} settings')
            trainModels(model_settings[model], dataset, model, False, args.train_s_v_family, args.not_verbalize)
            if args.not_verbalize: print()
    if args.predict_s_v_family:
        if args.not_verbalize: print('Predicting s variable v-families.')
        for model in args.models:
            if args.not_verbalize: print(f'Running {model} settings')
            compute_predictions(dataset, model, False, args.predict_s_v_family, args.not_verbalize)
            if args.not_verbalize: print()
    if args.compute_s_v_family_perfo:
        if args.not_verbalize: print('Computing v-models performance for predicting s')
        _ = compute_model_performances(dataset, args, False, True)
        if args.not_verbalize: print()

    print()
    print('==================================')
    print()
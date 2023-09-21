#Standard
import os
import argparse
import numpy as np
import pandas as pd
import warnings
import pickle
warnings.filterwarnings('ignore')

# Own modules and libraries
import source.fairness as fm


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


def obtain_results(S, y_pred, y_true, f_pve, metric):
    metric_result = []
    metric_high = []
    metric_low = []

    portion_unpr = []
    portion_unpr_high = []
    portion_unpr_low = []

    portion_pos = []
    portion_pos_high = []
    portion_pos_low = []

    thresholds = []
    threshold_groups = []
    threshold_bottom = np.percentile(f_pve, 0)-1e-6
    for p in range(5,101,5):
        threshold_top = np.percentile(f_pve, p)
        thresholds.append(f'[{p-5},{p}) \n [{round(threshold_bottom,2)},{round(threshold_top,2)})')
        threshold_groups.append(f'{p},{round(threshold_top, 2)}')
        
        # Get the indexes of the samples with high and low f_pve
        bottom = f_pve>threshold_bottom
        top = f_pve<=threshold_top
        selected = bottom & top
        high = f_pve>threshold_top
        low = f_pve<=threshold_top
        
        total_selected = sum(selected)
        total_high = sum(high)
        total_low = sum(low)

        portion_unpr.append(sum(S[selected])/len(S[selected]) if total_selected>0 else 0)
        portion_unpr_high.append(sum(S[high])/len(S[high]) if total_high>0 else 0)
        portion_unpr_low.append(sum(S[low])/len(S[low]) if total_low>0 else 0)
        portion_pos.append(sum(y_true[selected])/len(y_true[selected]) if total_selected>0 else 0)
        portion_pos_high.append(sum(y_true[high])/len(y_true[high]) if total_high>0 else 0)
        portion_pos_low.append(sum(y_true[low])/len(y_true[low]) if total_low>0 else 0)

        # Compute the metrics

        if metric == 'demographic_parity':
            if total_selected>0:
                metric_result.append(fm.demographic_parity_dif(y_pred[selected], S[selected], 1))
            else:
                metric_result.append(None)
            
            if total_high>0:
                metric_high.append(fm.demographic_parity_dif(y_pred[high],  S[high], 1))
            else:
                metric_high.append(None)

            if total_low>0:
                metric_low.append(fm.demographic_parity_dif(y_pred[low], S[low], 1))
            else:
                metric_low.append(None)

        elif metric == 'equalized_odd':
            if total_selected>0:
                metric_result.append(fm.equal_odd_dif(y_true[selected], y_pred[selected], S[selected], 1))
            else:
                metric_result.append(None)

            if total_high>0:
                metric_high.append(fm.equal_odd_dif(y_true[high], y_pred[high],  S[high], 1))
            else:
                metric_high.append(None)

            if total_low>0:
                metric_low.append(fm.equal_odd_dif(y_true[low], y_pred[low], S[low], 1))
            else:
                metric_low.append(None)

        elif metric == 'equalized_opportunity':
            if total_selected>0:
                metric_result.append(fm.equal_opp_dif(y_true[selected], y_pred[selected], S[selected], 1))
            else:
                metric_result.append(None)

            if total_high>0:
                metric_high.append(fm.equal_opp_dif(y_true[high], y_pred[high],  S[high], 1))
            else:
                metric_high.append(None)

            if total_low>0:
                metric_low.append(fm.equal_opp_dif(y_true[low], y_pred[low], S[low], 1))
            else:
                metric_low.append(None)
        else:
            raise ValueError(f'Given metric={metric} is not between the alternatives (demographic_parity, equalized_odd, equalized_opportunity)')
        
        threshold_bottom = threshold_top
    
    result_all = {
        'threshold': thresholds,
        'portion_unpr': portion_unpr,
        'portion_pos': portion_pos,
        'metric': metric_result,
    }

    result_groups = {
        'threshold': threshold_groups,
        'portion_unpr_high': portion_unpr_high,
        'portion_unpr_low': portion_unpr_low,
        'portion_pos_high': portion_pos_high,
        'portion_pos_low': portion_pos_low,
        'metric_high': metric_high,
        'metric_low': metric_low,
    }

    return (result_all, result_groups)

def compute_results_plots(model, data, data_sets, metric):
    best = pickle.load(open(f'v_info_scores/{model}/{data}_v_info.pkl', 'rb'))['id_scenario']
    _, S_test_original, Y_test = data_sets[data]['test']

    v_info_scores = pickle.load(open(f'v_info_scores/{model}/{data}_v_info.pkl', 'rb'))
    pve_y, pve_y_all, pve_y_s, pve_y_x = v_info_scores['pve_y'][0], v_info_scores['pve_y_all'][0], v_info_scores['pve_y_s'][0], v_info_scores['pve_y_x'][0]

    s_v_info_scores = pickle.load(open(f'v_info_scores/{model}/{data}_s_v_info.pkl', 'rb'))    
    pve_s, pve_s_all, pve_s_y, pve_s_x = s_v_info_scores['pve_s'][0], s_v_info_scores['pve_s_all'][0], s_v_info_scores['pve_s_y'][0], s_v_info_scores['pve_s_x'][0]

    if metric=='pvi_from_s_given_x':
        metric_pve = pve_y_x - pve_y_all
    elif metric=='pvi_from_x_given_s':
        metric_pve = pve_y_s - pve_y_all
    elif metric=='pvi_from_s':
        metric_pve = pve_y - pve_y_s
    elif metric=='pvi_from_x_and_s':
        metric_pve = pve_y - pve_y_all
    elif metric=='pvi_from_x':
        metric_pve = pve_y-pve_y_x
    elif metric=='pvi_s_from_x':
        metric_pve = pve_s-pve_s_x
    elif metric=='pvi_s_from_x_and_s':
        metric_pve = pve_s-pve_s_all
    elif metric=='pvi_s_from_x_given_y':
        metric_pve = pve_s_x-pve_s_all
    elif metric=='pvi_s_from_y_given_x':
        metric_pve = pve_s_y-pve_s_all
    elif metric=='pvi_s_from_y':
        metric_pve = pve_s-pve_s_y
        
    if not os.path.exists(f'pvi_results/{model}/{data}/'):
        os.makedirs(f'pvi_results/{model}/{data}/')

    pd.DataFrame(metric_pve, columns=['pve']).to_csv(f'pvi_results/{model}/{data}/{metric}.txt')

    results_plots = pd.DataFrame()
    
    results_groups_plots = pd.DataFrame()

    y_predictions = pickle.load(open(f'predictions/{model}/{data}_predictions.pkl', 'rb'))
    y_pred = y_predictions[best]['test']
    
    results_general, results_general_group = obtain_results(
                                    S_test_original, 
                                    y_pred,
                                    Y_test,
                                    metric_pve,
                                    'demographic_parity')

    portion_pos = results_general['portion_pos']
    portion_unpr = results_general['portion_unpr']
    thresholds = results_general['threshold']
    thresholds_groups = results_general_group['threshold']

    results_models = pd.read_csv(f'results/{model}/{data}_results.txt')

    for scenario in list(results_models['id_scenario']):
            y_predictions = pickle.load(open(f'predictions/{model}/{data}_predictions.pkl', 'rb'))
            y_pred = y_predictions[scenario]['test']
            
            acc_test = results_models[results_models['id_scenario']==scenario]['acc_test'][scenario]
            acc_train_mean = results_models[results_models['id_scenario']==scenario]['acc_train_mean'][scenario]
            acc_train_sd = results_models[results_models['id_scenario']==scenario]['acc_train_sd'][scenario]
            log_loss_test = results_models[results_models['id_scenario']==scenario]['log_loss_test'][scenario]
            log_loss_train_mean = results_models[results_models['id_scenario']==scenario]['log_loss_train_mean'][scenario]
            log_loss_train_sd = results_models[results_models['id_scenario']==scenario]['log_loss_train_sd'][scenario]
            
            acc_test_all = list([acc_test]) *len(thresholds)
            acc_train_mean_all = list([acc_train_mean]) *len(thresholds)
            acc_train_sd_all = list([acc_train_sd]) *len(thresholds)
            log_loss_test_all = list([log_loss_test])*len(thresholds)
            log_loss_train_mean_all = list([log_loss_train_mean])*len(thresholds)
            log_loss_train_sd_all = list([log_loss_train_sd])*len(thresholds)
            scenario_all = list([scenario]) * len(thresholds)
            high_all = list(['High']) * len(thresholds)
            low_all = list(['Low']) * len(thresholds)
            eq_odd_all = list(['Equalized Odd']) * len(thresholds)
            eq_opp_all = list(['Equalized Opportunity']) * len(thresholds)
            dem_p_all = list(['Demographic Parity']) * len(thresholds)
            
            results_odd, results_groups_odd = obtain_results(
                                    S_test_original, 
                                    y_pred,
                                    Y_test,
                                    metric_pve,
                                    'equalized_odd')

            odd = {'id_scenario': scenario_all,
                    'threshold': thresholds,
                    'metric': eq_odd_all,
                    'value_metric': results_odd['metric'],
                    'acc_test': acc_test_all,
                    'acc_train_mean': acc_train_mean_all,
                    'acc_train_sd': acc_train_sd_all,
                    'log_loss_test': log_loss_test_all,
                    'log_loss_train_mean': log_loss_train_mean_all,
                    'log_loss_train_sd': log_loss_train_mean_all,
                    'portion_pos': portion_pos,
                    'portion_unpr': portion_unpr
                    }
            
            odd_group_high = {'id_scenario': scenario_all,
                    'threshold': thresholds_groups,
                    'metric': eq_odd_all,
                    'high_low': high_all,
                    'value_metric': results_groups_odd['metric_high'],
                    'acc_test': acc_test_all,
                    'acc_train_mean': acc_train_mean_all,
                    'acc_train_sd': acc_train_sd_all,
                    'log_loss_test': log_loss_test_all,
                    'log_loss_train_mean': log_loss_train_mean_all,
                    'log_loss_train_sd': log_loss_train_sd_all,
                    'portion_pos': portion_pos,
                    'portion_unpr': portion_unpr
                    }
            
            odd_group_low = {'id_scenario': scenario_all,
                    'threshold': thresholds_groups,
                    'metric': eq_odd_all,
                    'high_low': low_all,
                    'value_metric': results_groups_odd['metric_low'],
                    'acc_test': acc_test_all,
                    'acc_train_mean': acc_train_mean_all,
                    'acc_train_sd': acc_train_sd_all,
                    'log_loss_test': log_loss_test_all,
                    'log_loss_train_mean': log_loss_train_mean_all,
                    'log_loss_train_sd': log_loss_train_sd_all,
                    'portion_pos': portion_pos,
                    'portion_unpr': portion_unpr
                    }

            results_opp, results_groups_opp = obtain_results(
                                    S_test_original, 
                                    y_pred,
                                    Y_test,
                                    metric_pve, 
                                    'equalized_opportunity')
            
            opp = {'id_scenario': scenario_all,
                    'threshold': thresholds,
                    'metric': eq_opp_all,
                    'value_metric': results_opp['metric'],
                    'acc_test': acc_test_all,
                    'acc_train_mean': acc_train_mean_all,
                    'acc_train_sd': acc_train_sd_all,
                    'log_loss_test': log_loss_test_all,
                    'log_loss_train_mean': log_loss_train_mean_all,
                    'log_loss_train_sd': log_loss_train_sd_all,
                    'portion_pos': portion_pos,
                    'portion_unpr': portion_unpr
                    }

            opp_group_high = {'id_scenario': scenario_all,
                    'threshold': thresholds_groups,
                    'metric': eq_opp_all,
                    'high_low': high_all,
                    'value_metric': results_groups_opp['metric_high'],
                    'acc_test': acc_test_all,
                    'acc_train_mean': acc_train_mean_all,
                    'acc_train_sd': acc_train_sd_all,
                    'log_loss_test': log_loss_test_all,
                    'log_loss_train_mean': log_loss_train_mean_all,
                    'log_loss_train_sd': log_loss_train_sd_all,
                    'portion_pos': portion_pos,
                    'portion_unpr': portion_unpr
                    }
            
            opp_group_low = {'id_scenario': scenario_all,
                    'threshold': thresholds_groups,
                    'metric': eq_opp_all,
                    'high_low': low_all,
                    'value_metric': results_groups_opp['metric_low'],
                    'acc_test': acc_test_all,
                    'acc_train_mean': acc_train_mean_all,
                    'acc_train_sd': acc_train_sd_all,
                    'log_loss_test': log_loss_test_all,
                    'log_loss_train_mean': log_loss_train_mean_all,
                    'log_loss_train_sd': log_loss_train_sd_all,
                    'portion_pos': portion_pos,
                    'portion_unpr': portion_unpr
                    }

            results_demp, results_groups_demp = obtain_results(
                                    S_test_original, 
                                    y_pred,
                                    Y_test,
                                    metric_pve, 
                                    'demographic_parity')

            demp = {'id_scenario': scenario_all,
                    'threshold': thresholds,
                    'metric': dem_p_all,
                    'value_metric': results_demp['metric'],
                    'acc_test': acc_test_all,
                    'acc_train_mean': acc_train_mean_all,
                    'acc_train_sd': acc_train_sd_all,
                    'log_loss_test': log_loss_test_all,
                    'log_loss_train_mean': log_loss_train_mean_all,
                    'log_loss_train_sd': log_loss_train_sd_all,
                    'portion_pos': portion_pos,
                    'portion_unpr': portion_unpr
                    }
                    
            demp_group_high = {'id_scenario': scenario_all,
                    'threshold': thresholds_groups,
                    'metric': dem_p_all,
                    'high_low': high_all,
                    'value_metric': results_groups_demp['metric_high'],
                    'acc_test': acc_test_all,
                    'acc_train_mean': acc_train_mean_all,
                    'acc_train_sd': acc_train_sd_all,
                    'log_loss_test': log_loss_test_all,
                    'log_loss_train_mean': log_loss_train_mean_all,
                    'log_loss_train_sd': log_loss_train_sd_all,
                    'portion_pos': portion_pos,
                    'portion_unpr': portion_unpr
                    }
            
            dem_group_low = {'id_scenario': scenario_all,
                    'threshold': thresholds_groups,
                    'metric': dem_p_all,
                    'high_low': low_all,
                    'value_metric': results_groups_demp['metric_low'],
                    'acc_test': acc_test_all,
                    'acc_train_mean': acc_train_mean_all,
                    'acc_train_sd': acc_train_sd_all,
                    'log_loss_test': log_loss_test_all,
                    'log_loss_train_mean': log_loss_train_mean_all,
                    'log_loss_train_sd': log_loss_train_sd_all,
                    'portion_pos': portion_pos,
                    'portion_unpr': portion_unpr
                    }

            results_plots = pd.concat([results_plots, pd.DataFrame(odd), pd.DataFrame(opp), pd.DataFrame(demp)], axis=0, ignore_index=True)
            results_groups_plots = pd.concat([results_groups_plots, 
                                            pd.DataFrame(odd_group_high), pd.DataFrame(odd_group_low),
                                            pd.DataFrame(opp_group_high), pd.DataFrame(opp_group_low),
                                            pd.DataFrame(demp_group_high), pd.DataFrame(dem_group_low)], axis=0, ignore_index=True)

            results_plots.to_csv(f'pvi_results/{model}/{data}/{metric}_results_plots.txt')
            results_groups_plots.to_csv(f'pvi_results/{model}/{data}/{metric}_results_groups_plots.txt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbalize', action='store_true', help='Activate verbalization')
    parser.add_argument('--models', nargs='+', type=str, default= 'sgd_lr mlp_one_layer mlp_two_layer' , help='v-models to compute results')
    parser.add_argument('--datasets', nargs='+', type=str, default='compas census_income dutch_census', help='Datasets to compute results')
    parser.add_argument('--metrics',  nargs='+', type=str,  help='Metric to compute results for plotting')
    args = parser.parse_args()
    
    print('=================')
    print('COMPUTING RESULTS FOR PLOTTING')
    print(f'Models: {args.models}')
    print(f'Datasets: {args.datasets}')
    print(f'Metric: {args.metrics}')
    print('=================')
    print()
    if args.verbalize: print('Loading datasets')
    data = get_dataset(args)
    if args.verbalize: print()

    if args.verbalize: print('Computing results for plotting')
    for m in args.models:
        if args.verbalize: print(f'...Model: {m}')
        for d in args.datasets:
            if args.verbalize: print(f'......data: {d}', end='\r')
            count=1
            for met in args.metrics:
                if args.verbalize: print(f'......data: {d}...metric: {count}/{len(args.metrics)}   ', end='\r')
                compute_results_plots(m, d,data, met)
                count+=1
            if args.verbalize: print()
    
    if args.verbalize: print()
    if args.verbalize: print('Process complete!')
    if args.verbalize: print('=================')
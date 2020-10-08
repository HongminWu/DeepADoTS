import glob
import os
import numpy as np
import pandas as pd
from src.algorithms import Donut, AutoEncoder, DAGMM, RecurrentEBM, LSTMAD, LSTMED
from src.datasets import KittingExp
from src.evaluation import Evaluator
from data import process_kitting_dataset_to_fit_this_implementation

import ipdb

# RUNS = 1

def detectors(seed):
    standard_epochs = 50
    dets = [
            AutoEncoder(num_epochs=standard_epochs, seed=seed, details = False),
            LSTMAD(num_epochs=standard_epochs, seed=seed, details = False),
            LSTMED(num_epochs=standard_epochs, seed=seed, details = False),
            DAGMM(num_epochs=standard_epochs, seed=seed, lr=1e-4,details = False),
            DAGMM(num_epochs=standard_epochs, autoencoder_type=DAGMM.AutoEncoder.LSTM, seed=seed,details = False),
            # Donut(num_epochs = standard_epochs, seed=seed),
            # RecurrentEBM(num_epochs=standard_epochs, seed=seed)   
            ]

    return sorted(dets, key=lambda x: x.framework)

def evaluate_real_datasets(folder_name=None, skill=None, anomaly_region=None):
    # seeds = np.random.randint(np.iinfo(np.uint32).max, size=RUNS, dtype=np.uint32)
    seeds = [0]
    # results = pd.DataFrame()
    for seed in seeds:
        datasets = [KittingExp(seed, folder_name = folder_name, skill=skill)]
        evaluator = Evaluator(datasets, detectors, seed=seed)
        evaluator.evaluate()
        result = evaluator.benchmarks()
        if anomaly_region is None:
            file = open('./reports/logs/%s_result_skill_%d.txt'%(folder_name, skill),'w');
        else:
            file = open('./reports/logs/%s_result_skill_%d_%s.txt'%(folder_name, skill, anomaly_region),'w');            
        file.write(str(result));
        file.close();

        evaluator.save_roc_curves(skill=skill)
        
        # evaluator.plot_roc_curves()        
        # evaluator.plot_threshold_comparison()
        # evaluator.plot_scores()
        # results = results.append(result, ignore_index=True)

    # avg_results = results.groupby(['dataset', 'algorithm'], as_index=False).mean()
    # evaluator.set_benchmark_results(avg_results)
    # evaluator.export_results('run_real_datasets')
    # evaluator.create_boxplots(runs=RUNS, data=results, detectorwise=False)
    # evaluator.create_boxplots(runs=RUNS, data=results, detectorwise=True)

def test_modalities():        
    folder_names = ['fx+fy+fz',
                   'fx+fy+fz+sl+sr',
                   'fx+fy+fz+tx+ty+tz',
                   'fx+fy+fz+tx+ty+tz+lx+ly+lz+ax+ay+az',
                   'fx+fy+fz+tx+ty+tz+lx+ly+lz+ax+ay+az+nx+nt+nl+na+sl+sr',
                   'lx+ly+lz',
                   'lx+ly+lz+ax+ay+az',
                   'lx+ly+lz+sl+sr',
                   'nf+nl',
                   'nf+nl+sl+sr',
                   'nf+nt+nl+na+sl+sr',
                   'nt+na',
                   'sl+sr',
                   'tx+ty+tz+lx+ly+lz+ax+ay+az+sl+sr',
                   'tx+ty+tz+sl+sr',
                   ]
    anomaly_region = [-1.0, 1.0] # (s) default [-1, 1] observation located at the forward and backward region are marked as anomalies

    for folder_name in folder_names:
    
        process_kitting_dataset_to_fit_this_implementation.run(folder_name = folder_name, anomaly_region = anomaly_region)

        skills = [3, 4, 5, 7, 8, 9]
        for skill in skills:
            evaluate_real_datasets(folder_name = folder_name, skill = skill)
            
def test_anomaly_bias():
    folder_name = 'nf+nt+nl+na+sl+sr'
    anomaly_regions = [
                          [-1, 0],
                          [0,  1],
                          [-1, 1],  # (s) default [-1, 1] observation located at the forward and backward region are marked as anomalies
                          [-2, 0],
                          [0,  2],
                          [-2, 1],
                          [-2, 2],
                          [-3, 3],                      
                       ]
    
    for anomaly_region in anomaly_regions:
        process_kitting_dataset_to_fit_this_implementation.run(folder_name = folder_name, anomaly_region = anomaly_region)
        skills = [3, 4, 5, 7, 8, 9]
        for skill in skills:
            evaluate_real_datasets(folder_name = folder_name, skill = skill, anomaly_region=anomaly_region)

def test_optimal_modality_n_bias():
    folder_name = 'nf+nt+nl+na+sl+sr'
    anomaly_region =[0,  2]
    
    process_kitting_dataset_to_fit_this_implementation.run(folder_name = folder_name, anomaly_region = anomaly_region)
    skills = [5, 7]
    for skill in skills:
        evaluate_real_datasets(folder_name = folder_name, skill = skill, anomaly_region=anomaly_region)
            
            
if __name__ == '__main__':
#    test_modalities()
#    test_anomaly_bias()
    test_optimal_modality_n_bias()

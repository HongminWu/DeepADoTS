import glob
import os
import numpy as np
import pandas as pd
from src.algorithms import AutoEncoder, DAGMM, RecurrentEBM, LSTMAD, LSTMED
from src.datasets import KDDCup, RealPickledDataset
from src.evaluation import Evaluator
import ipdb

RUNS = 1

def detectors(seed):
    standard_epochs = 40
    dets = [AutoEncoder(num_epochs=standard_epochs, seed=seed),
            DAGMM(num_epochs=standard_epochs, seed=seed, lr=1e-4),
            DAGMM(num_epochs=standard_epochs, autoencoder_type=DAGMM.AutoEncoder.LSTM, seed=seed),
            LSTMAD(num_epochs=standard_epochs, seed=seed), LSTMED(num_epochs=standard_epochs, seed=seed),
            RecurrentEBM(num_epochs=standard_epochs, seed=seed)]

    return sorted(dets, key=lambda x: x.framework)

def evaluate_real_datasets():
    REAL_DATASET_GROUP_PATH = 'data/raw/'
    real_dataset_groups = glob.glob(REAL_DATASET_GROUP_PATH + '*')
    seeds = np.random.randint(np.iinfo(np.uint32).max, size=RUNS, dtype=np.uint32)
    results = pd.DataFrame()
    datasets = [KDDCup(seed=1)]
    for real_dataset_group in real_dataset_groups:
        for data_set_path in glob.glob(real_dataset_group + '/labeled/train/*'):
            data_set_name = data_set_path.split('/')[-1].replace('.pkl', '')
            dataset = RealPickledDataset(data_set_name, data_set_path)
            datasets.append(dataset)

    for seed in seeds:
        datasets[0] = KDDCup(seed)
        evaluator = Evaluator(datasets, detectors, seed=seed)
        evaluator.evaluate()
        result = evaluator.benchmarks()
        evaluator.plot_roc_curves()
        evaluator.plot_threshold_comparison()
        evaluator.plot_scores()
        results = results.append(result, ignore_index=True)

    avg_results = results.groupby(['dataset', 'algorithm'], as_index=False).mean()
    evaluator.set_benchmark_results(avg_results)
    evaluator.export_results('run_real_datasets')
    evaluator.create_boxplots(runs=RUNS, data=results, detectorwise=False)
    evaluator.create_boxplots(runs=RUNS, data=results, detectorwise=True)

if __name__ == '__main__':
    evaluate_real_datasets()

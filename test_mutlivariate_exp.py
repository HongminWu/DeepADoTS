import glob
import os

import numpy as np
import pandas as pd

from experiments import run_multivariate_experiment, announce_experiment
from src.algorithms import AutoEncoder, DAGMM, RecurrentEBM, LSTMAD, LSTMED
from src.datasets import KDDCup, RealPickledDataset
from src.evaluation import Evaluator

import ipdb

RUNS = 1


def main():
    run_experiments()

def detectors(seed):
    standard_epochs = 40
    dets = [AutoEncoder(num_epochs=standard_epochs, seed=seed),
            DAGMM(num_epochs=standard_epochs, seed=seed, lr=1e-4),
            DAGMM(num_epochs=standard_epochs, autoencoder_type=DAGMM.AutoEncoder.LSTM, seed=seed),
            LSTMAD(num_epochs=standard_epochs, seed=seed), LSTMED(num_epochs=standard_epochs, seed=seed),
            RecurrentEBM(num_epochs=standard_epochs, seed=seed)]

    return sorted(dets, key=lambda x: x.framework)

def run_experiments():
    # Set the seed manually for reproducibility.
    seeds = np.random.randint(np.iinfo(np.uint32).max, size=RUNS, dtype=np.uint32)
    output_dir = 'reports/experiments'
    evaluators = []

    announce_experiment('Multivariate Datasets')
    ev_mv = run_multivariate_experiment(
        detectors, seeds, RUNS,
        output_dir=os.path.join(output_dir, 'multivariate'))
    evaluators.append(ev_mv)

    for ev in evaluators:
        ev.plot_single_heatmap()

if __name__ == '__main__':
    main()

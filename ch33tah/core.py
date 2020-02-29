from experiment import Experiment
from dataset import Dataset
import numpy as np
import json
from distributed import Client
import time
import joblib
import os

np.random.seed(1)

class Core:

    def __init__(self, data, label, task, name, **kw):
        self.experiment = Experiment(Dataset(data, label), task)
        self.name = name
        os.makedirs(self.name, exist_ok=True)

    def ch33t(self):
        cli = Client()
        results = cli.submit(self.experiment.fit)
        results = cli.gather(results)
        time.sleep(0.5)
        return results

    def sav3(self, res):
        for mdl in list(res):
            tosave = res[mdl]['model']
            fname = f"{self.name}/{mdl}.sav"
            joblib.dump(tosave, fname)

if __name__ == '__main__':
    X = np.round(np.random.randn(10000,5), 3)
    y = (X[:,[0]] > 0).astype(int)
    X = np.hstack((X, y))
    
    data = Dataset(X, 5)
    core = Core(X, 5, 'classification', 'test')
    results = core.ch33t()
    core.sav3(results)
    print(results)


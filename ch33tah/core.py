from experiment import Experiment
from dataset import Dataset
import numpy as np
import json
from distributed import Client
import time
import joblib
import os
from resultsmanager import ResultsManager
import shutil 


np.random.seed(1)


class Ch33tah:

    def __init__(self, data, label, task, name, **kw):
        ''' get ready to run '''
        self.experiment = Experiment(Dataset(data, label), task)
        self.name = name
        os.makedirs(self.name, exist_ok=True)
        self.resmngr = ResultsManager()

    def ch33t(self):
        ''' kick off the distributed grid search CV '''
        cli = Client()
        results = cli.submit(self.experiment.fit)
        results = cli.gather(results)
        time.sleep(0.5)
        return results

    def sav3(self, res):
        ''' save the results of the good models. we are going to dump the weigths 
        to an S3 bucket of the user '''
        bucket_name = self.resmngr.create_bucket(self.name)
        self.bucket = bucket_name
        for mdl in list(res):
            tosave = res[mdl]['model']
            fname = f"{self.name}/{mdl}.sav"
            joblib.dump(tosave, fname)
        self.resmngr.upload_run(bucket_name, self.name)
        shutil.rmtree(self.name)

    def load_mod3ls(self, bucket=None):
        ''' load some models in from a previous run for inference '''
        bucket = self.bucket if bucket is None else bucket
        models = self.resmngr.reload_models(bucket)
        return models


if __name__ == '__main__':
    X = np.round(np.random.randn(10000,5), 3)
    y = (X[:,[0]] > 0).astype(int)
    X = np.hstack((X, y))
    
    data = Dataset(X, 5)
    core = Ch33tah(X, 5, 'classification', 'test')
    results = core.ch33t()
    core.sav3(results)
    models = core.load_mod3ls()
    print(models)


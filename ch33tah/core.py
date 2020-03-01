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
from model import Model 
from sklearn.metrics import accuracy_score, mean_squared_error

np.random.seed(1)

class Ch33tah:

    def __init__(self, data, label, task, name, **kw):
        ''' get ready to run '''
        self.experiment = Experiment(Dataset(data, label), task)
        self.name = name
        os.makedirs(self.name, exist_ok=True)
        self.resmngr = ResultsManager()
        self.bucket = None

    def ch33t(self):
        ''' kick off the distributed grid search CV '''
        cli = Client()
        results = cli.submit(self.experiment.fit)
        results = cli.gather(results)
        time.sleep(0.5)
        return results

    def save_models(self, res):
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

    def load_models(self, bucket=None):
        ''' load some models in from a previous run for inference '''
        bucket = self.bucket if bucket is None else bucket
        models = self.resmngr.reload_models(bucket)
        self.models = models
        return self.models

    def evaluate_test_set(self, X, y, m='all'):
        ''' get predictions on data. If no model is specified, do it for all. 
        otherwise only predict with self.models[m]. return y_hat, array '''
        preds = []
        if m != 'all':
            if not isinstance(m, list):
                m = [m]
        else:
            m = list(self.models)
        preds = {}
        for name, model in models.items():
            model = Model((model, None), self.experiment.task)
            y_hat = model.mdl.predict(X)
            perf = model.get_performance_metric(y, y_hat)
            if self.experiment.task == 'classification':
                perf = {'accuracy': perf}
            else:
                perf = {'mean_squared_err': perf}
            preds[name] = {'predictions': y_hat, 'performance': perf}
        return preds

class Ch33tahRetest:

    def __init__(self, bucket, task):
        ''' point me towards the s3 bucket holding your models '''
        self.bucket = bucket
        self.models = None
        self.resmngr = ResultsManager()
        self.task = task
        self.load_models()

    def load_models(self):
        ''' same as above '''
        models = self.resmngr.reload_models(self.bucket)
        self.models = models
        return self.models

    def evaluate_test_set(self, X, y, m='all'):
        ''' same as above again '''
        preds = []
        if m != 'all':
            if not isinstance(m, list):
                m = [m]
        else:
            m = list(self.models)
        preds = {}
        for name, model in models.items():
            model = Model((model, None), self.task)
            y_hat = model.mdl.predict(X)
            perf = model.get_performance_metric(y, y_hat)
            if self.task == 'classification':
                perf = {'accuracy': perf}
            else:
                perf = {'mean_squared_err': perf}
            preds[name] = {'predictions': y_hat, 'performance': perf}
        return preds

if __name__ == '__main__':
    X = np.round(np.random.randn(10000,5), 3)
    y = (X[:,[0]] > 0).astype(int)
    X = np.hstack((X, y))
    
    data = Dataset(X, 5)
    core = Ch33tah(X, 5, 'classification', 'test')
    results = core.ch33t()
    core.save_models(results)

    models = core.load_models()

    X2 = np.round(np.random.randn(10, 5), 3)
    y2 = (X2[:, [0]] > 0).astype(int)

    core_retest = Ch33tahRetest(core.bucket, 'classification')

    print(core_retest.evaluate_test_set(X2, y2))


import numpy as np
import os
import joblib
import shutil 
import time
from distributed import Client
from sklearn.metrics import accuracy_score, mean_squared_error
from ch33tah.experiment import Experiment
from ch33tah.dataset import Dataset
from ch33tah.resultsmanager import ResultsManager
from ch33tah.model import Model 

np.random.seed(1)

class Ch33tah:
    ''' main class for this project. How the user will interact with Ch33tah core. '''

    def __init__(self, data, label, task, name, **kw):
        ''' get ready to run '''
        self.experiment = Experiment(Dataset(data, label), task)
        self.name = name
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
        self.bucket = self.resmngr.create_bucket(self.name.lower()+"-"+self.experiment.task.lower())
        os.makedirs(self.name, exist_ok=True)
        for mdl in list(res):
            tosave = res[mdl]['model']
            fname = f"{self.name}/{mdl}.sav"
            joblib.dump(tosave, fname)
        self.resmngr.upload_run(self.bucket, self.name)
        shutil.rmtree(self.name)
        return self.bucket

    def load_models(self, bucket=None):
        ''' load some models in from a previous run for inference '''
        bucket = self.bucket if bucket is None else bucket
        models = self.resmngr.reload_models(bucket.lower())
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
    ''' class to reload previous models and use them for prediction '''

    def __init__(self, bucket, load=0):
        ''' point me towards the s3 bucket holding your models '''
        self.bucket = bucket
        self.models = None
        self.resmngr = ResultsManager()
        self.task = bucket.split('-')[1]
        if load:
            self.load_models()

    def load_models(self):
        ''' same as above '''
        self.models = self.resmngr.reload_models(self.bucket)
        return self.models.copy()

    def evaluate_test_set(self, dataset, m='all'):
        ''' same as above again '''
        preds = self.predict(dataset.x, m=m)
        y = dataset.y
        for p in preds:
            y_hat = preds[p]['predictions']
            if self.task == 'classification':
                perf = {'accuracy': accuracy_score(y, y_hat)}
            else:
                perf = {'mean_squared_err': mean_squared_error(y, y_hat)}
            preds[p].update(**{'performance': perf})
        return preds

    def predict(self, X, m='all'):
        if m != 'all':
            m = [m] if not isinstance(m, list) else m
        else:
            m = list(self.models)
        preds = {}
        for name, model in self.models.items():
            model = Model((model, None), self.task)
            y_hat = model.mdl.predict(X)
            preds[name] = {'predictions': y_hat}
        return preds

import pandas as pd
import numpy as np
import time
from datetime import datetime
import os
import json
from distributed import worker_client
import copy 
from ch33tah.dataset import Dataset
from ch33tah.model import Model, classification, regression

tmap = {
    'classification': classification,
    'regression': regression
}


class Experiment:
    
    def __init__(self, dataset, task):
        self.data = dataset
        self.task = task
        self.modelzoo = tmap[self.task]
        self.analytics = None

    def test_models(self):
        ''' run gridsearch on all models '''
        res = {}
        futures = []
        names = []
        with worker_client() as wc:
            for mdl_name, mdl in self.modelzoo.items():
                mdl = Model(mdl, self.task)
                futures.append(wc.submit(mdl.gridsearchcv, self.data))
                names.append(mdl_name)
            futures = wc.gather(futures)
        for mdl_name, mdl_res in zip(names, futures):
            res[mdl_name] = mdl_res 
        return res

    def analyze_results(self, res):
        ''' check out the results to see what is doing well '''
        scoremap = {}
        for mdl_name in res:
            model = res[mdl_name]
            for pi in model:
                pset = model[pi]
                model_ = pset['model']
                cvr = pset['cv_report']  # get the cross validation metrics
                cvr = [cvr[_]['accuracy'] for _ in cvr]
                avg_acc = np.mean(cvr)  
                scoremap[f"{mdl_name}-{pi}"] = [avg_acc, model_]
        scoremap = sorted(list(scoremap.items()), key=lambda x: x[-1][0], reverse=True)
        scoremap = {_[0]: _[1] for _ in scoremap}
        analytics = {}
        for sm in scoremap:
            mname, pi = sm.split('-')
            if mname not in analytics:
                params = res[mname][int(pi)]['params']
                analytics[mname] = {'model': scoremap[sm][-1], 'average_accuracy': round(scoremap[sm][0],3)}
        return analytics

    def fit(self):
        ''' fit models and detemine the winner '''
        res = self.test_models()
        analytics = self.analyze_results(res)
        self.analytics = copy.deepcopy(analytics)
        return analytics
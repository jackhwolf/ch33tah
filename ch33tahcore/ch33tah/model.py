import pandas as pd
import numpy as np
from sklearn import tree, svm, linear_model, metrics
import xgboost
import itertools as it
import copy
from distributed import worker_client

class classification:
    DecisionTreeClassifier_ = [
        tree.DecisionTreeClassifier,
        {'criterion': ['gini', 'entropy'], 'max_depth': np.append([None], np.arange(5, 105, 5))}
    ]
    SVC_ = [
        svm.SVC,
        {'C': np.array([1e-4, 1e-3, 1e-1, 1, 3, 6, 10]), 
         'kernel': ['rbf', 'linear', 'sigmoid', 'poly'],
         'gamma': np.array(['scale', 'auto', 0.001, 0.01, 0.1, 1])}
    ]
    LinearSVC_ = [
        svm.LinearSVC,
        {'C': np.array([1e-4, 1e-3, 1e-1, 1, 3, 6, 10]), 
         'penalty': ['l2'],
         'loss': ['hinge', 'squared_hinge']}
    ]
    SGDClassifier_ = [
        linear_model.SGDClassifier,
        {'loss': [ 'hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
         'penalty': ['l1', 'l2', 'elasticnet'],
         'alpha': np.array([1e-4, 1e-3, 1e-2]),
         'epsilon': np.array([0.001, 0.01])}
    ]
    Perceptron_ = [
        linear_model.Perceptron,
        {'penalty': ['l1', 'l2', 'elasticnet'],
         'alpha': np.array([1e-4, 1e-3, 1e-2]),
         'eta0': np.linspace(0.08, 1.2, 5)}]
    XGBClassifier_ = [
        xgboost.XGBClassifier,
        {'objective':['binary:logistic'],
         'learning_rate': [1e-5, 1e-4, 1e-3, 1e-2], 
         'max_depth': [6],
         'min_child_weight': [11],
         'silent': [1],
         'n_estimators': np.arange(5, 105, 5), 
         'missing':[np.nan]}]

class regression:
    pass

def _factory(cls):
    ''' turn the above declarations of models and grids into a dict to iter '''
    foo = [f for f in dir(cls) if f[0] != '_' and f[-1] == '_']
    return {f.strip('_'): getattr(cls, f) for f in foo}

classification = _factory(classification)
regression = _factory(regression)


class Model:
    ''' class to wrap all of our models for consistency'''

    def __init__(self, mdl, task, **kw):
        self.mdl, self.grid = mdl
        self.task = task
        self.exhaustive = kw.get('exhaustive', True)

    def reset_model(self, mdl):
        ''' give our wrapper another model to use '''
        self.mdl, self.grid = mdl

    def _param_iter(self):
        ''' iterate params '''
        keys = list(self.grid)
        vals = [self.grid[key] if isinstance(self.grid[key], (list, np.ndarray)) else [self.grid[key]] for key in keys]
        for i, foo in enumerate(it.product(*vals)):
            cpy = copy.deepcopy(foo)
            params = {keys[i]: cpy[i] for i, _ in enumerate(keys)}
            for p in params:
                try:
                    params[p] = float(params[p])
                    if params[p].is_integer():
                        params[p] = int(params[p])
                except:
                    pass
            if i > 2:
                break
            yield params

    def get_acc(self, y, y_hat):
        ''' report the accuracy '''
        if self.task == 'classification':
            acc = metrics.accuracy_score(y, y_hat)
        else:
            acc = metrics.mean_squared_error(y, y_hat)
        return acc
    
    def train_and_report(self, params, params_idx, data, fold, fold_idx):
        ''' distribute the training of one (param, CV fold) pair '''
        print(params_idx, fold_idx, "START")
        mdl = self.mdl(**params)
        trainx, trainy, testx, testy = data.tr_te_xy(fold)
        mdl.fit(trainx, trainy)
        y_hat = mdl.predict(testx)
        acc = self.get_acc(testy, y_hat)
        print(params_idx, fold_idx, "END")
        return params, params_idx, fold_idx, acc, mdl

    def gridsearchcv(self, data, **kw):
        ''' perform a distributed grid search CV on the data '''
        folds = data.get_cv_folds(kw.get('folds', 5))
        report = {}
        futures = []
        with worker_client() as wc:
            for pi, params in enumerate(self._param_iter()):
                for fi, fold in enumerate(folds):
                    futures.append(wc.submit(self.train_and_report, params, pi, data, fold, fi))
            futures = wc.gather(futures)
        for f in futures:
            param_entry = report.get(f[1], {"params": f[0], "model": f[-1], "cv_report": {}})
            fold_entry = {'accuracy': f[3]}
            param_entry["cv_report"][f[2]] = fold_entry
            report[f[1]] = param_entry
        return report

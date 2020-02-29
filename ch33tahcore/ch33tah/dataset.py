import pandas as pd 
import numpy as np 
import json 

class splitters:

    def random(*args, tr_perc=0.8):
        y = args[-1]
        k = int(y.shape[0] * tr_perc)
        mask = np.array([False] * y.shape[0])
        idx = np.random.permutation(np.arange(mask.size))[:k]
        mask[idx] = True
        return mask


class Dataset:

    def __init__(self, data, label, **kw):
        ''' create a new standard dataset 
        @params:
            - data: DataFrame or Ndarray
            - label: index to label column '''
        self.label = label
        self.splitters = splitters
        if isinstance(data, str):
            if data.split('.')[-1] == '.npy':
                data = np.load(data)
            elif data.split('.')[-1] == 'csv':
                data = pd.read_csv(data)
            else:
                raise ValueError("data must be mat, .npy, or .csv")
        if isinstance(data, pd.core.frame.DataFrame):
            if isinstance(data.columns[0], str):
                if isinstance(label, (int, float)):
                    label = data.columns.tolist()[int(label)]
                elif isinstance(label, str):
                    label = data.columns.tolist().index(label)
            data = data.to_numpy()
        self.y = data[:, label]
        self.x = np.delete(data, [label], axis=1)

    def tr_xy(self, mask):
        ''' get the training data '''
        return self.x[mask], self.y[mask]

    def te_xy(self, mask):
        ''' get the testing data '''
        return self.x[~mask], self.y[~mask]

    def tr_te_xy(self, mask):
        ''' util '''
        return [*self.tr_xy(mask), *self.te_xy(mask)]

    def get_cv_folds(self, k, **kw):
        ''' get some folds '''
        folds = []
        for i in range(k):
            mask = getattr(self.splitters, kw.get('method', 'random'))(self.x, self.y)
            folds.append(mask.copy())
        return folds

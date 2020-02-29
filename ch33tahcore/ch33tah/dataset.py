import pandas as pd 
import numpy as np 
import json 


class splitters:

    def random(*args, tr_perc=0.8):
        y = args[-1]
        k = int(y.shape[-1] * tr_perc)
        mask = np.array([False] * y.shape[1])
        idx = np.random.permutation(np.arange(mask.size))[:k]
        mask[idx] = True
        return mask

    def balanced(*args, tr_perc=0.8):
        y = args[-1]
        mask = np.array([False] * y.size)
        y = pd.DataFrame(y)
        unq = y.value_counts()

class Dataset:

    def __init__(self, data, label, desc):
        ''' create a new standard dataset 
        @params:
            - data: DataFrame or Ndarray
            - label: index to label column
            - desc: str, description of this dataset '''
        self.desc = desc
        if isinstance(data, pd.core.frame.DataFrame):
            if isinstance(data.columns[0], str):
                if isinstance(label, (int, float)):
                    label = data.columns.tolist()[int(label)]
                elif isinstance(label, str):
                    label = data.columns.tolist().index(label)
            data = data.to_numpy()
        self.y = data[:, [label]]
        self.x = np.delete(data, [label], axis=1)
        self.label = label
        self.training_mask = None

    @property
    def tr_xy(self):
        ''' get the training data '''
        return self.x[self.training_mask], self.y[self.training_mask]

    @property
    def te_xy(self):
        ''' get the testing data '''
        return self.x[~self.training_mask], self.y[~self.training_mask]

    def set_training_mask(self, method):
        ''' set the training mask for this dataset '''
        mask = getattr(splitters, method)(self.x, self.y)
        self.training_mask = mask


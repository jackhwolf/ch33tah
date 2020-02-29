import pandas as pd
import numpy as np

class BaseModel:
    ''' class to wrap all of our models in to make sure 
    they have fit, predict '''
    
    def __init__(self, t, fitFunc=None, predictFunc=None):
        self.model = T_MAP[t]
        self.fitFunc = fitFunc
        self.predictFunc = predictFunc
        
    def fit(self, x, y, *args, **kw):
        if hasattr(self.model, 'fit'):
            return self.model.fit(x, y, *args, **kw)
        else:
            return getattr(self.model, self.fitFunc)(x, y, *args, **kw)
        
    def predict(self, x, *args, **kw):
        if hasattr(self.model, 'predict'):
            return self.model.predict(y, *args, **kw)
        else:
            return getattr(self.model, self.predictFunc)(x, *args, **kw)

        

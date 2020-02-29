import pandas as pd
import numpy as np
import time
from datetime import datetime
import os
import json
import model

class Experiment:
    
    def __init__(self, dataset, *models_and_params):
        self.data = dataset
        self.to_test = models_and_params
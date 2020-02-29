import pandas as pd
import numpy as np
import time
from datetime import datetime
import os
import json
import model

self.task_map = {
    'classification': [],
    'regression': []
}

class Experiment:
    
    def __init__(self, dataset, task):
        self.data = dataset
        self.task = task
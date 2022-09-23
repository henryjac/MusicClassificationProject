import pandas as pd
import random
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import statistics as s
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

# ------------------- IMPORT DATA ----------------
data = pd.read_csv (r'data/project_train.csv', encoding='utf-8') # Import data



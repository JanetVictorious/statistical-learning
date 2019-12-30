# -----------------------------------------
# Chapter 6 - Lab2: Ridge regression and the Lasso
# -----------------------------------------


#%% -----------------------------------------
# Import packages
# -------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

# Regression libs
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import resample
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.base import BaseEstimator, RegressorMixin

# Sampling lib
# from random import seed
# from random import random
# from random import gauss
from numpy.random import seed
from numpy.random import rand
from numpy.random import randn
from numpy.random import randint



#%% -----------------------------------------
# Settings
# -------------------------------------------

# Set working directory
os.getcwd()
os.chdir(os.getcwd()+'/'+'miscellanious/stanford-statistical-learning')
os.getcwd()

del os

# Plot settings
sns.set()


#%% -----------------------------------------
# Load data
# -------------------------------------------

# Datasets from ISLR
default = pd.read_csv('data/default.csv')
default.dtypes
default.head()

weekly = pd.read_csv('data/weekly.csv')
weekly.dtypes
weekly.head()

boston = pd.read_csv('data/boston.csv')
boston.dtypes
boston.head()


#%% -----------------------------------------
# Functions
# -------------------------------------------


#%% -----------------------------------------
# Exercises
# -------------------------------------------

# ------------------------
#### Question 5 ####
# ------------------------
#### (a - c) ####

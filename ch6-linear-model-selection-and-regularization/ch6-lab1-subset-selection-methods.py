# -----------------------------------------
# Chapter 6 - Lab1: Subset selection methods
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

import itertools
import time

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
hitters = pd.read_csv('data/hitters.csv')
hitters.dtypes
hitters.head()

# A data frame with 322 observations of major league players on the following 20 variables.

# AtBat
# Number of times at bat in 1986

# Hits
# Number of hits in 1986

# HmRun
# Number of home runs in 1986

# Runs
# Number of runs in 1986

# RBI
# Number of runs batted in in 1986

# Walks
# Number of walks in 1986

# Years
# Number of years in the major leagues

# CAtBat
# Number of times at bat during his career

# CHits
# Number of hits during his career

# CHmRun
# Number of home runs during his career

# CRuns
# Number of runs during his career

# CRBI
# Number of runs batted in during his career

# CWalks
# Number of walks during his career

# League
# A factor with levels A and N indicating player's league at the end of 1986

# Division
# A factor with levels E and W indicating player's division at the end of 1986

# PutOuts
# Number of put outs in 1986

# Assists
# Number of assists in 1986

# Errors
# Number of errors in 1986

# Salary
# 1987 annual salary on opening day in thousands of dollars

# NewLeague
# A factor with levels A and N indicating player's league at the beginning of 1987


#%% -----------------------------------------
# Functions
# -------------------------------------------

def processSubset(feature_set):
    # Fit model on feature_set and calculate RSS
    features = list(feature_set)
    model = sm.OLS(y,X[list(feature_set)])
    regr = model.fit()
    RSS = ((regr.predict(X[list(feature_set)]) - y) ** 2).sum()
    return {'model':regr, 'RSS':RSS, 'variables':features}

def getBest(k):
    tic = time.time()
    results = []
    for combo in itertools.combinations(X.columns, k):
        results.append(processSubset(combo))
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    toc = time.time()
    print("Processed", models.shape[0], "models on", k, "predictors in", (toc-tic), "seconds.")
    # Return the best model, along with some other useful information about the model
    return best_model


#%% -----------------------------------------
# Best Subset Selection
# Here we apply the best subset selection approach to the Hitters data. We wish to predict
# a baseball player’s Salary on the basis of various statistics associated with 
# performance in the previous year.
# -------------------------------------------

hitters['Salary'].describe()
print('Dimensions of original data: ', hitters.shape)
print('Number of null values: ', hitters['Salary'].isna().sum())

# Remove rows where salary is missing
hitters2 = hitters.dropna()
print('Dimensions of cleaned data: ', hitters2.shape)
print('Number of null values: ', hitters2['Salary'].isna().sum())

hitters2.head(10)

# Create dummy variables for categorical predictors
dummies = pd.get_dummies(hitters2[['League', 'Division', 'NewLeague']])

# Create new dataframe with predictiors and response
y = hitters2['Salary']
X_ = hitters2.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1)
X = pd.concat([X_, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)

# Best subset selection
models_best = pd.DataFrame(columns=['RSS', 'model', 'variables'])

tic = time.time()
for i in range(1, 8):
    models_best.loc[i] = getBest(i)

toc = time.time()
print('Total elapsed time:', (toc-tic)/60, 'minutes.')

models_best

for i in range(1,8):
    print(models_best.loc[i, 'model'].summary())

# Add R2, AIC and BIC to dataframe
models_best = models_best.assign(
    r2 = lambda x: x['model'].map(lambda y: y.rsquared),
    r2_adj = lambda x: x['model'].map(lambda y: y.rsquared_adj),
    aic = lambda x: x['model'].map(lambda y: y.aic),
    bic = lambda x: x['model'].map(lambda y: y.bic)
)

# Plot RSS, R2, AIC and BIC
plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size': 18, 'lines.markersize': 10})

plt.subplot(2, 2, 1)
sns.lineplot(x=models_best.index, y='r2', data=models_best)
plt.xlabel('# predictors')
plt.ylabel('R2')

plt.subplot(2, 2, 2)
sns.lineplot(x=models_best.index, y='r2_adj', data=models_best)
plt.xlabel('# predictors')
plt.ylabel('R2 adj')

plt.subplot(2, 2, 3)
sns.lineplot(x=models_best.index, y='aic', data=models_best)
plt.xlabel('# predictors')
plt.ylabel('AIC')

plt.subplot(2, 2, 4)
sns.lineplot(x=models_best.index, y='bic', data=models_best)
plt.xlabel('# predictors')
plt.ylabel('BIC')


#%% -----------------------------------------
# Forward and Backward Stepwise Selection
# -------------------------------------------
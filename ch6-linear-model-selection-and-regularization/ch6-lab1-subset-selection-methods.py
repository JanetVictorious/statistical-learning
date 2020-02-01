# -----------------------------------------
# Chapter 6 - Lab1: Subset selection methods
# -----------------------------------------

# Solution to lab is taken from http://www.science.smith.edu/~jcrouser/SDS293/labs/lab8-py.html

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

"""
A data frame with 322 observations of major league players on the following 20 variables.

AtBat
Number of times at bat in 1986

Hits
Number of hits in 1986

HmRun
Number of home runs in 1986

Runs
Number of runs in 1986

RBI
Number of runs batted in in 1986

Walks
Number of walks in 1986

Years
Number of years in the major leagues

CAtBat
Number of times at bat during his career

CHits
Number of hits during his career

CHmRun
Number of home runs during his career

CRuns
Number of runs during his career

CRBI
Number of runs batted in during his career

CWalks
Number of walks during his career

League
A factor with levels A and N indicating player's league at the end of 1986

Division
A factor with levels E and W indicating player's division at the end of 1986

PutOuts
Number of put outs in 1986

Assists
Number of assists in 1986

Errors
Number of errors in 1986

Salary
1987 annual salary on opening day in thousands of dollars

NewLeague
A factor with levels A and N indicating player's league at the beginning of 1987
"""


#%% -----------------------------------------
# Functions
# -------------------------------------------

# Mean squared error
mse = make_scorer(mean_squared_error)

def processSubset(feature_set):
    # Fit model on feature_set and calculate RSS
    features = list(feature_set)
    # X - predictors
    # y - Salary (continuous)
    model = sm.OLS(y, X[list(feature_set)])
    regr = model.fit()
    RSS = ((regr.predict(X[list(feature_set)]) - y) ** 2).sum()
    return {'model':regr, 'RSS':RSS, 'variables':features}

def processSubsetGLM(feature_set):
    # Fit model on feature_set and calculate RSS
    features = list(feature_set)
    # X - predictors
    # y2 - binary variable indicating above or below median salary
    model = sm.GLM(y2, X[list(feature_set)], family=sm.families.Binomial())
    regr = model.fit()
    # Instead of RSS we use the deviance for logistic regression
    RSS = regr.deviance
    return {'model':regr, 'RSS':RSS, 'variables':features}

def processSubsetCrossValidation(feature_set):
    # Fit model on feature_set and calculate RSS
    features = list(feature_set)
    # X - predictors
    # y - Salary (continuous)
    model = sm.OLS(y, X[list(feature_set)])
    regr = model.fit()
    # 5 fold cross validation
    cv_results1 = cross_val_score(LinearRegression(), X[list(feature_set)], y, scoring=mse, cv=10)
    MSE = cv_results1.mean()
    return {'model':regr, 'MSE':MSE, 'variables':features}

# Best subset selection function
def getBest(k, model):
    tic = time.time()
    results = []
    for combo in itertools.combinations(X.columns, k):
        results.append(processSubset(combo)) if model == 'OLS' else results.append(processSubsetGLM(combo))
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    toc = time.time()
    print("Processed", models.shape[0], "models on", k, "predictors in", (toc-tic), "seconds.")
    # Return the best model, along with some other useful information about the model
    return best_model

def getBestCrossValidation(k):
    tic = time.time()
    results = []
    for combo in itertools.combinations(X.columns, k):
        results.append(processSubsetCrossValidation(combo))
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    # Choose the model with the highest RSS
    best_model = models.loc[models['MSE'].argmin()]
    toc = time.time()
    print("Processed", models.shape[0], "models on", k, "predictors in", (toc-tic), "seconds.")
    # Return the best model, along with some other useful information about the model
    return best_model

# Forward stepwise selection function
def forward(predictors, model):
    # Pull out predictors we still need to process
    remaining_predictors = [p for p in X.columns if p not in predictors]
    tic = time.time()
    results = []
    for p in remaining_predictors:
        results.append(processSubset(predictors+[p])) if model == 'OLS' else results.append(processSubsetGLM(predictors+[p]))
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors)+1, "predictors in", (toc-tic), "seconds.")
    # Return the best model, along with some other useful information about the model
    return best_model

def forwardCrossValidation(predictors):
    # Pull out predictors we still need to process
    remaining_predictors = [p for p in X.columns if p not in predictors]
    tic = time.time()
    results = []
    for p in remaining_predictors:
        results.append(processSubsetCrossValidation(predictors+[p]))
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    # Choose the model with the highest RSS
    best_model = models.loc[models['MSE'].argmin()]
    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors)+1, "predictors in", (toc-tic), "seconds.")
    # Return the best model, along with some other useful information about the model
    return best_model

# Backward stepwise selection function
def backward(predictors, model):
    tic = time.time()
    results = []
    for combo in itertools.combinations(predictors, len(predictors)-1):
        results.append(processSubset(combo)) if model == 'OLS' else results.append(processSubsetGLM(combo))
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors)-1, "predictors in", (toc-tic), "seconds.")
    # Return the best model, along with some other useful information about the model
    return best_model

def backwardCrossValidation(predictors):
    tic = time.time()
    results = []
    for combo in itertools.combinations(predictors, len(predictors)-1):
        results.append(processSubsetCrossValidation(combo))
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    # Choose the model with the highest RSS
    best_model = models.loc[models['MSE'].argmin()]
    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors)-1, "predictors in", (toc-tic), "seconds.")
    # Return the best model, along with some other useful information about the model
    return best_model

#%% -----------------------------------------
# Data prepping
# -------------------------------------------

hitters['Salary'].describe()
print('Dimensions of original data: ', hitters.shape)
print('Number of null values: ', hitters['Salary'].isna().sum())

# Remove rows where salary is missing
hitters2 = hitters.dropna()
print('Dimensions of cleaned data: ', hitters2.shape)
print('Number of null values: ', hitters2['Salary'].isna().sum())

hitters2.head(10)

avg_sal = hitters2['Salary'].mean()
med_sal = hitters2['Salary'].median()

# Salary distribution
sns.distplot(hitters2['Salary'])
plt.axvline(x=avg_sal, color='r')
plt.axvline(x=med_sal, color='b')
plt.legend(['mean', 'median'])

# Add variables indicating above or below mean/median salary
hitters2 = hitters2.assign(
    avg_sal = lambda x: x['Salary'].map(lambda y: 0 if y < avg_sal else 1),
    med_sal = lambda x: x['Salary'].map(lambda y: 0 if y < med_sal else 1)
)

# Create dummy variables for categorical predictors
dummies = pd.get_dummies(hitters2[['League', 'Division', 'NewLeague']])

# Create new dataframe with predictiors and response
y = hitters2['Salary']
y2 = hitters2['med_sal']
X_ = hitters2.drop(['Salary', 'League', 'Division', 'NewLeague', 'avg_sal', 'med_sal'], axis=1)
X = pd.concat([X_, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
X2 = pd.concat([X, hitters2[['avg_sal', 'med_sal']]], axis=1)

# Pairplot labeled on median salary
plt.figure(figsize=(20,10))
sns.pairplot(data=X2[['Hits', 'CRBI', 'Division_W', 'PutOuts', 'AtBat', 'Walks', 'CAtBat', 'CHits', 'CHmRun', 'med_sal']], hue='med_sal')


#%% -----------------------------------------
# Best Subset Selection
# -------------------------------------------
"""
Here we apply the best subset selection approach to the Hitters data. We wish to predict
a baseball player’s Salary on the basis of various statistics associated with 
performance in the previous year.

Two approaches are implemented:
(i) OLS to predict Salary
(ii) Logistic regression to predict wether or not a player will have a salary above the median salary
"""

# ------------------------
# (i) Subset selection OLS
# ------------------------

# Best subset selection OLS
models_best = pd.DataFrame(columns=['RSS', 'model', 'variables'])

tic = time.time()
for i in range(1, 8):
    models_best.loc[i] = getBest(i, 'OLS')

toc = time.time()
print('Total elapsed time:', (toc-tic)/60, 'minutes.')

# Add R2, AIC and BIC to dataframe
models_best = models_best.assign(
    r2 = lambda x: x['model'].map(lambda y: y.rsquared),
    r2_adj = lambda x: x['model'].map(lambda y: y.rsquared_adj),
    aic = lambda x: x['model'].map(lambda y: y.aic),
    bic = lambda x: x['model'].map(lambda y: y.bic)
)

# Print summary of models
for i in range(1,8):
    print(models_best.loc[i, 'model'].summary())

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

# BIC suggests a model with 6 predictors. R2, R2 adj. and AIC suggests a more 
# complex model.


# ------------------------
# (ii) Subset selection logistic regression
# ------------------------

# The logistic regression is used to predict wether or not the salary is above or below the median salary

# Best subset selection GLM
models_best_glm = pd.DataFrame(columns=['RSS', 'model', 'variables'])

tic = time.time()
for i in range(1, 8):
    models_best_glm.loc[i] = getBest(i, 'GLM')

toc = time.time()
print('Total elapsed time:', (toc-tic)/60, 'minutes.')

# Create ROC dataframe for each GLM model
roc_df = pd.DataFrame(columns=['fpr', 'tpr', 'threshold', 'i'])

for i in range(1, 8):
    lr_fpr1, lr_tpr1, thresholds1 = roc_curve(y2, models_best_glm.loc[i, 'model'].predict(X[list(models_best_glm.loc[i, 'variables'])]))
    it = [models_best_glm.loc[i, 'variables'] for x in range(len(lr_fpr1))]
    df = pd.DataFrame({'fpr':lr_fpr1, 'tpr':lr_tpr1, 'threshold':thresholds1, 'i':it})
    roc_df = roc_df.append(df)

# Create variable to lable on in lineplot
roc_df['hue'] = ['$%s$' % x for x in roc_df['i']]

# Extract optimal threshold for each model
roc_df = roc_df.assign(
    tf = lambda x: (x['fpr']-(1-x['tpr'])).abs(),
)

# Add optimal threshold to models dataframe
min_tf = roc_df.groupby(['hue'])['tf'].min().reset_index()

# Make sure same tf does not occur in two different models
min_tf_per_hue = pd.merge(left=roc_df, right=min_tf, left_on=['hue', 'tf'], right_on=['hue', 'tf'])

models_best_glm['optimal_threshold'] = np.asarray(min_tf_per_hue[['threshold']])

# Add AUC for each model
models_best_glm['auc'] = [roc_auc_score(y2, models_best_glm.loc[i, 'model'].predict(X[list(models_best_glm.loc[i, 'variables'])])) for i in models_best_glm.index]

# Print results for AUC
for i in range(1, 8):
    print('Logistic model %s has AUC: %f, with predictor(s) %s' % (i, models_best_glm.loc[i, 'auc'], models_best_glm.loc[i, 'variables']))

# Print summary of models
for i in range(1,8):
    print(models_best_glm.loc[i, 'model'].summary())

# Plot ROC curves
plt.figure(figsize=(14,10))
sns.lineplot(x='fpr', y='tpr', data=roc_df, hue='hue', ci=None)
plt.plot([x for x in np.arange(0, 1.1, 0.1)], [x for x in np.arange(0, 1.1, 0.1)], 'b--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# The logistic regression peforms best using only one predictor, namely CRBI, if we look at AUC.


#%% -----------------------------------------
# Forward and Backward Stepwise Selection
# -------------------------------------------
"""
Two approaches are implemented:
(i) OLS to predict Salary
(ii) Logistic regression to predict wether or not a player will have a salary above the median salary
"""
# ------------------------
# (i) Forward selection OLS
# ------------------------

models_fwd = pd.DataFrame(columns=['RSS', 'model', 'variables'])

tic = time.time()
predictors = []

for i in range(1,len(X.columns)+1):    
    models_fwd.loc[i] = forward(predictors, 'OLS')
    predictors = models_fwd.loc[i]["model"].model.exog_names

toc = time.time()
print('Total elapsed time:', (toc-tic)/60, 'minutes.')

# Add R2, AIC and BIC to dataframe
models_fwd = models_fwd.assign(
    r2 = lambda x: x['model'].map(lambda y: y.rsquared),
    r2_adj = lambda x: x['model'].map(lambda y: y.rsquared_adj),
    aic = lambda x: x['model'].map(lambda y: y.aic),
    bic = lambda x: x['model'].map(lambda y: y.bic)
)

# Print summary of models
for i in range(1,len(X.columns)+1):
    print(models_fwd.loc[i, 'model'].summary())

# Plot RSS, R2, AIC and BIC
plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size': 18, 'lines.markersize': 10})

plt.subplot(2, 2, 1)
sns.lineplot(x=models_fwd.index, y='r2', data=models_fwd)
plt.xlabel('# predictors')
plt.ylabel('R2')

plt.subplot(2, 2, 2)
sns.lineplot(x=models_fwd.index, y='r2_adj', data=models_fwd)
plt.xlabel('# predictors')
plt.ylabel('R2 adj')

plt.subplot(2, 2, 3)
sns.lineplot(x=models_fwd.index, y='aic', data=models_fwd)
plt.xlabel('# predictors')
plt.ylabel('AIC')

plt.subplot(2, 2, 4)
sns.lineplot(x=models_fwd.index, y='bic', data=models_fwd)
plt.xlabel('# predictors')
plt.ylabel('BIC')


# ------------------------
# (ii) Forward selection logistic regression
# ------------------------

models_fwd_glm = pd.DataFrame(columns=['RSS', 'model', 'variables'])

tic = time.time()
predictors = []

for i in range(1,len(X.columns)+1):    
    models_fwd_glm.loc[i] = forward(predictors, 'GLM')
    predictors = models_fwd_glm.loc[i]["model"].model.exog_names

toc = time.time()
print('Total elapsed time:', (toc-tic)/60, 'minutes.')

# Create ROC dataframe for each GLM model
roc_fwd_df = pd.DataFrame(columns=['fpr', 'tpr', 'threshold', 'i'])

for i in range(1,len(X.columns)+1):
    lr_fpr1, lr_tpr1, thresholds1 = roc_curve(y2, models_fwd_glm.loc[i, 'model'].predict(X[list(models_fwd_glm.loc[i, 'variables'])]))
    it = [models_fwd_glm.loc[i, 'variables'] for x in range(len(lr_fpr1))]
    df = pd.DataFrame({'fpr':lr_fpr1, 'tpr':lr_tpr1, 'threshold':thresholds1, 'i':it})
    roc_fwd_df = roc_fwd_df.append(df)

# Create variable to lable on in lineplot
roc_fwd_df['hue'] = ['$%s$' % x for x in roc_fwd_df['i']]

# Extract optimal threshold for each model
roc_fwd_df = roc_fwd_df.assign(
    tf = lambda x: (x['fpr']-(1-x['tpr'])).abs(),
)

# Add optimal threshold to models dataframe
min_tf = roc_fwd_df.groupby(['hue'])['tf'].min().reset_index()

# Make sure same tf does not occur in two different models
min_tf_per_hue = pd.merge(left=roc_fwd_df, right=min_tf, left_on=['hue', 'tf'], right_on=['hue', 'tf'])

models_fwd_glm['optimal_threshold'] = np.asarray(min_tf_per_hue[['threshold']])

# Add AUC for each model
models_fwd_glm['auc'] = [roc_auc_score(y2, models_fwd_glm.loc[i, 'model'].predict(X[list(models_fwd_glm.loc[i, 'variables'])])) for i in models_fwd_glm.index]

# Print results for AUC
for i in range(1,len(X.columns)+1):
    print('Logistic model %s has AUC: %f, with predictor(s) %s' % (i, models_fwd_glm.loc[i, 'auc'], models_fwd_glm.loc[i, 'variables']))

# Print summary of models
for i in range(1,len(X.columns)+1):
    print(models_fwd_glm.loc[i, 'model'].summary())

# Plot ROC curves
plt.figure(figsize=(14,10))
sns.lineplot(x='fpr', y='tpr', data=roc_fwd_df, hue='hue', ci=None)
plt.plot([x for x in np.arange(0, 1.1, 0.1)], [x for x in np.arange(0, 1.1, 0.1)], 'b--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')


# ------------------------
# (i) Backward selection OLS
# ------------------------

models_bwd = pd.DataFrame(columns=['RSS', 'model', 'variables'], index = range(1,len(X.columns)))

tic = time.time()
predictors = X.columns

while(len(predictors) > 1):  
    models_bwd.loc[len(predictors)-1] = backward(predictors, 'OLS')
    predictors = models_bwd.loc[len(predictors)-1]['model'].model.exog_names

toc = time.time()
print('Total elapsed time:', (toc-tic)/60, 'minutes.')

# Add R2, AIC and BIC to dataframe
models_bwd = models_bwd.assign(
    r2 = lambda x: x['model'].map(lambda y: y.rsquared),
    r2_adj = lambda x: x['model'].map(lambda y: y.rsquared_adj),
    aic = lambda x: x['model'].map(lambda y: y.aic),
    bic = lambda x: x['model'].map(lambda y: y.bic)
)

# Print summary of models
for i in range(1,len(X.columns)):
    print(models_bwd.loc[i, 'model'].summary())

# Plot RSS, R2, AIC and BIC
plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size': 18, 'lines.markersize': 10})

plt.subplot(2, 2, 1)
sns.lineplot(x=models_bwd.index, y='r2', data=models_bwd)
plt.xlabel('# predictors')
plt.ylabel('R2')

plt.subplot(2, 2, 2)
sns.lineplot(x=models_bwd.index, y='r2_adj', data=models_bwd)
plt.xlabel('# predictors')
plt.ylabel('R2 adj')

plt.subplot(2, 2, 3)
sns.lineplot(x=models_bwd.index, y='aic', data=models_bwd)
plt.xlabel('# predictors')
plt.ylabel('AIC')

plt.subplot(2, 2, 4)
sns.lineplot(x=models_bwd.index, y='bic', data=models_bwd)
plt.xlabel('# predictors')
plt.ylabel('BIC')


# ------------------------
# (ii) Backward selection logistic regression
# ------------------------

models_bwd_glm = pd.DataFrame(columns=['RSS', 'model', 'variables'], index = range(1,len(X.columns)))

tic = time.time()
predictors = X.columns

while(len(predictors) > 1):  
    models_bwd_glm.loc[len(predictors)-1] = backward(predictors, 'GLM')
    predictors = models_bwd_glm.loc[len(predictors)-1]['model'].model.exog_names

toc = time.time()
print('Total elapsed time:', (toc-tic)/60, 'minutes.')

# Create ROC dataframe for each GLM model
roc_bwd_df = pd.DataFrame(columns=['fpr', 'tpr', 'threshold', 'i'])

for i in range(1,len(X.columns)):
    lr_fpr1, lr_tpr1, thresholds1 = roc_curve(y2, models_bwd_glm.loc[i, 'model'].predict(X[list(models_bwd_glm.loc[i, 'variables'])]))
    it = [models_bwd_glm.loc[i, 'variables'] for x in range(len(lr_fpr1))]
    df = pd.DataFrame({'fpr':lr_fpr1, 'tpr':lr_tpr1, 'threshold':thresholds1, 'i':it})
    roc_bwd_df = roc_bwd_df.append(df)

# Create variable to lable on in lineplot
roc_bwd_df['hue'] = ['$%s$' % x for x in roc_bwd_df['i']]

# Extract optimal threshold for each model
roc_bwd_df = roc_bwd_df.assign(
    tf = lambda x: (x['fpr']-(1-x['tpr'])).abs(),
)

# Add optimal threshold to models dataframe
min_tf = roc_bwd_df.groupby(['hue'])['tf'].min().reset_index()

# Make sure same tf does not occur in two different models
min_tf_per_hue = pd.merge(left=roc_bwd_df, right=min_tf, left_on=['hue', 'tf'], right_on=['hue', 'tf'])

models_bwd_glm['optimal_threshold'] = np.asarray(min_tf_per_hue[['threshold']])

# Add AUC for each model
models_bwd_glm['auc'] = [roc_auc_score(y2, models_bwd_glm.loc[i, 'model'].predict(X[list(models_bwd_glm.loc[i, 'variables'])])) for i in models_bwd_glm.index]

# Print results for AUC
for i in range(1,len(X.columns)):
    print('Logistic model %s has AUC: %f, with predictor(s) %s' % (i, models_bwd_glm.loc[i, 'auc'], models_bwd_glm.loc[i, 'variables']))

# Print summary of models
for i in range(1,len(X.columns)):
    print(models_bwd_glm.loc[i, 'model'].summary())

# Plot ROC curves
plt.figure(figsize=(14,10))
sns.lineplot(x='fpr', y='tpr', data=roc_bwd_df, hue='hue', ci=None)
plt.plot([x for x in np.arange(0, 1.1, 0.1)], [x for x in np.arange(0, 1.1, 0.1)], 'b--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')


#%% -----------------------------------------
# Best Subset Selection with cross validation
# -------------------------------------------

# ------------------------
# (i) Subset selection OLS
# ------------------------

# Best subset selection OLS
models_best_cv = pd.DataFrame(columns=['MSE', 'model', 'variables'])

tic = time.time()
for i in range(1, 6):
    models_best_cv.loc[i] = getBestCrossValidation(i)

toc = time.time()
print('Total elapsed time:', (toc-tic)/60, 'minutes.')

# Plot results
sns.lineplot(x=models_best_cv.index, y='MSE', data=models_best_cv)
plt.xlabel('# predictors')
plt.ylabel('MSE')


# ------------------------
# (i) Forward selection OLS
# ------------------------

models_fwd_cv = pd.DataFrame(columns=['MSE', 'model', 'variables'])

tic = time.time()
predictors = []

for i in range(1,len(X.columns)+1):    
    models_fwd_cv.loc[i] = forwardCrossValidation(predictors)
    predictors = models_fwd_cv.loc[i]["model"].model.exog_names

toc = time.time()
print('Total elapsed time:', (toc-tic)/60, 'minutes.')

# Plot results
sns.lineplot(x=models_fwd_cv.index, y='MSE', data=models_fwd_cv)
plt.xlabel('# predictors')
plt.ylabel('MSE')

# ------------------------
# (i) Backward selection OLS
# ------------------------

models_bwd_cv = pd.DataFrame(columns=['MSE', 'model', 'variables'], index = range(1,len(X.columns)))

tic = time.time()
predictors = X.columns

while(len(predictors) > 1):  
    models_bwd_cv.loc[len(predictors)-1] = backwardCrossValidation(predictors)
    predictors = models_bwd_cv.loc[len(predictors)-1]['model'].model.exog_names

toc = time.time()
print('Total elapsed time:', (toc-tic)/60, 'minutes.')

models_bwd_cv['MSE'] = models_bwd_cv['MSE'].astype(float)

# Plot results
sns.lineplot(x=models_bwd_cv.index, y='MSE', data=models_bwd_cv)
plt.xlabel('# predictors')
plt.ylabel('MSE')

#%% -----------------------------------------
# Results
# -------------------------------------------

print('-------------------------------------')
print('Subset selection OLS:')
print('-------------------------------------')
print('-------------------------------------')
print('Best subset OLS:')
print('-------------------------------------')
print('#### BIC ####')
print(models_best[models_best['bic'] == models_best['bic'].min()].iloc[0, 1].params)
print('#### CV ####')
print(models_best_cv[models_best_cv['MSE'] == models_best_cv['MSE'].min()].iloc[0, 1].params)

print('-------------------------------------')
print('Forward selection OLS:')
print('-------------------------------------')
print('#### BIC ####')
print(models_fwd[models_fwd['bic'] == models_fwd['bic'].min()].iloc[0, 1].params)
print('#### CV ####')
print(models_fwd_cv[models_fwd_cv['MSE'] == models_fwd_cv['MSE'].min()].iloc[0, 1].params)

print('-------------------------------------')
print('Backward selection OLS:')
print('-------------------------------------')
print('#### BIC ####')
print(models_bwd[models_bwd['bic'] == models_bwd['bic'].min()].iloc[0, 1].params)
print('#### CV ####')
print(models_bwd_cv[models_bwd_cv['MSE'] == models_bwd_cv['MSE'].min()].iloc[0, 1].params)

# Logistic regression model is based on highest AUC

print('-------------------------------------')
print('Subset selection logistic regression:')
print('-------------------------------------')
print('-------------------------------------')
print('Best subset logistic regression:')
print('-------------------------------------')
print(models_best_glm[models_best_glm['auc'] == models_best_glm['auc'].max()].iloc[0, 1].params)

print('-------------------------------------')
print('Forward selection logistic regression:')
print('-------------------------------------')
print(models_fwd_glm[models_fwd_glm['auc'] == models_fwd_glm['auc'].max()].iloc[0, 1].params)

print('-------------------------------------')
print('Backward selection logistic regression:')
print('-------------------------------------')
print(models_bwd_glm[models_bwd_glm['auc'] == models_bwd_glm['auc'].max()].iloc[0, 1].params)
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
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.utils import resample
import statsmodels.api as sm
from sklearn import preprocessing


#%% -----------------------------------------
# Settings
# -------------------------------------------

# Set working directory
# os.getcwd()
# os.chdir(os.getcwd()+'/'+'miscellanious/stanford-statistical-learning')
os.chdir('/Users/viktor.eriksson2/Documents/python_files/miscellanious/stanford-statistical-learning')
os.getcwd()

# Plot settings
sns.set()


#%% -----------------------------------------
# Load data
# -------------------------------------------

# Datasets from ISLR
hitters = pd.read_csv('data/hitters.csv')
hitters.info()
hitters.head(10)


#%% -----------------------------------------
# Functions
# -------------------------------------------

mse = make_scorer(mean_squared_error)

def pcr_cv(X, y, seed, cv_folds, shuffle):
    """
    Perform Principle Component Regression evaluated with
    k-fold cross validation
    """
    
    # Get all principle components
    pca = PCA()
    X_reduced = pca.fit_transform(preprocessing.scale(X))
    
    # Get cv MSE for cumulative components
    M = X_reduced.shape[1]
    MSEs = []
    for m in range(M):
        model = LinearRegression()
        cv    = KFold(n_splits=cv_folds, random_state=seed, shuffle=True) if (shuffle == True) else cv_folds
        cv10  = cross_val_score(model, X_reduced[:, 0:m+1], y, cv=cv, scoring=mse)
        MSEs += [np.mean(np.abs(cv10))]
    
    # Plot results
    plt.figure(figsize=(10, 7))
    ax = sns.lineplot(x='# principal components', y='MSE', 
                 data=pd.DataFrame({'# principal components': np.arange(1, X_reduced.shape[1]+1), 
                                    'MSE': MSEs}))
    plt.title('PCR with %s-fold crossvalidation' % cv_folds)
    ax.axes.set_ylim(100000, 140000)

def pcr_validation_set(X, y, seed, test_share):
    """
    Perform Principle Component Regression evaluated with
    k-fold hold-out set
    """
    # With 50% holdout set

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(preprocessing.scale(X), y, test_size=test_share, random_state=seed)

    # PCR
    pca = PCA()
    X_train_ = pca.fit_transform(X_train)
    X_test_  = pca.fit_transform(X_test)
    
    # Get cv MSE for cumulative components
    M = X_train_.shape[1]
    MSEs = []
    for m in range(M):
        model = LinearRegression().fit(X_train_[:, 0:m+1], y_train)
        y_hat = model.predict(X_test_[:, 0:m+1])
        MSEs += [mean_squared_error(y_hat, y_test)]
    
    # Plot results
    plt.figure(figsize=(10, 7))
    ax = sns.lineplot(x='# principal components', y='MSE', 
                      data=pd.DataFrame({'# principal components': np.arange(1, X_train_.shape[1]+1), 
                                         'MSE': MSEs}))
    plt.title('PCR using validation-set approach with %s test share' % test_share)
    ax.axes.set_ylim(50000, 200000)

def pls_cv(X, y, seed, cv_folds):
    """
    Perform Partial Least Squares Regression evaluated with
    k-fold cross validation
    """
    # Standarize data
    X_ = preprocessing.scale(X)
    
    # Get cv MSE for cumulative components
    M = X_.shape[1]
    MSEs = []
    for m in range(M):
        cv = KFold(n_splits=cv_folds, random_state=seed, shuffle=True)

        results = cross_val_score(PLSRegression(n_components=m+1, scale=True, max_iter=10), 
                                  X_, y, cv=cv, scoring=mse)
        MSEs += [np.mean(np.abs(results))]

    # Plot results
    plt.figure(figsize=(10, 7))
    ax = sns.lineplot(x='# principal components', y='MSE', 
                 data=pd.DataFrame({'# principal components': np.arange(1, X_.shape[1]+1), 
                                    'MSE': MSEs}))
    plt.title('PLS with %s-fold crossvalidation' % cv_folds)
    ax.axes.set_ylim(100000, 140000)


#%% -----------------------------------------
# Data prepping
# -------------------------------------------

# Remove rows where salary is missing
hitters2 = hitters.dropna(subset=['Salary'])

# Create dummy variables for categorical predictors
dummies = pd.get_dummies(hitters2[['League', 'Division', 'NewLeague']])

# Create new dataframe with predictiors and response
y = hitters2[['Salary']]
X_ = hitters2.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1)
X = pd.concat([X_, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)


#%% -----------------------------------------
# PCR
# -------------------------------------------

# Get principle components
pca = PCA()
X_reduced = pca.fit_transform(preprocessing.scale(X))

# Print first 5 variables and their corresponding principal components
pd.DataFrame(pca.components_.T).loc[:5,:5]

# Plot the cumulative sum of variance explained by principle components
plt.figure(figsize=(10, 7))
sns.lineplot(x='# principal components', y='% variance explained', data=pd.DataFrame({
    '# principal components': np.arange(1, len(pca.explained_variance_ratio_)+1),
    '% variance explained': np.cumsum(pca.explained_variance_ratio_)
}))

X_reduced.shape

# How does linear regression perform when these principle components are used as predictors

# Cross-validation
pcr_cv(X=X, y=y, seed=1, cv_folds=10, shuffle=True)
# pcr_cv(X=X, y=y, seed=1, cv_folds=10, shuffle=False)
pcr_cv(X=X, y=y, seed=1, cv_folds=30, shuffle=True)
pcr_cv(X=X, y=y, seed=1, cv_folds=50, shuffle=True)

# Validation set approach
pcr_validation_set(X=X, y=y, seed=3, test_share=0.5)
pcr_validation_set(X=X, y=y, seed=3, test_share=0.3)
pcr_validation_set(X=X, y=y, seed=3, test_share=0.7)


#%% -----------------------------------------
# PLS
# -------------------------------------------

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(preprocessing.scale(X), y, test_size=0.1, random_state=1)

# PLS regression with K-fold cross-validation
pls_cv(X=X_train, y=y_train, seed=1, cv_folds=10) # M=2 smallest MSE
pls_cv(X=X_train, y=y_train, seed=1, cv_folds=30) # M=12 smallest MSE
pls_cv(X=X_train, y=y_train, seed=1, cv_folds=50) # M=12 smallest MSE

# Evaluate on test data
pls2 = PLSRegression(n_components=2)
pls12 = PLSRegression(n_components=12)

pls2.fit(X_train, y_train)
pls12.fit(X_train, y_train)

mean_squared_error(y_test, pls2.predict(X_test))
mean_squared_error(y_test, pls12.predict(X_test))

# M=12 yields the lowest MSE in this case










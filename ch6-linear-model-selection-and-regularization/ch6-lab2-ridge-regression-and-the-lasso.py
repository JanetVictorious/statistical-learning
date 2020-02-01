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
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import resample
import statsmodels.api as sm
from sklearn import preprocessing


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
hitters.head()


#%% -----------------------------------------
# Functions
# -------------------------------------------

# Mean squared error
mse = make_scorer(mean_squared_error)

def my_mse(y, y_pred):
    return np.sum(np.square(y_pred - y)) / y.size

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

X.info()

# Lets have a look at the different values in hitters2 to see wether we should 
# normalize or standardize variables

# Lets take some variables
sns.distplot(X['Years'])
sns.distplot(X['Walks'])
sns.distplot(X['AtBat'])
sns.distplot(X['League_N'])
sns.distplot(X['NewLeague_N'])

# Standardize variables

# Create the Scaler object
scaler = preprocessing.StandardScaler()

# Fit data on the scaler object
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=[i for i in X.columns])
y_scaled = scaler.fit_transform(y)
y_scaled = pd.DataFrame(y_scaled, columns=[i for i in y.columns])

# Lets look at the standardized variables
sns.distplot(X_scaled['Years'])
sns.distplot(y_scaled['Salary'])

#%% -----------------------------------------
# Ridge regression
# -------------------------------------------

alphas = 10**np.linspace(10, -2, 100)

# Remember to standardize variables. Done by setting normalize=True

# Matrix for storing coefficients
models_ridge = pd.DataFrame(columns=['coef', 'alpha', 'MSE'])

# Perform ridge regression for each of the different alphas
for a in alphas:
    # Standardized variables
    model = Ridge(alpha=a, normalize=False).fit(X_scaled, y_scaled)
    pred = model.predict(X_scaled)
    MSEs = mean_squared_error(y_scaled, pred)
    # Normalized variables
    # model = Ridge(alpha=a, normalize=True).fit(X, y)
    # pred = model.predict(X)
    # MSEs = mean_squared_error(y, pred)
    df = pd.DataFrame({'coef': model.coef_.tolist(), 'alpha': a, 'MSE': MSEs})
    models_ridge = models_ridge.append(df, ignore_index=True)
    
# Split out coefficients by their corresponding predictor
models_ridge[[i for i in X_scaled.columns]] = pd.DataFrame(models_ridge['coef'].values.tolist(), index=models_ridge.index, columns=[i for i in X_scaled.columns])

models_ridge.iloc[49]
models_ridge.iloc[59]

# Plot results
plt.figure(figsize=(10, 10))

plt.subplot(2, 1, 1)
for i in models_ridge.drop(['coef', 'alpha', 'MSE'], axis=1).columns:
    ax = sns.lineplot(x='alpha', y=i, data=models_ridge) if models_ridge[i].abs().max() >= 0.3 else sns.lineplot(x='alpha', y=i, data=models_ridge, color='grey')
ax.set_xscale('log')
plt.axis('tight')
plt.legend([i for i in models_ridge.drop(['coef', 'alpha', 'MSE'], axis=1).columns])
plt.xlabel('alpha')
plt.ylabel('weights')

plt.subplot(2, 1, 2)
ax = sns.lineplot(x='alpha', y='MSE', data=models_ridge)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('MSE')

# In the plot we see that 4 predictors are larger than the rest,
# namely; Years, League_N, Division_W and NewLeague_N

# Split data into training and test sets
X_train, X_test , y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.5, random_state=1)

# Validation set approach with lambda=4
model = Ridge(alpha=4, normalize=False).fit(X_train, y_train)
pred = model.predict(X_test)
mean_squared_error(y_test, pred)

# If we fit with only the intercept the MSE would be as follows
mean_squared_error(y_test, np.ones(len(y_test))*(y_train.mean().values))

# Validation set approach with lambda=10^10 (will make coefficients close to 0)
model = Ridge(alpha=10**10, normalize=False).fit(X_train, y_train)
pred = model.predict(X_test)
mean_squared_error(y_test, pred)

# We will now check if Ridge regression has any benefit over OLS (alhpa=0)
model = Ridge(alpha=0, normalize=False).fit(X_train, y_train)
pred = model.predict(X_test)
mean_squared_error(y_test, pred)

# Ridge regression with alpha=4 performs better than OLS looking at MSE.
# We can use cross validation to better derive an alpha which minimize the MSE
model = RidgeCV(alphas=alphas, normalize=False, scoring='neg_mean_squared_error').fit(X_scaled, y_scaled)
model.coef_
model.alpha_
pred = model.predict(X)
mean_squared_error(y, pred)

# The results is from LOOCV and alpha=0.013 yields the lowest MSE=93936.
# If we want to plot MSE against alpha we need to do this in a for-loop.
# The loop will use MSE instead of negative mean squared error

# Matrix for storing MSE and alpha
models_ridge_cv = pd.DataFrame(columns=['alpha', 'MSE'])

# LOOCV for each value of alpha
for a in alphas:
    model = RidgeCV(alphas=[a], normalize=False, scoring='neg_mean_squared_error', store_cv_values=True).fit(X_scaled, y_scaled)
    MSEs = model.cv_values_.mean()
    df = pd.DataFrame({'alpha': [a], 'MSE': [MSEs]})
    models_ridge_cv = models_ridge_cv.append(df, ignore_index=True)

# Plot results
plt.figure(figsize=(10, 7))
ax = sns.lineplot(x='alpha', y='MSE', data=models_ridge_cv)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('MSE')

# Use optimal alpha in final Ridge regression
opt_alpha = models_ridge_cv[models_ridge_cv['MSE']==models_ridge_cv['MSE'].min()]['alpha'].iloc[0]
opt_model = Ridge(alpha=opt_alpha, normalize=True).fit(X, y)
opt_pred = opt_model.predict(X)
mean_squared_error(y, opt_pred)














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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
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

def ridge_lasso_cv(X, y, a, k, model):
    """Perform ridge regresion with 
    k-fold cross validation to return mean MSE scores for each fold"""
    # Split dataset into k-folds
    # Note: np.array_split doesn't raise excpetion is folds are unequal in size
    X_folds = np.array_split(X, k)
    y_folds = np.array_split(y, k)
    
    MSEs = []
    for f in np.arange(len(X_folds)):
        # Create training and test sets
        X_test  = X_folds[f]
        y_test  = y_folds[f]
        X_train = X.drop(X_folds[f].index)
        y_train = y.drop(y_folds[f].index)
        
        # Fit model
        model = Ridge(alpha=a, fit_intercept=False).fit(X_train, y_train) if (model == 'ridge') else (
            Lasso(alpha=a, fit_intercept=False, max_iter=10000).fit(X_train, y_train)
            )
        
        # Measure MSE
        y_hat = model.predict(X_test)
        MSEs += [mean_squared_error(y_test, y_hat)]
    return MSEs


#%% -----------------------------------------
# Data prepping
# -------------------------------------------

# Pair plot
plt.figure(figsize=(10, 7))
sns.pairplot(hitters.drop(columns=['League', 'Division', 'NewLeague']))

# Remove rows where salary is missing
hitters['Salary'].isna().sum()
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
sns.distplot(y['Salary'])

# Standardize variables

# Create the Scaler object
scaler = preprocessing.StandardScaler()

# Fit data on the scaler object
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=[i for i in X.columns])
# y_scaled = scaler.fit_transform(y)
# y_scaled = pd.DataFrame(y_scaled, columns=[i for i in y.columns])

# Lets look at the standardized variables
sns.distplot(X_scaled['Years'])
sns.distplot(y_scaled['Salary'])

# Split data into training and test sets
X_train, X_test , y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=1)

# Beta values. In ridge package refered to as alpha
alphas = 10**np.linspace(-3, 10, 100)


#%% -----------------------------------------
# Ridge regression
# -------------------------------------------

# Matrix for storing coefficients
models_ridge = pd.DataFrame(columns=['coef', 'alpha', 'MSE'])

# Perform ridge regression for each of the different alphas
for a in alphas:
    # Standardized variables
    model = Ridge(alpha=a, normalize=False).fit(X_train, y_train)
    pred = model.predict(X_test)
    MSEs = mean_squared_error(y_test, pred)
    
    # # Normalized variables
    # model = Ridge(alpha=a, normalize=True).fit(X, y)
    # pred = model.predict(X)
    # MSEs = mean_squared_error(y, pred)
    
    # Save results
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

# In the plot we see that 6 predictors are larger than the rest,
# namely; AtBat, Hits, CAtBat, Years, League_N, Division_W and NewLeague_N

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
pred = model.predict(X_scaled)
mean_squared_error(y_scaled, pred)

# The results is from LOOCV and alpha=0.013 yields the lowest MSE=93936.
# If we want to plot MSE against alpha we need to do this in a for-loop.
# The loop will use MSE instead of negative mean squared error

# Matrix for storing MSE and alpha
models_ridge_cv = pd.DataFrame(columns=['alpha', 'MSE'])

# LOOCV for each value of alpha
for a in alphas:
    model = RidgeCV(alphas=[a], normalize=False, scoring='neg_mean_squared_error', store_cv_values=True).fit(X_scaled, y)
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

# Perform k-fold cross-validation
models_ridge_cv_2 = pd.DataFrame(columns=['alpha', 'MSE'])

for a in alphas:
    MSEs = np.mean(ridge_lasso_cv(X=X_scaled, y=y, a=a, k=30, model='ridge'))
    df = pd.DataFrame({'alpha': [a], 'MSE': [MSEs]})
    models_ridge_cv_2 = models_ridge_cv_2.append(df, ignore_index=True)

plt.figure(figsize=(10, 7))
ax = sns.lineplot(x='alpha', y='MSE', data=models_ridge_cv_2)
ax.set_xscale('log')
plt.axis('tight')

# Use optimal alpha in final Ridge regression
opt_alpha = models_ridge_cv_2[models_ridge_cv_2['MSE'] == models_ridge_cv_2['MSE'].min()]['alpha'].iloc[0]
opt_model = Ridge(alpha=opt_alpha, normalize=False).fit(X_scaled, y_scaled)
opt_model.coef_
opt_pred = opt_model.predict(X_scaled)
mean_squared_error(y_scaled, opt_pred)


#%% -----------------------------------------
# Lasso
# -------------------------------------------

# Matrix for storing coefficients
models_lasso = pd.DataFrame(columns=['coef', 'alpha', 'MSE'])

# Perform lasso regression for each of the different alphas
for a in alphas:
    # Standardized variables
    model = Lasso(alpha=a, fit_intercept=False, max_iter=10000).fit(X_train, y_train)
    pred = model.predict(X_test)
    MSEs = mean_squared_error(y_test, pred)
    
    # Save results
    df = pd.DataFrame({'coef': [model.coef_.tolist()], 'alpha': a, 'MSE': MSEs})
    models_lasso = models_lasso.append(df, ignore_index=True)
    
# Split out coefficients by their corresponding predictor
models_lasso[[i for i in X_scaled.columns]] = pd.DataFrame(models_lasso['coef'].values.tolist(), index=models_lasso.index, columns=[i for i in X_scaled.columns])

# Plot results
plt.figure(figsize=(10, 10))

plt.subplot(2, 1, 1)
for i in models_lasso.drop(['coef', 'alpha', 'MSE'], axis=1).columns:
    ax = sns.lineplot(x='alpha', y=i, data=models_lasso) if models_lasso[i].abs().max() >= 0.3 else sns.lineplot(x='alpha', y=i, data=models_lasso, color='grey')
ax.set_xscale('log')
plt.axis('tight')
plt.legend([i for i in models_lasso.drop(['coef', 'alpha', 'MSE'], axis=1).columns])
plt.xlabel('alpha')
plt.ylabel('weights')

plt.subplot(2, 1, 2)
ax = sns.lineplot(x='alpha', y='MSE', data=models_lasso)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('MSE')

# Perform k-fold cross-validation
models_lasso_cv_2 = pd.DataFrame(columns=['alpha', 'MSE'])

for a in alphas:
    MSEs = np.mean(ridge_lasso_cv(X=X_scaled, y=y, a=a, k=30, model='lasso'))
    df = pd.DataFrame({'alpha': [a], 'MSE': [MSEs]})
    models_lasso_cv_2 = models_lasso_cv_2.append(df, ignore_index=True)

plt.figure(figsize=(10, 7))
ax = sns.lineplot(x='alpha', y='MSE', data=models_lasso_cv_2)
ax.set_xscale('log')
plt.axis('tight')

# Compare Ridge and Lasso MSE
plt.figure(figsize=(10, 7))
sns.lineplot(x='alpha', y='MSE', data=models_ridge_cv_2, label='Ridge, k=30')
ax = sns.lineplot(x='alpha', y='MSE', data=models_lasso_cv_2, label='Lasso, k=30')
ax.set_xscale('log')
plt.axis('tight')
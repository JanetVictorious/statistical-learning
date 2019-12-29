# -----------------------------------------
# Chapter 5 - Resampling methods
# -----------------------------------------

# Mutate/Assign example
# default[['balance']].assign(
#     bal2 = lambda x: x['balance'].map(lambda balance: balance*0.1),
#     test = lambda x: x['balance'].map(lambda balance: True if balance > 1000 else False)
# )

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
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import statsmodels.api as sm
import statsmodels.formula.api as smf

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


#%% -----------------------------------------
# Functions
# -------------------------------------------


#%% -----------------------------------------
# Exercises
# -------------------------------------------

default['student2'] = np.where(default['student'] == 'No', 0, 1)
default['def01'] = np.where(default['default'] == 'No', 0, 1)

sns.pairplot(default, hue='def01')

default.groupby('default').size()

# ------------------------
#### Question 5 ####
# ------------------------
#### (a - c) ####
# Fit a logistic regression model that uses income and balance to predict default.
results_train_glm1 = smf.glm(formula='def01 ~ income + balance', data=default, family=sm.families.Binomial()).fit()
pred_test_glm1 = [0 if x < 0.5 else 1 for x in results_train_glm1.predict(default)]
print(results_train_glm1.summary())

# Model predictions are with respect to default=No
np.column_stack((default.as_matrix(columns = ['def01']).flatten(), results_train_glm1.model.endog))

sns.scatterplot(x='income', y='balance', data=default, hue='def01')
sns.boxplot(x='default', y='income', data=default, hue='def01')
sns.boxplot(x='default', y='balance', data=default, hue='def01')

# Update using only balance as predictor
results_train_glm1 = smf.glm(formula='def01 ~ balance', data=default, family=sm.families.Binomial()).fit()
pred_test_glm1 = [0 if x < 0.5 else 1 for x in results_train_glm1.predict(default)]
print(results_train_glm1.summary())
results_train_glm1.params

# Plot probabilities against balance
default['probs'] = results_train_glm1.predict(default)

sns.scatterplot(x='balance', y='def01', data=default, hue='def01')
sns.lineplot(x='balance', y='probs', data=default)

# ROC
lr_fpr1, lr_tpr1, thresholds1 = roc_curve(default['def01'], results_train_glm1.predict(default))
print('AUC: %f' % roc_auc_score(default['def01'], results_train_glm1.predict(default)))

# ROC dataframes
roc_glm1 = pd.DataFrame({
    'fpr': pd.Series(lr_fpr1, index=np.arange(len(lr_tpr1))),
    'tpr': pd.Series(lr_tpr1, index=np.arange(len(lr_tpr1))),
    '1-fpr': pd.Series(1-lr_fpr1, index=np.arange(len(lr_tpr1))),
    'tf': pd.Series(lr_tpr1-(1-lr_fpr1), index=np.arange(len(lr_tpr1))),
    'thresholds': pd.Series(thresholds1, index=np.arange(len(lr_tpr1)))
})

# Plot ROC curve
sns.lineplot(x='fpr', y='tpr', data=roc_glm1, ci=None)
plt.plot([x for x in np.arange(0, 1.1, 0.1)], [x for x in np.arange(0, 1.1, 0.1)], 'b--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# Optimal threshold
roc_glm1.ix[(roc_glm1['tf']-0).abs().argsort()[:1]]

# (b) Using the validation set approach, estimate the test error of this model.
# In order to do this, you must perform the following steps:
# i. Split the sample set into a training set and a validation set.
# ii. Fit a multiple logistic regression model using only the training observations.
# iii. Obtain a prediction of default status for each individual in the validation 
# set by computing the posterior probability of default for 
# that individual, and classifying the individual to the default category if the 
# posterior probability is greater than 0.5.
# iv. Compute the validation set error, which is the fraction of the observations in 
# the validation set that are misclassified.

def val_set_fn():
    # Train and test data
    seed = randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(default[['balance', 'income']], default[['def01']], test_size=0.5, random_state=seed)

    results_train_glm1 = smf.glm(formula='def01 ~ income + balance', data=default.iloc[list(X_train.index)], family=sm.families.Binomial()).fit()
    pred_test_glm1 = [0 if x < 0.5 else 1 for x in results_train_glm1.predict(default.iloc[list(X_test.index)])]

    conf1 = confusion_matrix(default.iloc[list(X_test.index)]['def01'], pred_test_glm1)
    
    print('Validation set error: %f' % ((conf1[1,0] + conf1[0,1])/conf1.sum()))

val_set_fn()
val_set_fn()
val_set_fn()
val_set_fn()

#### (d) ####
# Now consider a logistic regression model that predicts the probability of default 
# using income, balance, and a dummy variable for student. 
# Estimate the test error for this model using the validation set approach. 
# Comment on whether or not including a dummy variable for student leads to a 
# reduction in the test error rate.

def val_set_fn2():
    # Train and test data
    seed = randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(default[['balance', 'income', 'student2']], default[['def01']], test_size=0.5, random_state=seed)

    results_train_glm1 = smf.glm(formula='def01 ~ income + balance', data=default.iloc[list(X_train.index)], family=sm.families.Binomial()).fit()
    pred_test_glm1 = [0 if x < 0.5 else 1 for x in results_train_glm1.predict(default.iloc[list(X_test.index)])]

    conf1 = confusion_matrix(default.iloc[list(X_test.index)]['def01'], pred_test_glm1)
    
    print('Validation set error: %f' % ((conf1[1,0] + conf1[0,1])/conf1.sum()))

val_set_fn2()
val_set_fn2()
val_set_fn2()
val_set_fn2()


# ------------------------
#### Question 6 ####
# ------------------------
# We continue to consider the use of a logistic regression model to predict the 
# probability of default using income and balance on the Default data set. 
# In particular, we will now compute estimates for the standard errors of the income 
# and balance logistic regression coefficients in two different ways: 
# (1) using the bootstrap, and 
# (2) using the standard formula for computing the standard errors in the 
# glm() function. Do not forget to set a random seed before beginning your analysis.

# (1)

# Dataframe for storing coefficient estimates
coef_df = pd.DataFrame(0, index=np.arange(1000), columns=['income', 'balance'])

for i in range(1000):
    # Bootstrap dataset
    boot = resample(default, replace=True, n_samples=None, random_state=randint(0,1000))
    # Logistic regression
    results_boot_glm1 = smf.glm(formula='def01 ~ income + balance', data=boot, family=sm.families.Binomial()).fit()
    # Parameters
    params = results_boot_glm1.params.to_frame()
    params.columns = ['value']
    # Insert coefficients into dataframe
    coef_df.iloc[i, 0] = params[params.index == 'income'].iloc[0]['value']
    coef_df.iloc[i, 1] = params[params.index == 'balance'].iloc[0]['value']

print('Income, mean: %f, sd: %f' % (coef_df['income'].mean(), coef_df['income'].std()))
print('Balance, mean: %f, sd: %f' % (coef_df['balance'].mean(), coef_df['balance'].std()))

# (2)
results_glm1 = smf.glm(formula='def01 ~ income + balance', data=default, family=sm.families.Binomial()).fit()
print(results_glm1.summary())


# ------------------------
#### Question 7 ####
# ------------------------
# In Sections 5.3.2 and 5.3.3, we saw that the cv.glm() function can be used in order 
# to compute the LOOCV test error estimate. Alternatively, one could compute those 
# quantities using just the glm() and predict.glm() functions, and a for loop. 
# You will now take this approach in order to compute the LOOCV error for a simple 
# logistic regression model on the Weekly data set. Recall that in the context of 
# classification problems, the LOOCV error is given in (5.4).

weekly['dir01'] = np.where(weekly['Direction'] == 'Down', 0 , 1)

#### (a) ####
# Fit a logistic regression model that predicts Direction using Lag1 and Lag2.

results_glm1 = smf.glm(formula='dir01 ~ Lag1 + Lag2', data=weekly, family=sm.families.Binomial()).fit()
print(results_glm1.summary())

#### (b) ####
# Fit a logistic regression model that predicts Direction using Lag1 and Lag2 using all but the first observation.
results_glm2 = smf.glm(formula='dir01 ~ Lag1 + Lag2', data=weekly.iloc[1:,], family=sm.families.Binomial()).fit()
print(results_glm2.summary())

#### (c) ####
# Use the model from (b) to predict the direction of the first observation. 
# You can do this by predicting that the first observation will go up if 
# P(Direction="Up"|Lag1, Lag2) > 0.5. 
# Was this observation correctly classified?
pred_glm2 = results_glm2.predict(weekly.iloc[0:1,])
pred_nominal = ['Up' if x >= 0.5 else 'Down' for x in pred_glm2]
weekly.iloc[0:1,]

# The prediction of the first observation was Up, but the actual direction was Down.

#### (d) ####
# Write a for loop from i=1 to i=n, where n is the number of observations in the data set,
# that performs each of the following steps:
# i. Fit a logistic regression model using all but the ith observation to predict 
# Direction using Lag1 and Lag2.
# ii. Compute the posterior probability of the market moving up for the ith observation.
# iii. Use the posterior probability for the ith observation in order to predict 
# whether or not the market moves up.
# iv. Determine whether or not an error was made in predicting the direction for the 
# ith observation. If an error was made, then indicate this as a 1, 
# and otherwise indicate it as a 0.

err_df = pd.DataFrame(0, index=np.arange(len(weekly)), columns=['error'])

for i in range(len(weekly)):
    # i.
    results_glm = smf.glm(formula='dir01 ~ Lag1 + Lag2', data=weekly[weekly.index != i], family=sm.families.Binomial()).fit()
    # ii.
    pred_glm = results_glm.predict(weekly[weekly.index == i])
    # iii.
    pred_glm_nom = [1 if x >= 0.5 else 0 for x in pred_glm]
    # iv.
    err_df.iloc[i] = np.where(pred_glm_nom != weekly[weekly.index == i]['dir01'], 1, 0)


#### (e) ####
# Take the average of the n numbers obtained in (d)iv in order to obtain the LOOCV 
# estimate for the test error. Comment on the results.

# Nr. faulty classifications and error rate
err_df.sum()
err_df.sum()/len(err_df)


# ------------------------
#### Question 8 ####
# ------------------------
# We will now perform cross-validation on a simulated data set.

#### (a) ####
# Generate a simulated data set as follows:
seed(1)
x = randn(100)
y = x - 2*x*x + randn(100)

# In this data set, what is n and what is p? Write out the model used to generate 
# the data in equation form.

# n is 100 (number of observations), and p is 2 (first for x and second for x^2)

#### (b) ####
# Create a scatterplot of X against Y . Comment on what you find.
df1 = pd.DataFrame({
    'x': x,
    'y': y
})

df1['x'].describe()
df1['y'].describe()

sns.scatterplot(x=x, y=y, data=df1)
plt.xlabel('x')
plt.ylabel('y')

# Looks like a parabola. The variance between observations seems to be largest at 
# the top of the parabola (x = 0).

#### (c) ####
# Set a random seed, and then compute the LOOCV errors that result from fitting 
# the following four models using least squares:
# i.   Y = β0 + β1X + ε
# ii.  Y = β0 + β1X + β2X2 + ε
# iii. Y = β0 + β1X + β2X2 + β3X3 + ε
# iv.  Y = β0 + β1X + β2X2 + β3X3 + β4X4 + ε.
# Note you may find it helpful to use the data.frame() function
# to create a single data set containing both X and Y.

# i.
res1_glm = smf.ols(formula='y ~ x', data=df1)








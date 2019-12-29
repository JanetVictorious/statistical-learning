# -----------------------------------------
# Chapter 3 - Linear regression
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
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Sampling lib
# from random import seed
# from random import random
# from random import gauss
from numpy.random import seed
from numpy.random import rand
from numpy.random import randn



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
auto = pd.read_csv('data/auto.csv')
auto.dtypes

carseats = pd.read_csv('data/carseats.csv')
carseats.dtypes

#%% -----------------------------------------
# Functions
# -------------------------------------------


#%% -----------------------------------------
# Exercises
# -------------------------------------------

#### Question 8 ####

# Simple linear regression
model = LinearRegression().fit(auto[['horsepower']], auto[['mpg']])
y_pred = model.predict(auto[['horsepower']])
model.get_params()

# Print results
print('R2: ', model.score(auto[['horsepower']], auto[['mpg']]))
print('intercept:', model.intercept_)
print('coefficient:', model.coef_)

sns.scatterplot(x='horsepower', y='mpg', data=auto)
plt.plot(auto[['horsepower']], y_pred, 'r--')


#### Question 9 ####
sns.pairplot(auto)
auto.corr()

# Multi-linear regression
x_val = auto.drop(['mpg', 'name', 'acceleration', 'cylinders', 'horsepower', 'displacement'], axis=1)
x_val = sm.add_constant(x_val)
y_val = auto[['mpg']]
model = sm.OLS(y_val, x_val)
results = model.fit()

# Print results
print(results.summary())

# Interaction between significant predictors
x_val = auto.drop(['mpg', 'name', 'acceleration', 'cylinders', 'horsepower', 'displacement'], axis=1)
interaction = PolynomialFeatures(degree=3, include_bias=False, interaction_only=True)
x_inter = interaction.fit_transform(x_val)
x_inter = pd.DataFrame(x_inter)
x_inter.columns = [colname for colname in interaction.get_feature_names(x_val.columns)]

x_inter = sm.add_constant(x_inter)
y_val = auto[['mpg']]
model = sm.OLS(y_val, x_inter)
results = model.fit()

print(results.summary())

# Polynomial regression
x_val = auto[['horsepower']]
x_val[['horsepower^2']] = x_val[['horsepower']].apply(lambda x: x*x, axis=1)

x_val = sm.add_constant(x_val)
y_val = auto[['mpg']]
model = sm.OLS(y_val, x_val)
results = model.fit()

# Create predicted values from linear fit
pred = pd.DataFrame(np.arange(46, 231, 1)) # Create horsepower values
pred.columns = ['x']
pred = sm.add_constant(pred) # Create constant for beta0
pred['x^2'] = pred[['x']].apply(lambda x: x*x, axis=1) # Create polynomial term
pred['y'] = results.predict(pred) # Predicted y-values

# Plot results
sns.scatterplot(x='horsepower', y='mpg', data=auto)
plt.plot(pred[['x']], pred[['y']], 'r--')


#### Question 10 ####
carseats.dtypes

# Plot Sales vs Price for Urban and Us
sns.scatterplot(x='Price', y='Sales', hue='Urban', data=carseats)
sns.scatterplot(x='Price', y='Sales', hue='US', data=carseats)

model = smf.ols(formula='Sales ~ Price + Urban + US', data=carseats)
results = model.fit()

print(results.summary())

# The coefficients for the dummy variables tells us the effect of these variables vs the baseline.
# I.e. what is the effect of Urban Yes and US Yes vs the baseline of Urban No and US No.
# 
# The model on equation form is the following:
# 
# y = b0 + b1*x1 + b2*x2 + b3*x3 + e =
# 
#   = (b0 + b2 + b3) + b1*x1, if Urban yes Us yes
#   = (b0 + b2)      + b1*x1, if Urban yes US no
#   = (b0 + b3)      + b1*x1, if Urban no US yes
#   = (b0)           + b1*x1, if Urban no US yes

# Plot regression Sales vs Price to see if Urban in US have interaction
fit1 = smf.ols(formula='Sales ~ Price', data=carseats[carseats['Urban']=='Yes']).fit()
fit2 = smf.ols(formula='Sales ~ Price', data=carseats[carseats['Urban']!='Yes']).fit()

pred1 = fit1.predict(carseats[carseats['Urban']=='Yes'][['Price']])
pred2 = fit2.predict(carseats[carseats['Urban']!='Yes'][['Price']])

sns.scatterplot(x='Price', y='Sales', data=carseats)
plt.plot(carseats[carseats['Urban']=='Yes']['Price'], np.asarray(pred1), 'r--')
plt.plot(carseats[carseats['Urban']!='Yes']['Price'], np.asarray(pred2), 'g--')
plt.legend(['UrbanYes', 'UrbanNo', 'Observed data'])

fit1 = smf.ols(formula='Sales ~ Price', data=carseats[carseats['US']=='Yes']).fit()
fit2 = smf.ols(formula='Sales ~ Price', data=carseats[carseats['US']!='Yes']).fit()

pred1 = fit1.predict(carseats[carseats['US']=='Yes'][['Price']])
pred2 = fit2.predict(carseats[carseats['US']!='Yes'][['Price']])

sns.scatterplot(x='Price', y='Sales', data=carseats)
plt.plot(carseats[carseats['US']=='Yes']['Price'], np.asarray(pred1), 'r--')
plt.plot(carseats[carseats['US']!='Yes']['Price'], np.asarray(pred2), 'g--')
plt.legend(['USYes', 'USNo', 'Observed data'])

# Interaction for Urban. No interaction for US
model = smf.ols(formula='Sales ~ Price + US ', data=carseats)
results = model.fit()

print(results.summary())


#### Question 11 ####
seed(1)

x1 = randn(100)
y1 = 2*x1 + randn(100)

df1 = pd.DataFrame()
df1['x1'] = x1
df1['y1'] = y1

fit1 = smf.ols(formula='y1 ~ x1', data=df1).fit()
pred1 = fit1.predict(df1[['x1']])

print(fit1.summary())

sns.scatterplot(x='x1', y='y1', data=df1)
plt.plot(df1['x1'], np.asarray(pred1), 'r--')
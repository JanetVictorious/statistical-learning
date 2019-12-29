# -----------------------------------------
# Chapter 4 - Classification
# -----------------------------------------


# Confusion matrix and rates
#          
# +--------------------------------------+
# |               |   True condition     |
# |               |    0         1       |
# | --------------|----------------------|
# |  Predicted  0 |    a         b       |
# |  condition    |                      |
# |             1 |    c         d       |
# +--------------------------------------+
# 
# a - true negative
# b - false negative
# c - false positive
# d - true positive
# 
# True Positive Rate (sensitivity) = True Positives / (True Positives + False Negatives)
# --> d / (d + b)
# 
# False Positive Rate = False Positives / (False Positives + True Negatives)
# --> c / (c + a)
# 
# Specificity = True Negatives / (True Negatives + False Positives)
# --> a / (a + c)
# 
# False Positive Rate = 1 - Specificity
# 
# Misclassification rate = (False Positives + False Negatives) / All
# --> (b + c) / (a + b + c + d)


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
weekly = pd.read_csv('data/weekly.csv')
weekly.dtypes
weekly.head()

auto = pd.read_csv('data/auto.csv')
auto.dtypes
auto.head()

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
#### Question 10 ####
# ------------------------
#### (a) ####
sns.pairplot(weekly)
weekly.corr()

# No clear trends or correlations between lags. Volume traded has increased over time but otherwise the there are no
# clear or well separated classes.

#### (b) ####
model = smf.glm(formula='Direction ~ Lag1+Lag2+Lag3+Lag4+Lag5+Volume', data=weekly, family=sm.families.Binomial())
results = model.fit()

print(results.summary())

# From the logistic regression one variable is significant, namely Lag2.

#### (c) ####
pred = results.predict()
pred[0:10]

# Model predictions are with respect to Direction=Down
np.column_stack((weekly.as_matrix(columns = ["Direction"]).flatten(), results.model.endog))

pred_nominal = ['Up' if x < 0.5 else 'Down' for x in pred]

# Confusion matrix
conf_mat = confusion_matrix(weekly["Direction"], pred_nominal)

(conf_mat[0,0] + conf_mat[1,1])/conf_mat.sum()

# The diagonal of the confusion matrix show correct predictions while the off-diagonal shows faulty predictions.
# Since the regression is done on the whole data set the ratio of correct predictions is called null-ratio.
# Given the results we can see that the model is sligthly better than flipping an even coin which we expect from 
# stock data.

#### (d) ####
# Train on data before 2009
train = weekly['Year'] < 2009

model_train = smf.glm(formula='Direction ~ Lag1+Lag2+Lag3+Lag4+Lag5+Volume', data=weekly[train], family=sm.families.Binomial())
results_train = model.fit()

print(results_train.summary())

pred_test = results_train.predict(weekly[train == False])
pred_test[0:10]

pred_test_nominal = ['Up' if x < 0.5 else 'Down' for x in pred_test]

# Model from train data on test data
conf_mat_test = confusion_matrix(weekly[train == False]['Direction'], pred_test_nominal)

(conf_mat_test[0,0] + conf_mat_test[1,1])/conf_mat_test.sum()

#### (e) ####
# Linear discriminant analysis
sns.pairplot(weekly, hue='Direction')

x_train = np.asarray(weekly[train][['Lag2']])
y_train = np.asarray(weekly[train][['Direction']])
x_test = np.asarray(weekly[train == False][['Lag2']])

# LDA
model_train_lda = LinearDiscriminantAnalysis().fit(x_train, y_train)
pred_test_lda = model_train_lda.predict(x_test)
confusion_matrix(weekly[train == False]['Direction'], pred_test_lda)

#### (f) ####
# Quadratic discriminant analysis

# QDA
model_train_qda = QuadraticDiscriminantAnalysis().fit(x_train, y_train)
pred_test_qda = model_train_qda.predict(x_test)
print(confusion_matrix(weekly[train == False]['Direction'], pred_test_qda))

#### (g) ####
# K-Nearest-Neighbor
model_train_knn = KNeighborsClassifier(n_neighbors=1).fit(x_train, y_train)
pred_test_knn = model_train_knn.predict(x_test)
confusion_matrix(weekly[train == False]['Direction'], pred_test_knn)

#### (h) ####
# Logistic regression and LDA gives the best test error rates.

#### (i) ####

# Try different classifications methods but this time we include more or other variables than just Lag2.

# Training data
train = weekly['Year'] < 2005

x_train = np.asarray(weekly[train].apply(lambda x: x['Lag1']*x['Lag2'], axis=1)).reshape(-1, 1)
y_train = np.asarray(weekly[train][['Direction']])
x_test = np.asarray(weekly[train == False].apply(lambda x: x['Lag1']*x['Lag2'], axis=1)).reshape(-1, 1)

# Regression models
results_train_ln = smf.glm(formula='Direction ~ Lag1:Lag2', data=weekly[train], family=sm.families.Binomial()).fit()
results_train_lda = LinearDiscriminantAnalysis().fit(x_train, y_train)
results_train_qda = QuadraticDiscriminantAnalysis().fit(x_train, y_train)
results_train_knn1 = KNeighborsClassifier(n_neighbors=1).fit(x_train, y_train)
results_train_knn2 = KNeighborsClassifier(n_neighbors=2).fit(x_train, y_train)

# Predicted results
pred_test_ln = ['Up' if x < 0.5 else 'Down' for x in results_train_ln.predict(weekly[train == False])]
pred_test_lda = results_train_lda.predict(x_test)
pred_test_qda = results_train_qda.predict(x_test)
pred_test_knn1 = results_train_knn1.predict(x_test)
pred_test_knn2 = results_train_knn2.predict(x_test)

# Confustion matrices
print('GLM')
conf1 = confusion_matrix(weekly[train == False]['Direction'], pred_test_ln)
print((conf1[0,0] + conf1[1,1])/conf1.sum())
print('LDA')
conf1 = confusion_matrix(weekly[train == False]['Direction'], pred_test_lda)
print((conf1[0,0] + conf1[1,1])/conf1.sum())
print('QDA')
conf1 = confusion_matrix(weekly[train == False]['Direction'], pred_test_qda)
print((conf1[0,0] + conf1[1,1])/conf1.sum())
print('KNN1')
conf1 = confusion_matrix(weekly[train == False]['Direction'], pred_test_knn1)
print((conf1[0,0] + conf1[1,1])/conf1.sum())
print('KNN2')
conf1 = confusion_matrix(weekly[train == False]['Direction'], pred_test_knn2)
print((conf1[0,0] + conf1[1,1])/conf1.sum())


# ------------------------
#### Question 11 ####
# ------------------------
#### (a) ####
auto2 = auto.copy()
auto2.dtypes

med_mpg = np.median(auto2['mpg'])

auto2['mpg01'] = auto2[['mpg']].apply(lambda x: 1 if x['mpg'] >= med_mpg else 0, axis=1)

#### (b) ####
sns.pairplot(auto2, hue='mpg01')

# Box plots
sns.boxplot(x='mpg01', y='acceleration', data=auto2, hue='mpg01')
sns.boxplot(x='mpg01', y='displacement', data=auto2, hue='mpg01')
sns.boxplot(x='mpg01', y='horsepower', data=auto2, hue='mpg01')
sns.boxplot(x='mpg01', y='weight', data=auto2, hue='mpg01')
sns.boxplot(x='mpg01', y='year', data=auto2, hue='mpg01')
sns.boxplot(x='mpg01', y='origin', data=auto2, hue='mpg01')

# So the distribution between cars that have an mpg above the median are more separated within some variables than others.
# Looking at the scatter plots, and box plots, one can see that the distribution of mpg01 with regards to
# the cars acceleration is not well separated. Looking at horsepower we see that the distribution w.r.t. mpg01 is better
# separated. The same can be said for weight and displacement. 

#### (c) ####

#############
auto_train = auto2['year'] < 76
#############

x_train = np.asarray(auto2[auto_train][['horsepower', 'weight', 'displacement']])
y_train = np.asarray(auto2[auto_train][['mpg01']])
x_test = np.asarray(auto2[auto_train == False][['horsepower', 'weight', 'displacement']])

#### (d+e) ####
# Linear discriminant analysis
results_train_lda1 = LinearDiscriminantAnalysis().fit(np.delete(x_train, 2, 1), y_train)
results_train_lda2 = LinearDiscriminantAnalysis().fit(x_train, y_train)
results_train_qda1 = QuadraticDiscriminantAnalysis().fit(np.delete(x_train, 2, 1), y_train)
results_train_qda2 = QuadraticDiscriminantAnalysis().fit(x_train, y_train)

pred_test_lda1 = results_train_lda1.predict(np.delete(x_test, 2, 1))
pred_test_lda2 = results_train_lda2.predict(x_test)
pred_test_qda1 = results_train_qda1.predict(np.delete(x_test, 2, 1))
pred_test_qda2 = results_train_qda2.predict(x_test)


print('LDA ~ horsepower + weight')
conf1 = confusion_matrix(auto2[auto_train == False]['mpg01'], pred_test_lda1)
print((conf1[0,0] + conf1[1,1])/conf1.sum())
print('LDA ~ horsepower + weight + displacement')
conf2 = confusion_matrix(auto2[auto_train == False]['mpg01'], pred_test_lda2)
print((conf2[0,0] + conf2[1,1])/conf2.sum())
print('QDA ~ horsepower + weight')
conf1 = confusion_matrix(auto2[auto_train == False]['mpg01'], pred_test_qda1)
print((conf1[0,0] + conf1[1,1])/conf1.sum())
print('QDA ~ horsepower + weight + displacement')
conf2 = confusion_matrix(auto2[auto_train == False]['mpg01'], pred_test_qda2)
print((conf2[0,0] + conf2[1,1])/conf2.sum())

#### (f) ####
# Logistic regression

results_train_glm1 = smf.glm(formula='mpg01 ~ horsepower + weight', data=auto2[auto_train], family=sm.families.Binomial()).fit()
results_train_glm2 = smf.glm(formula='mpg01 ~ horsepower + weight + displacement', data=auto2[auto_train], family=sm.families.Binomial()).fit()

# Threshold at 0.5
pred_test_glm1 = [0 if x < 0.5 else 1 for x in results_train_glm1.predict(auto2[auto_train == False])]
pred_test_glm2 = [0 if x < 0.5 else 1 for x in results_train_glm2.predict(auto2[auto_train == False])]

print('GLM ~ horsepower + weight')
conf1 = confusion_matrix(auto2[auto_train == False]['mpg01'], pred_test_glm1)
print('Classification rate:', (conf1[0,0] + conf1[1,1])/conf1.sum())
print('TPR:', conf1[1, 1]/(conf1[:, 1].sum()))
print('FPR:', conf1[1, 0]/(conf1[:, 0].sum()))

print('GLM ~ horsepower + weight + displacement')
conf2 = confusion_matrix(auto2[auto_train == False]['mpg01'], pred_test_glm2)
print('Classification rate:', (conf2[0,0] + conf2[1,1])/conf2.sum())
print('TPR:', conf2[1, 1]/(conf2[:, 1].sum()))
print('FPR:', conf2[1, 0]/(conf2[:, 0].sum()))

# AUC for both models
print('AUC for model 1 :%f' % roc_auc_score(auto2[auto_train == False]['mpg01'], results_train_glm1.predict(auto2[auto_train == False])))
print('AUC for model 2 :%f' % roc_auc_score(auto2[auto_train == False]['mpg01'], results_train_glm2.predict(auto2[auto_train == False])))

# Confusion matrix rates for models
lr_fpr1, lr_tpr1, thresholds1 = roc_curve(auto2[auto_train == False]['mpg01'], results_train_glm1.predict(auto2[auto_train == False]))
lr_fpr2, lr_tpr2, thresholds2 = roc_curve(auto2[auto_train == False]['mpg01'], results_train_glm2.predict(auto2[auto_train == False]))

# Plot ROC curves
plt.plot(lr_fpr1, lr_tpr1, marker='.', label='Logistic1')
plt.plot(lr_fpr2, lr_tpr2, marker='.', label='Logistic2')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

# ROC dataframes
roc1 = pd.DataFrame({
    'fpr': pd.Series(lr_fpr1, index=np.arange(len(lr_tpr1))),
    'tpr': pd.Series(lr_tpr1, index=np.arange(len(lr_tpr1))),
    '1-fpr': pd.Series(1-lr_fpr1, index=np.arange(len(lr_tpr1))),
    'tf': pd.Series(lr_tpr1-(1-lr_fpr1), index=np.arange(len(lr_tpr1))),
    'thresholds': pd.Series(thresholds1, index=np.arange(len(lr_tpr1)))
})

roc2 = pd.DataFrame({
    'fpr': pd.Series(lr_fpr2, index=np.arange(len(lr_tpr2))),
    'tpr': pd.Series(lr_tpr2, index=np.arange(len(lr_tpr2))),
    '1-fpr': pd.Series(1-lr_fpr2, index=np.arange(len(lr_tpr2))),
    'tf': pd.Series(lr_tpr2-(1-lr_fpr2), index=np.arange(len(lr_tpr2))),
    'thresholds': pd.Series(thresholds2, index=np.arange(len(lr_tpr2)))
})

# Optimal threshold
roc1.ix[(roc1['tf']-0).abs().argsort()[:1]]
roc2.ix[(roc2['tf']-0).abs().argsort()[:1]]

#### (g) ####
# K-Nearest Neighbor classification

sns.pairplot(auto2[['horsepower', 'weight', 'mpg01']], hue='mpg01')

# Train and test data
x_train = np.asarray(auto2[auto_train][['horsepower', 'weight']])
y_train = np.asarray(auto2[auto_train][['mpg01']])
x_test = np.asarray(auto2[auto_train == False][['horsepower', 'weight']])

# KNN models
results_train_knn1 = KNeighborsClassifier(n_neighbors=1).fit(x_train, y_train)
results_train_knn2 = KNeighborsClassifier(n_neighbors=2).fit(x_train, y_train)
results_train_knn5 = KNeighborsClassifier(n_neighbors=5).fit(x_train, y_train)
results_train_knn10 = KNeighborsClassifier(n_neighbors=10).fit(x_train, y_train)
results_train_knn100 = KNeighborsClassifier(n_neighbors=100).fit(x_train, y_train)

# Predicted results
pred_test_knn1 = results_train_knn1.predict(x_test)
pred_test_knn2 = results_train_knn2.predict(x_test)
pred_test_knn5 = results_train_knn5.predict(x_test)
pred_test_knn10 = results_train_knn10.predict(x_test)
pred_test_knn100 = results_train_knn100.predict(x_test)

# AUC for models
print('KNN1 ~ horsepower + weight')
conf1 = confusion_matrix(auto2[auto_train == False]['mpg01'], pred_test_knn1)
print('Classification rate:', (conf1[0,0] + conf1[1,1])/conf1.sum())
print('TPR:', conf1[1, 1]/(conf1[:, 1].sum()))
print('FPR:', conf1[1, 0]/(conf1[:, 0].sum()))
print('AUC: %f' % roc_auc_score(auto2[auto_train == False]['mpg01'], pred_test_knn1))

print('KNN2 ~ horsepower + weight')
conf1 = confusion_matrix(auto2[auto_train == False]['mpg01'], pred_test_knn2)
print('Classification rate:', (conf1[0,0] + conf1[1,1])/conf1.sum())
print('TPR:', conf1[1, 1]/(conf1[:, 1].sum()))
print('FPR:', conf1[1, 0]/(conf1[:, 0].sum()))
print('AUC: %f' % roc_auc_score(auto2[auto_train == False]['mpg01'], pred_test_knn2))

print('KNN5 ~ horsepower + weight')
conf1 = confusion_matrix(auto2[auto_train == False]['mpg01'], pred_test_knn5)
print('Classification rate:', (conf1[0,0] + conf1[1,1])/conf1.sum())
print('TPR:', conf1[1, 1]/(conf1[:, 1].sum()))
print('FPR:', conf1[1, 0]/(conf1[:, 0].sum()))
print('AUC: %f' % roc_auc_score(auto2[auto_train == False]['mpg01'], pred_test_knn5))

print('KNN10 ~ horsepower + weight')
conf1 = confusion_matrix(auto2[auto_train == False]['mpg01'], pred_test_knn10)
print('Classification rate:', (conf1[0,0] + conf1[1,1])/conf1.sum())
print('TPR:', conf1[1, 1]/(conf1[:, 1].sum()))
print('FPR:', conf1[1, 0]/(conf1[:, 0].sum()))
print('AUC: %f' % roc_auc_score(auto2[auto_train == False]['mpg01'], pred_test_knn10))

print('KNN100 ~ horsepower + weight')
conf1 = confusion_matrix(auto2[auto_train == False]['mpg01'], pred_test_knn100)
print('Classification rate:', (conf1[0,0] + conf1[1,1])/conf1.sum())
print('TPR:', conf1[1, 1]/(conf1[:, 1].sum()))
print('FPR:', conf1[1, 0]/(conf1[:, 0].sum()))
print('AUC: %f' % roc_auc_score(auto2[auto_train == False]['mpg01'], pred_test_knn100))


# ------------------------
#### Question 12 ####
# ------------------------

# +---------------------------------------------------------------------------+
# | Description of Boston data                                                
# | 
# | crim
# | per capita crime rate by town.
# | 
# | zn
# | proportion of residential land zoned for lots over 25,000 sq.ft.
# | 
# | indus
# | proportion of non-retail business acres per town.
# | 
# | chas
# | Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
# | 
# | nox
# | nitrogen oxides concentration (parts per 10 million).
# | 
# | rm
# | average number of rooms per dwelling.
# | 
# | age
# | proportion of owner-occupied units built prior to 1940.
# | 
# | dis
# | weighted mean of distances to five Boston employment centres.
# | 
# | rad
# | index of accessibility to radial highways.
# | 
# | tax
# | full-value property-tax rate per \$10,000.
# | 
# | ptratio
# | pupil-teacher ratio by town.
# | 
# | black
# | 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.
# | 
# | lstat
# | lower status of the population (percent).
# | 
# | medv
# | median value of owner-occupied homes in \$1000s.
# +----------------------------------------------------------------------------

boston.dtypes
boston.head()

# Median crime rate
med_crim = np.median(boston['crim'])

# Assign classification output as either being above or below median crime rate 
boston['crim01'] = boston[['crim']].apply(lambda x: 1 if x['crim'] >= med_crim else 0, axis=1)

# Pairs plot
sns.pairplot(boston, hue='crim01')

# Box plots
sns.boxplot(x='crim01', y='zn', data=boston, hue='crim01')
sns.boxplot(x='crim01', y='indus', data=boston, hue='crim01') # somewhat separated
sns.boxplot(x='crim01', y='chas', data=boston, hue='crim01')
sns.boxplot(x='crim01', y='nox', data=boston, hue='crim01') # somewhat separated
sns.boxplot(x='crim01', y='rm', data=boston, hue='crim01')
sns.boxplot(x='crim01', y='age', data=boston, hue='crim01') # somewhat separated
sns.boxplot(x='crim01', y='dis', data=boston, hue='crim01') 
sns.boxplot(x='crim01', y='rad', data=boston, hue='crim01')
sns.boxplot(x='crim01', y='ptratio', data=boston, hue='crim01')


# Train and test data
X_train, X_test, y_train, y_test = train_test_split(boston[['indus', 'nox', 'age']], boston[['crim01']], test_size=0.33, random_state=2)

boston.iloc[list(X_train.index)][['crim']]


# Logistic regression
results_train_glm1 = smf.glm(formula='crim01 ~ indus + nox + age', data=boston.iloc[list(X_train.index)], family=sm.families.Binomial()).fit()

# Threshold at 0.5
pred_test_glm1 = [0 if x < 0.5 else 1 for x in results_train_glm1.predict(boston.iloc[list(X_test.index)])]

# Result, initial threshold
print('GLM ~ indus + nox + age')
conf1 = confusion_matrix(boston.iloc[list(X_test.index)]['crim01'], pred_test_glm1)
print(conf1)
print('Classification rate:', (conf1[0,0] + conf1[1,1])/conf1.sum())
print('TPR:', conf1[1, 1]/(conf1[:, 1].sum()))
print('FPR:', conf1[1, 0]/(conf1[:, 0].sum()))
print('AUC: %f' % roc_auc_score(boston.iloc[list(X_test.index)]['crim01'], results_train_glm1.predict(boston.iloc[list(X_test.index)])))

# ROC
lr_fpr1, lr_tpr1, thresholds1 = roc_curve(boston.iloc[list(X_test.index)]['crim01'], results_train_glm1.predict(boston.iloc[list(X_test.index)]))

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


# Linear discriminant analysis
results_train_lda1 = LinearDiscriminantAnalysis().fit(X_train, y_train)
pred_test_lda1 = results_train_lda1.predict(X_test)
prob_test_lda1 = results_train_lda1.predict_proba(X_test)

# Result, initial threshold
print('LDA ~ indus + nox + age')
conf1 = confusion_matrix(boston.iloc[list(X_test.index)]['crim01'], pred_test_lda1)
print(conf1)
print('Classification rate:', (conf1[0,0] + conf1[1,1])/conf1.sum())
print('TPR:', conf1[1, 1]/(conf1[:, 1].sum()))
print('FPR:', conf1[1, 0]/(conf1[:, 0].sum()))
print('AUC: %f' % roc_auc_score(boston.iloc[list(X_test.index)]['crim01'], results_train_lda1.predict(X_test)))

# ROC
lr_fpr1, lr_tpr1, thresholds1 = roc_curve(boston.iloc[list(X_test.index)]['crim01'], prob_test_lda1[:,1])

# ROC dataframes
roc_lda1 = pd.DataFrame({
    'fpr': pd.Series(lr_fpr1, index=np.arange(len(lr_tpr1))),
    'tpr': pd.Series(lr_tpr1, index=np.arange(len(lr_tpr1))),
    '1-fpr': pd.Series(1-lr_fpr1, index=np.arange(len(lr_tpr1))),
    'tf': pd.Series(lr_tpr1-(1-lr_fpr1), index=np.arange(len(lr_tpr1))),
    'thresholds': pd.Series(thresholds1, index=np.arange(len(lr_tpr1)))
})

# Quadratic discriminant analysis
results_train_qda1 = QuadraticDiscriminantAnalysis().fit(X_train, y_train)
pred_test_qda1 = results_train_qda1.predict(X_test)
prob_test_qda1 = results_train_qda1.predict_proba(X_test)

# Result, initial threshold
print('QDA ~ indus + nox + age')
conf1 = confusion_matrix(boston.iloc[list(X_test.index)]['crim01'], pred_test_qda1)
print(conf1)
print('Classification rate:', (conf1[0,0] + conf1[1,1])/conf1.sum())
print('TPR:', conf1[1, 1]/(conf1[:, 1].sum()))
print('FPR:', conf1[1, 0]/(conf1[:, 0].sum()))
print('AUC: %f' % roc_auc_score(boston.iloc[list(X_test.index)]['crim01'], results_train_qda1.predict(X_test)))

# ROC
lr_fpr1, lr_tpr1, thresholds1 = roc_curve(boston.iloc[list(X_test.index)]['crim01'], prob_test_qda1[:,1])

# ROC dataframes
roc_qda1 = pd.DataFrame({
    'fpr': pd.Series(lr_fpr1, index=np.arange(len(lr_tpr1))),
    'tpr': pd.Series(lr_tpr1, index=np.arange(len(lr_tpr1))),
    '1-fpr': pd.Series(1-lr_fpr1, index=np.arange(len(lr_tpr1))),
    'tf': pd.Series(lr_tpr1-(1-lr_fpr1), index=np.arange(len(lr_tpr1))),
    'thresholds': pd.Series(thresholds1, index=np.arange(len(lr_tpr1)))
})


# K-Nearest Neighbor classification
results_train_knn1 = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
results_train_knn2 = KNeighborsClassifier(n_neighbors=2).fit(X_train, y_train)
results_train_knn10 = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train)

pred_test_knn1 = results_train_knn1.predict(X_test)
pred_test_knn2 = results_train_knn2.predict(X_test)
pred_test_knn10 = results_train_knn10.predict(X_test)

prob_test_knn1 = results_train_knn1.predict_proba(X_test)
prob_test_knn2 = results_train_knn2.predict_proba(X_test)
prob_test_knn10 = results_train_knn10.predict_proba(X_test)

# ROC
lr_fpr1, lr_tpr1, thresholds1 = roc_curve(boston.iloc[list(X_test.index)]['crim01'], prob_test_knn1[:,1])
lr_fpr2, lr_tpr2, thresholds2 = roc_curve(boston.iloc[list(X_test.index)]['crim01'], prob_test_knn2[:,1])
lr_fpr10, lr_tpr10, thresholds10 = roc_curve(boston.iloc[list(X_test.index)]['crim01'], prob_test_knn10[:,1])

# ROC dataframes
roc_knn1 = pd.DataFrame({
    'fpr': pd.Series(lr_fpr1, index=np.arange(len(lr_tpr1))),
    'tpr': pd.Series(lr_tpr1, index=np.arange(len(lr_tpr1))),
    '1-fpr': pd.Series(1-lr_fpr1, index=np.arange(len(lr_tpr1))),
    'tf': pd.Series(lr_tpr1-(1-lr_fpr1), index=np.arange(len(lr_tpr1))),
    'thresholds': pd.Series(thresholds1, index=np.arange(len(lr_tpr1)))
})
roc_knn2 = pd.DataFrame({
    'fpr': pd.Series(lr_fpr2, index=np.arange(len(lr_tpr2))),
    'tpr': pd.Series(lr_tpr2, index=np.arange(len(lr_tpr2))),
    '1-fpr': pd.Series(1-lr_fpr2, index=np.arange(len(lr_tpr2))),
    'tf': pd.Series(lr_tpr2-(1-lr_fpr2), index=np.arange(len(lr_tpr2))),
    'thresholds': pd.Series(thresholds2, index=np.arange(len(lr_tpr2)))
})
roc_knn10 = pd.DataFrame({
    'fpr': pd.Series(lr_fpr10, index=np.arange(len(lr_tpr10))),
    'tpr': pd.Series(lr_tpr10, index=np.arange(len(lr_tpr10))),
    '1-fpr': pd.Series(1-lr_fpr10, index=np.arange(len(lr_tpr10))),
    'tf': pd.Series(lr_tpr10-(1-lr_fpr10), index=np.arange(len(lr_tpr10))),
    'thresholds': pd.Series(thresholds10, index=np.arange(len(lr_tpr10)))
})

# Plot all ROC curves
sns.lineplot(x='fpr', y='tpr', data=roc_glm1, ci=None)
sns.lineplot(x='fpr', y='tpr', data=roc_lda1, ci=None)
sns.lineplot(x='fpr', y='tpr', data=roc_qda1, ci=None)
sns.lineplot(x='fpr', y='tpr', data=roc_knn1, ci=None)
sns.lineplot(x='fpr', y='tpr', data=roc_knn2, ci=None)
sns.lineplot(x='fpr', y='tpr', data=roc_knn10, ci=None)
plt.plot([x for x in np.arange(0, 1.1, 0.1)], [x for x in np.arange(0, 1.1, 0.1)], 'b--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(labels=['GLM', 'LDA', 'QDA', 'KNN1', 'KNN2', 'KNN10'])










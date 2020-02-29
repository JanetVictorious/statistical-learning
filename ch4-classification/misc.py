
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve

# Regression libs
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score


import statsmodels.formula.api as smf

os.getcwd()
os.chdir('/Users/viktor.eriksson2/Documents/python_files/miscellanious/stanford-statistical-learning')
os.getcwd()

sns.set()

auto = pd.read_csv('data/auto.csv')

med_mpg = auto['mpg'].median()

auto['med_mpg'] = [0 if x < med_mpg else 1 for x in auto['mpg']]

plt.figure(figsize=(10,7))
sns.scatterplot(x='horsepower', y='weight', data=auto, hue='med_mpg')

plt.figure(figsize=(10,7))
plt.subplot(2,1,1)
sns.scatterplot(x='horsepower', y='med_mpg', data=auto, hue='med_mpg')
plt.subplot(2,1,2)
sns.scatterplot(x='weight', y='med_mpg', data=auto, hue='med_mpg')

X = auto[['horsepower', 'weight']]
X = sm.add_constant(X)
y = auto[['med_mpg']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

fit1 = sm.GLM(y_train, X_train, family=sm.families.Binomial()).fit()

print(fit1.summary())

pred1 = fit1.predict(X_test)

pred_nom1 = [0 if x < 0.5 else 1 for x in pred1]

confusion_matrix(y_test, pred_nom1)

lr_fpr1, lr_tpr1, thresholds1 = roc_curve(y_test, pred1)

roc_glm1 = pd.DataFrame({
    'fpr': pd.Series(lr_fpr1, index=np.arange(len(lr_tpr1))),
    'tpr': pd.Series(lr_tpr1, index=np.arange(len(lr_tpr1))),
    '1-fpr': pd.Series(1-lr_fpr1, index=np.arange(len(lr_tpr1))),
    'tf': pd.Series(lr_tpr1-(1-lr_fpr1), index=np.arange(len(lr_tpr1))),
    'thresholds': pd.Series(thresholds1, index=np.arange(len(lr_tpr1)))
})

roc_glm1

plt.figure(figsize=(10,7))
sns.lineplot(x='fpr', y='tpr', data=roc_glm1, ci=None)
plt.plot([x for x in np.arange(0, 1.1, 0.1)], [x for x in np.arange(0, 1.1, 0.1)], 'b--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
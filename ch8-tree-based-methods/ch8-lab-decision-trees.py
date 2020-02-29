# -----------------------------------------
# Chapter 8 - Lab: Decision Trees
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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import tree
from sklearn.metrics import confusion_matrix, mean_squared_error, make_scorer
import sklearn.ensemble as ens

import graphviz


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
os.chdir('/Users/viktor.eriksson2/Documents/python_files/miscellanious/stanford-statistical-learning')
os.getcwd()

# Plot settings
sns.set()


#%% -----------------------------------------
# Load data
# -------------------------------------------

# Datasets from ISLR
carseats = pd.read_csv('data/carseats.csv')
carseats.info()
carseats.head(10)

boston = pd.read_csv('data/boston.csv')
boston.info()
boston.head(10)


#%% -----------------------------------------
# Functions
# -------------------------------------------


#%% -----------------------------------------
# Data prepping
# -------------------------------------------

#######################
# Carseats dataset
#######################

# Check null values
carseats.isna().sum()

# Plot distribution of Sales variable
plt.figure(figsize=(10, 7))
sns.distplot(carseats['Sales'])

# Mean and median of Sales variable
carseats['Sales'].mean()
carseats['Sales'].median()

# Create binary variable indicating if Sales are high or not
# carseats['high'] = [0 if x <= 8 else 1 for x in carseats['Sales']]
carseats['high'] = carseats['Sales'].map(lambda x: 0 if x <= 8 else 1)
carseats.head(10)

# Create dummy variables for categorical predictors
dummies = pd.get_dummies(carseats[['ShelveLoc', 'Urban', 'US']])

# Create new dataframe with predictiors and response
y = carseats[['high']]
X_ = carseats.drop(columns=['Sales', 'high', 'ShelveLoc', 'Urban', 'US'])
X = pd.concat([X_, dummies[['ShelveLoc_Bad', 'ShelveLoc_Good', 'ShelveLoc_Medium', 'Urban_Yes', 'US_Yes']]], axis=1)

# Pairplot
plt.figure(figsize=(10, 7))
sns.pairplot(pd.concat([y, X], axis=1), hue='high')

#######################
# Boston dataset
#######################

# Check null values
boston.isna().sum()

# Pairplot
plt.figure(figsize=(10, 7))
sns.pairplot(boston)


#%% -----------------------------------------
# Fitting classification trees
# -------------------------------------------

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Fit tree classifier
clf_tree = tree.DecisionTreeClassifier(max_depth=6).fit(X_train, y_train)

# Score
clf_tree.score(X_train, y_train)
clf_tree.score(X_test, y_test)

# Confusion matrix
cnf_mat = confusion_matrix(y_test, clf_tree.predict(X_test))
cnf_mat[1,1]/cnf_mat[:,1].sum() # true positive rate
cnf_mat[1,0]/cnf_mat[:,0].sum() # false positive rate
np.diag(cnf_mat).sum()/cnf_mat.sum() # classification rate

clf_tree.tree_.node_count
clf_tree.tree_.max_depth

# Visualize
dot_data = tree.export_graphviz(clf_tree, out_file=None,
                                feature_names=X_train.columns,  
                                class_names=['0', '1'],  
                                filled=True, rounded=True)

graph = graphviz.Source(dot_data)
graph
# display(HTML(graph._repr_svg_()))

# Plot most important features in model
plt.figure(figsize=(10, 7))
sns.barplot(x='importance', y='feature', data=pd.DataFrame({
    'feature': X.columns, 'importance': clf_tree.feature_importances_
}).sort_values(['importance'], ascending=False), color='b')


#######################
# Tree pruning
#######################

clf_pr = tree.DecisionTreeClassifier()
path = clf_pr.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

plt.figure(figsize=(10, 7))
sns.lineplot(x=ccp_alphas[:-1], y=impurities[:-1])
plt.xlabel('effective alpha')
plt.ylabel('total impurity of leaves')
plt.title('Total Impurity vs effective alpha for training set')

clfs = []
RMSe = []

for a in ccp_alphas:
    clf = tree.DecisionTreeClassifier(ccp_alpha=a)
    
    # Cross validation
    scores = cross_val_score(clf, X_train, y_train, cv=7)
    rmse = np.mean(scores)
    ci = np.std(scores)*1.96

    # Fit to train data
    clf.fit(X_train, y_train)
    no_nodes = clf.tree_.node_count

    # Predict on test data
    y_pred = clf.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)

    # Save results
    RMSe += [[rmse, rmse+ci, rmse-ci, test_mse, no_nodes, a]]
    clfs.append(clf)
    

clfs
RMSe

print('Number of nodes in the last tree is: {} with ccp_alpha: {}'.format(
      clfs[-1].tree_.node_count, ccp_alphas[-1]))

clfs[-1].tree_.max_depth

# Plot results
plt.figure(figsize=(10, 7))
plt.subplot(3, 1, 1)
sns.lineplot(x=ccp_alphas[:-1], y=[x.tree_.node_count for x in clfs[:-1]])
plt.xlabel('alphas')
plt.ylabel('nr of nodes')

plt.subplot(3, 1, 2)
sns.lineplot(x=ccp_alphas[:-1], y=[x.tree_.max_depth for x in clfs[:-1]])
plt.xlabel('alphas')
plt.ylabel('depth of tree')

plt.subplot(3, 1, 3)
sns.lineplot(x=ccp_alphas[:-1], y=pd.DataFrame(RMSe).iloc[:-1,0])
plt.xlabel('alphas')
plt.ylabel('MSE')

# Plot MSE against number of nodes
plt.figure(figsize=(10, 7))
sns.lineplot(x=[x.tree_.node_count for x in clfs[:-1]], y=pd.DataFrame(RMSe).iloc[:-1,0], label='MSE')
sns.lineplot(x=[x.tree_.node_count for x in clfs[:-1]], y=pd.DataFrame(RMSe).iloc[:-1,1], label='CI upper')
sns.lineplot(x=[x.tree_.node_count for x in clfs[:-1]], y=pd.DataFrame(RMSe).iloc[:-1,2], label='CI lower')
plt.xlabel('nr of nodes')
plt.ylabel('MSE')

train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

plt.figure(figsize=(10, 7))
sns.lineplot(x=ccp_alphas[:-1], y=[x.score(X_train, y_train) for x in clfs[:-1]], label='train')
sns.lineplot(x=ccp_alphas[:-1], y=[x.score(X_test, y_test) for x in clfs[:-1]], label='test')
plt.xlabel('alphas')
plt.ylabel('accuracy')

# Min MSE
MSE_df = pd.DataFrame(RMSe)
MSE_df[MSE_df.iloc[:,0] == MSE_df.iloc[:,0].min()]

# Tree corresponding to smallest MSE
opt_clf = clfs[388]
opt_clf.fit(X_train, y_train)
opt_clf.tree_.max_depth
opt_clf.tree_.node_count

y_pred = opt_clf.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # ~3.49
"""
This means that the optimal model from our training set yields a model which 
indicates that this model leads to test predictions that are within 
$3,490 of the true median home value for the suburb.
"""


#%% -----------------------------------------
# Fitting regression trees
# -------------------------------------------

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(boston.drop(columns=['medv']), boston[['medv']], test_size=0.1, random_state=1)

# Fit regression tree
regr_tree = tree.DecisionTreeRegressor(max_depth=6).fit(X_train, y_train)

# Prediction
y_pred = regr_tree.predict(X_test)

# Root mean-square-error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE test: {}'.format(np.round(rmse, 2)))

# Visualise the tree with GraphViz
dot_data = tree.export_graphviz(regr_tree, out_file=None,
                                feature_names=X_train.columns, 
                                filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph

regr_tree.tree_.node_count
regr_tree.tree_.max_depth

#######################
# Tree pruning
#######################

regr_pr = tree.DecisionTreeRegressor()
path = regr_pr.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

plt.figure(figsize=(10, 7))
sns.lineplot(x=ccp_alphas[:-1], y=impurities[:-1])
plt.xlabel('effective alpha')
plt.ylabel('total impurity of leaves')
plt.title('Total Impurity vs effective alpha for training set')

clfs = []
RMSe = []

mse = make_scorer(mean_squared_error)

for a in ccp_alphas:
    clf = tree.DecisionTreeRegressor(ccp_alpha=a)
    
    # Cross validation
    scores = cross_val_score(clf, X_train, y_train, cv=7, scoring=mse)
    rmse = np.mean(scores)
    ci = np.std(scores)*1.96

    # Fit to train data
    clf.fit(X_train, y_train)
    no_nodes = clf.tree_.node_count

    # Predict on test data
    y_pred = clf.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)

    # Save results
    RMSe += [[rmse, rmse+ci, rmse-ci, test_mse, no_nodes, a]]
    clfs.append(clf)

clfs
RMSe

print('Number of nodes in the last tree is: {} with ccp_alpha: {}'.format(
      clfs[-1].tree_.node_count, ccp_alphas[-1]))

clfs[-1].tree_.max_depth

# Plot results
plt.figure(figsize=(10, 7))
plt.subplot(3, 1, 1)
sns.lineplot(x=ccp_alphas, y=[x.tree_.node_count for x in clfs])
plt.xlabel('alphas')
plt.ylabel('nr of nodes')

plt.subplot(3, 1, 2)
sns.lineplot(x=ccp_alphas, y=[x.tree_.max_depth for x in clfs])
plt.xlabel('alphas')
plt.ylabel('depth of tree')

plt.subplot(3, 1, 3)
sns.lineplot(x=ccp_alphas, y=pd.DataFrame(RMSe).iloc[:,0])
plt.xlabel('alphas')
plt.ylabel('MSE')

# Plot MSE against number of nodes
plt.figure(figsize=(10, 7))
sns.lineplot(x=[x.tree_.node_count for x in clfs], y=pd.DataFrame(RMSe).iloc[:,0], label='cross-validation')
sns.lineplot(x=[x.tree_.node_count for x in clfs], y=pd.DataFrame(RMSe).iloc[:,1], label='CI upper')
sns.lineplot(x=[x.tree_.node_count for x in clfs], y=pd.DataFrame(RMSe).iloc[:,2], label='CI lower')
plt.xlabel('nr of nodes')
plt.ylabel('MSE')

train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

plt.figure(figsize=(10, 7))
sns.lineplot(x=ccp_alphas, y=[x.score(X_train, y_train) for x in clfs], label='train')
sns.lineplot(x=ccp_alphas, y=[x.score(X_test, y_test) for x in clfs], label='test')
plt.xlabel('alphas')
plt.ylabel('accuracy')

# Min MSE
MSE_df = pd.DataFrame(RMSe)
MSE_df[MSE_df.iloc[:,0] == MSE_df.iloc[:,0].min()]
"""
We have two trees with the same MSE. We pick the one with the least
number of nodes
"""

# Tree corresponding to smallest MSE
opt_clf = clfs[388]
opt_clf.fit(X_train, y_train)
opt_clf.tree_.max_depth
opt_clf.tree_.node_count

y_pred = opt_clf.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # ~3.49
"""
This means that the optimal model from our training set yields a model which 
indicates that this model leads to test predictions that are within 
$3,490 of the true median home value for the suburb.
"""


#%% -----------------------------------------
# Bagging and random forest
# -------------------------------------------

# Split data into training and test sets
X = boston.drop(columns=['medv'])
y = boston[['medv']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Bagging regression. Same as random forest but all predictors are used
regr_bg = ens.RandomForestRegressor(max_features=X_train.shape[1], n_estimators=100, random_state=0).fit(X_train, y_train)

# Prediction
y_pred = regr_bg.predict(X_test)

print('The MSE is: {}'.format(mean_squared_error(y_test, y_pred)))
print('The RMSE is: {}'.format(np.sqrt(mean_squared_error(y_test, y_pred))))
"""
Bagging significantly improves the RMSE
"""

# Random forest with 6 predictors
regr_rf = ens.RandomForestRegressor(max_features=6, n_estimators=100, random_state=0).fit(X_train, y_train)

# Prediction
y_pred = regr_rf.predict(X_test)

print('The MSE is: {}'.format(mean_squared_error(y_test, y_pred)))
print('The RMSE is: {}'.format(np.sqrt(mean_squared_error(y_test, y_pred))))
"""
Random forest yields slightly higher results than bagging but almost the same
"""

# Compare different forest algorithms
results = []

for i in np.arange(2, 101):
    # Tree algorithms
    regr_bg = ens.RandomForestRegressor(max_features=X_train.shape[1], n_estimators=100, oob_score=True, max_leaf_nodes=i)
    regr_rf1 = ens.RandomForestRegressor(max_features=4, n_estimators=100, oob_score=True, max_leaf_nodes=i)
    regr_rf2 = ens.RandomForestRegressor(max_features=6, n_estimators=100, oob_score=True, max_leaf_nodes=i)

    # Fit models
    fit_bg = regr_bg.fit(X_train, y_train)
    fit_rf1 = regr_rf1.fit(X_train, y_train)
    fit_rf2 = regr_rf2.fit(X_train, y_train)

    # Predictions
    pred_bg = fit_bg.predict(X_test)
    pred_rf1 = fit_rf1.predict(X_test)
    pred_rf2 = fit_rf2.predict(X_test)

    # MSE
    mse_bg = mean_squared_error(y_test, pred_bg)
    mse_rf1 = mean_squared_error(y_test, pred_rf1)
    mse_rf2 = mean_squared_error(y_test, pred_rf2)

    # OOBs
    oob_bg = fit_bg.oob_score_
    oob_rf1 = fit_rf1.oob_score_
    oob_rf2 = fit_rf2.oob_score_

    # Results
    results += [[i, mse_bg, mse_rf1, mse_rf2, oob_bg, oob_rf1, oob_rf2]]

res_df = pd.DataFrame(results, columns=['no_leafs', 'mse_bg', 'mse_rf1', 'mse_rf2', 'oob_bg', 'oob_rf1', 'oob_rf2'])

# Plot results
plt.figure(figsize=(10,7))
plt.subplot(2, 1, 1)
sns.lineplot(x='no_leafs', y='mse_bg', data=res_df, label='bagging')
sns.lineplot(x='no_leafs', y='mse_rf1', data=res_df, label='random forest m=4')
sns.lineplot(x='no_leafs', y='mse_rf2', data=res_df, label='random forest m=6')
plt.ylabel('MSE')

plt.subplot(2, 1, 2)
sns.lineplot(x='no_leafs', y='oob_bg', data=res_df, label='bagging')
sns.lineplot(x='no_leafs', y='oob_rf1', data=res_df, label='random forest m=4')
sns.lineplot(x='no_leafs', y='oob_rf2', data=res_df, label='random forest m=6')
plt.ylabel('OOB score')
"""
Trying different sizes of training data will give different results.
For a larger training set, bagging seems to perform better.
For smaller training sets, random forest might be a better option.

It looks like about 20 leafs gives a good enough prediction
"""

# Plot most important features for each model
fit_bg = ens.RandomForestRegressor(max_features=X_train.shape[1], n_estimators=100, oob_score=True, max_leaf_nodes=20).fit(X_train, y_train)
fit_rf1 = ens.RandomForestRegressor(max_features=4, n_estimators=100, oob_score=True, max_leaf_nodes=20).fit(X_train, y_train)
fit_rf2 = ens.RandomForestRegressor(max_features=6, n_estimators=100, oob_score=True, max_leaf_nodes=20).fit(X_train, y_train)

plt.figure(figsize=(10, 15))
plt.subplot(3, 1, 1)
sns.barplot(x='importance', y='feature', data=pd.DataFrame({
    'feature': X.columns, 'importance': fit_bg.feature_importances_
}).sort_values(['importance'], ascending=False), color='b')
plt.title('Bagging')
plt.xlabel('')

plt.subplot(3, 1, 2)
sns.barplot(x='importance', y='feature', data=pd.DataFrame({
    'feature': X.columns, 'importance': fit_rf1.feature_importances_
}).sort_values(['importance'], ascending=False), color='b')
plt.title('Random forest m=4')
plt.xlabel('')

plt.subplot(3, 1, 3)
sns.barplot(x='importance', y='feature', data=pd.DataFrame({
    'feature': X.columns, 'importance': fit_rf2.feature_importances_
}).sort_values(['importance'], ascending=False), color='b')
plt.title('Random forest m=6')


# # OOB error rates
# ensemble_clfs = [
#     ("RandomForestClassifier, max_features=13",
#         ens.RandomForestClassifier(warm_start=True, oob_score=True,
#                                max_features=X_train.shape[1],
#                                random_state=1)),
#     ("RandomForestClassifier, max_features=4",
#         ens.RandomForestClassifier(warm_start=True, max_features=4,
#                                oob_score=True,
#                                random_state=1)),
#     ("RandomForestClassifier, max_features=6",
#         ens.RandomForestClassifier(warm_start=True, max_features=6,
#                                oob_score=True,
#                                random_state=1))
# ]

# # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
# error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

# # Range of `n_estimators` values to explore.
# min_estimators = 2
# max_estimators = 13
# OOBs = pd.DataFrame({
#     'Bagging': [],
#     'Random forest 1': [],
#     'Random forest 2': []
# })

# OOBs

# for clf in ensemble_clfs:
#     for i in range(min_estimators, max_estimators + 1):
#         clf.set_params(n_estimators=i)
#         clf.fit(X, y)

#         # Record the OOB error for each `n_estimators=i` setting.
#         oob_error = 1 - clf.oob_score_
#         pd.DataFrame()
#         error_rate[label].append((i, oob_error))


#%% -----------------------------------------
# Boosting
# -------------------------------------------

# Split data into training and test sets
X = boston.drop(columns=['medv'])
y = boston[['medv']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=3)

# Boosting regression
regr_boo = ens.GradientBoostingRegressor(max_features='auto', n_estimators=5000, max_depth=4, learning_rate=0.01).fit(X_train, y_train)

# Predict
y_pred = regr_boo.predict(X_test)

# MSE
print('The MSE for the boosted tree is: {}'.format(mean_squared_error(y_test, y_pred)))
print('The RMSE for the boosted tree is: {}'.format(np.sqrt(mean_squared_error(y_test, y_pred))))
"""
Using a boosted tree on the data yields results of the magnitude of bagging and random forest
"""

# Try boosting with different depths
MSEs = []
for i in np.arange(2, 201):
    # Models
    regr_boo1 = ens.GradientBoostingRegressor(max_features='auto', n_estimators=i, max_depth=4, learning_rate=0.08).fit(X_train, y_train)
    regr_boo2 = ens.GradientBoostingRegressor(max_features='auto', n_estimators=i, max_depth=6, learning_rate=0.08).fit(X_train, y_train)
    regr_boo3 = ens.GradientBoostingRegressor(max_features='sqrt', n_estimators=i, max_depth=6, learning_rate=0.08).fit(X_train, y_train)
    regr_boo4 = ens.GradientBoostingRegressor(max_features='log2', n_estimators=i, max_depth=6, learning_rate=0.08).fit(X_train, y_train)

    # Predictions
    mse1 = mean_squared_error(y_test, regr_boo1.predict(X_test))
    mse2 = mean_squared_error(y_test, regr_boo2.predict(X_test))
    mse3 = mean_squared_error(y_test, regr_boo3.predict(X_test))
    mse4 = mean_squared_error(y_test, regr_boo4.predict(X_test))

    # Results
    MSEs += [[i, mse1, mse2, mse3, mse4]]

res_df = pd.DataFrame(MSEs, columns=['no_trees', 'mse1', 'mse2', 'mse3', 'mse4'])
res_df

# Plot results
plt.figure(figsize=(10, 7))
sns.lineplot(x='no_trees', y='mse1', data=res_df, label='boosting depth=4')
sns.lineplot(x='no_trees', y='mse2', data=res_df, label='boosting depth=6')
sns.lineplot(x='no_trees', y='mse3', data=res_df, label="boosting depth=6, features='sqrt'")
sns.lineplot(x='no_trees', y='mse4', data=res_df, label="boosting depth=6, features='log2'")
plt.xlabel('number of trees')
plt.ylabel('MSE')

# Lets try all again but for just 1 iteration with 5000 trees
regr_boo1 = ens.GradientBoostingRegressor(max_features='auto', n_estimators=5000, max_depth=4, learning_rate=0.05).fit(X_train, y_train)
regr_boo2 = ens.GradientBoostingRegressor(max_features='auto', n_estimators=5000, max_depth=6, learning_rate=0.05).fit(X_train, y_train)
regr_boo3 = ens.GradientBoostingRegressor(max_features='sqrt', n_estimators=5000, max_depth=6, learning_rate=0.05).fit(X_train, y_train)
regr_boo4 = ens.GradientBoostingRegressor(max_features='log2', n_estimators=5000, max_depth=6, learning_rate=0.05).fit(X_train, y_train)

print('The RMSE for auto, depth=4 is: {}'.format(np.sqrt(mean_squared_error(y_test, regr_boo1.predict(X_test)))))
print('The RMSE for auto, depth=6 is: {}'.format(np.sqrt(mean_squared_error(y_test, regr_boo2.predict(X_test)))))
print('The RMSE for sqrt, depth=6 is: {}'.format(np.sqrt(mean_squared_error(y_test, regr_boo3.predict(X_test)))))
print('The RMSE for log2, depth=6 is: {}'.format(np.sqrt(mean_squared_error(y_test, regr_boo4.predict(X_test)))))


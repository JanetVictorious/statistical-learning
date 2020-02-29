
#%% -----------------------------------------
# Import packages
# -------------------------------------------

# Standard libs
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
import time

# Regression libs
from sklearn import linear_model
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
import statsmodels.api as sm
import xgboost as xgb
from sklearn import tree
import sklearn.ensemble as ens

# Dataset libs
from sklearn.datasets import load_boston

import graphviz


#%% -----------------------------------------
# Settings
# -------------------------------------------

# Set working directory
# os.getcwd()
# os.chdir(os.getcwd()+'/'+'miscellanious/stanford-statistical-learning')
# os.chdir('/Users/viktor.eriksson2/Documents/python_files/miscellanious/stanford-statistical-learning')
# os.getcwd()

# Plot settings
sns.set()


#%% -----------------------------------------
# Load data
# -------------------------------------------

# Datasets from ISLR
# hitters = pd.read_csv('data/hitters.csv')
# hitters.info()
# hitters.head(10)

boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
boston['PRICE'] = load_boston().target

boston.isna().sum()
boston.head(10)


#%% -----------------------------------------
# Functions
# -------------------------------------------

def processSubsetGLM(feature_set):
    # Fit model on feature_set and calculate score
    features = list(feature_set)
    regr_lg = linear_model.LogisticRegression().fit(X_train[list(feature_set)], y_train)
    # Instead of RSS we use accuracy score
    score = accuracy_score(y_test, regr_lg.predict(X_test[list(feature_set)]))
    return {'model':regr_lg, 'score':score, 'variables':features}

def processSubsetCrossValidation(feature_set, k_fold):
    # Fit model on feature_set and calculate RSS
    features = list(feature_set)
    regr_lg = linear_model.LogisticRegression().fit(X_train[list(feature_set)], y_train)
    # 5 fold cross validation
    cv_results1 = cross_val_score(linear_model.LogisticRegression(), X_train[list(feature_set)], y_train, scoring='accuracy', cv=k_fold)
    score = np.mean(cv_results1)
    return {'model':regr_lg, 'score':score, 'variables':features}

# Forward stepwise selection function
def forward(predictors):
    # Pull out predictors we still need to process
    remaining_predictors = [p for p in X_train.columns if p not in predictors]
    tic = time.time()
    results = []
    for p in remaining_predictors:
        results.append(processSubsetGLM(predictors+[p]))
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    # Choose the model with the highest RSS
    best_model = models.loc[models['score'].argmax()]
    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors)+1, "predictors in", (toc-tic), "seconds.")
    # Return the best model, along with some other useful information about the model
    return best_model

def forwardCrossValidation(predictors, k_fold):
    # Pull out predictors we still need to process
    remaining_predictors = [p for p in X_train.columns if p not in predictors]
    tic = time.time()
    results = []
    for p in remaining_predictors:
        results.append(processSubsetCrossValidation(predictors+[p], k_fold))
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    # Choose the model with the highest RSS
    best_model = models.loc[models['score'].argmax()]
    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors)+1, "predictors in", (toc-tic), "seconds.")
    # Return the best model, along with some other useful information about the model
    return best_model


#%% -----------------------------------------
# Data prepping
# -------------------------------------------

plt.figure(figsize=(10, 7))
sns.distplot(boston['CRIM'])

med_crim = np.median(boston['CRIM'])

boston['med_crim01'] = [0 if x < med_crim else 1 for x in boston['CRIM']]

# Pair plot
plt.figure(figsize=(20, 14))
sns.pairplot(boston, hue='med_crim01', diag_kind='hist')

# Regression variables
X = boston.drop(columns=['med_crim01', 'CRIM'])
y = boston['med_crim01']

# Split data into training and test sets
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Data matrix for XGBoost
dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=X_test, label=y_test)

####################################
# Validation-set approach
####################################

# --------------------------
# Logistic regression
# --------------------------

regr_val_lg = linear_model.LogisticRegression().fit(X_train, y_train)

# Summary of regression
pd.DataFrame(regr_val_lg.coef_, columns=X_train.columns)
regr_val_lg.intercept_
regr_val_lg.n_iter_

# ROC
lr_fpr1, lr_tpr1, thresholds1 = roc_curve(y_test, regr_val_lg.predict_proba(X_test)[:,1])

# ROC dataframe
roc_val_lg = pd.DataFrame({
    'fpr': pd.Series(lr_fpr1, index=np.arange(len(lr_tpr1))),
    'tpr': pd.Series(lr_tpr1, index=np.arange(len(lr_tpr1))),
    '1-fpr': pd.Series(1-lr_fpr1, index=np.arange(len(lr_tpr1))),
    'tf': pd.Series(lr_tpr1-(1-lr_fpr1), index=np.arange(len(lr_tpr1))),
    'threshold': pd.Series(thresholds1, index=np.arange(len(lr_tpr1)))
})

del lr_fpr1, lr_tpr1, thresholds1

# Precision-recall
precision1, recall1, thresholds1 = precision_recall_curve(y_test, regr_val_lg.predict_proba(X_test)[:,1])

# Precision-recall dataframe
prc_val_lg = pd.DataFrame({
    'precision': pd.Series(precision1, index=np.arange(len(precision1))),
    'recall': pd.Series(recall1, index=np.arange(len(recall2))),
    'threshold': pd.Series(thresholds1, index=np.arange(len(thresholds1)))
})

del precision1, recall1, thresholds1

# Plot ROC and PRC
plt.figure(figsize=(10, 10))

plt.subplot(2, 1, 1)
sns.lineplot(x='fpr', y='tpr', data=roc_val_lg, ci=None)
plt.plot([x for x in np.arange(0, 1.1, 0.1)], [x for x in np.arange(0, 1.1, 0.1)], 'b--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.subplot(2, 1, 2)
sns.lineplot(x='recall', y='precision', data=prc_val_lg, ci=None)

# # Optimal threshold
# opt_tr = roc_val_lg['threshold'].ix[(roc_val_lg['tf']-0).abs().argsort()[:1]].iloc[0]

# Predictions
pred_val_lg = regr_val_lg.predict(X_test)
accuracy_score(y_test, pred_val_lg)
f1_score(y_test, pred_val_lg)

# --------------------------
# Forward selection
# --------------------------

models_fwd_lg = pd.DataFrame(columns=['model', 'score', 'variables'])

tic = time.time()
predictors = []
nr_features = []

for i in range(1,len(X_train.columns)+1):    
    models_fwd_lg.loc[i] = forward(predictors)
    # predictors = models_fwd_glm.loc[i]["model"].model.exog_names
    predictors = models_fwd_lg.loc[i]['variables']
    nr_features += [i]

del i, predictors

toc = time.time()
print('Total elapsed time:', (toc-tic)/60, 'minutes.')

# Plot score against nr variables
plt.figure(figsize=(10, 7))
sns.lineplot(x=nr_features, y=models_fwd_lg['score'], label='Logistic regression')
plt.xlabel('Nr features')
plt.ylabel('Score')
plt.title('Accuracy logistic regression forward selection')
"""
Forward selection with 8 predictors has the highes accuracy
"""

# Create ROC dataframe for each model
roc_fwd_df = pd.DataFrame(columns=['fpr', 'tpr', 'threshold', 'i'])

for i in range(1,len(X_train.columns)+1):
    lr_fpr1, lr_tpr1, thresholds1 = roc_curve(y_test, models_fwd_lg.loc[i, 'model'].predict_proba(X_test[list(models_fwd_lg.loc[i, 'variables'])])[:,1])
    it = [models_fwd_lg.loc[i, 'variables'] for x in range(len(lr_fpr1))]
    df = pd.DataFrame({'fpr':lr_fpr1, 'tpr':lr_tpr1, 'threshold':thresholds1, 'i':it})
    roc_fwd_df = roc_fwd_df.append(df)

del i, lr_fpr1, lr_tpr1, thresholds1, it, df

# Create variable to lable on in lineplot
roc_fwd_df['hue'] = ['$%s$' % x for x in roc_fwd_df['i']]

# Create PRC dataframe for each model
prc_fwd_df = pd.DataFrame(columns=['precision', 'recall', 'threshold', 'i'])

for i in range(1,len(X_train.columns)+1):
    precision1, recall1, thresholds1 = precision_recall_curve(y_test, models_fwd_lg.loc[i, 'model'].predict_proba(X_test[list(models_fwd_lg.loc[i, 'variables'])])[:,1])
    it = [models_fwd_lg.loc[i, 'variables'] for x in range(len(precision1))]
    df = pd.DataFrame({'precision':precision1, 'recall':recall1, 'i':it})
    prc_fwd_df = prc_fwd_df.append(df)

del i, precision1, recall1, thresholds1, it, df

# Create variable to lable on in lineplot
prc_fwd_df['hue'] = ['$%s$' % x for x in prc_fwd_df['i']]

# Plot ROC curves
plt.figure(figsize=(14, 10))
sns.lineplot(x='fpr', y='tpr', data=roc_fwd_df, hue='hue', ci=None)
plt.plot([x for x in np.arange(0, 1.1, 0.1)], [x for x in np.arange(0, 1.1, 0.1)], 'b--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for logistic regression using forward selection')

# Plot precision-recall curves
plt.figure(figsize=(14, 10))
sns.lineplot(x='recall', y='precision', data=prc_fwd_df, hue='hue', ci=None)
plt.title('Precision-recall curves for logistic regression using forward selection')

# Optimal forward model
opt_index = models_fwd_lg[models_fwd_lg['score'] == np.max(models_fwd_lg['score'])].index.values
opt_var = models_fwd_lg['variables'].loc[opt_index].iloc[0]

regr_fwd_lg = models_fwd_lg['model'].loc[opt_index].iloc[0]
pred_fwd_lg = regr_fwd_lg.predict(X_test[opt_var])
accuracy_score(y_test, pred_fwd_lg)
f1_score(y_test, pred_fwd_lg)

del opt_index, opt_var

# --------------------------
# Random forest
# --------------------------

# Tuning based on number of leafs
ACC_rf = []
for i in np.arange(2, 101):
    rf_classifier = ens.RandomForestClassifier(n_estimators=1000, max_depth=4, max_features='sqrt', random_state=123, max_leaf_nodes=i).fit(X_train, y_train)
    rf_pred = rf_classifier.predict(X_test)
    score = accuracy_score(y_test, rf_pred)
    ACC_rf += [[i, score]]

rf_df = pd.DataFrame(ACC_rf, columns=['no_leafs', 'score'])
del i, ACC_rf, rf_classifier, rf_pred, score

# Plot results
plt.figure(figsize=(10, 7))
sns.lineplot(x='no_leafs', y='score', data=rf_df, label='Random forest m=sqrt(p), trees=1000')
plt.title('Accuracy of random forest')
plt.xlabel('no leafs')

# Best accuracy achieved at 6 leafs

# Tuning on number of trees
ACC_rf = []
for i in np.arange(2, 2001, step=10):
    rf_classifier = ens.RandomForestClassifier(n_estimators=i, max_depth=4, max_features='sqrt', random_state=123, max_leaf_nodes=6).fit(X_train, y_train)
    rf_pred = rf_classifier.predict(X_test)
    score = accuracy_score(y_test, rf_pred)
    ACC_rf += [[i, score]]

rf_df2 = pd.DataFrame(ACC_rf, columns=['no_trees', 'score'])
del i, ACC_rf, rf_classifier, rf_pred, score

# Plot results
plt.figure(figsize=(10, 7))
sns.lineplot(x='no_trees', y='score', data=rf_df2, label='Random forest m=sqrt(p), leafs=6')
plt.title('Accuracy of random forest')
plt.xlabel('nr of trees')

# Simple approach optimal random forest classifier
cl_rf = ens.RandomForestClassifier(n_estimators=200, max_depth=4, max_features='sqrt', random_state=123, max_leaf_nodes=6).fit(X_train, y_train)
pred_rf = cl_rf.predict(X_test)
accuracy_score(y_test, pred_rf)

# Feature importance
plt.figure(figsize=(10, 7))
sns.barplot(x='importance', y='feature', data=pd.DataFrame({
    'feature': X_train.columns, 'importance': cl_rf.feature_importances_
}).sort_values(['importance'], ascending=False), color='b')
plt.title('Feature importance, random forest m=sqrt(p), nr_leafs=6, nr_trees=200')
plt.xlabel('')

# Plot tree
rf_tree = tree.export_graphviz(cl_rf.estimators_[1], out_file=None,
                                feature_names=X_train.columns,  
                                class_names=['0', '1'],  
                                filled=True, rounded=True)

graph = graphviz.Source(rf_tree)
graph


# --------------------------
# XGBoost
# --------------------------

cl_xgb = xgb.XGBClassifier(max_depth=6, learning_rate=0.07, n_estimators=1000, colsample_bytree=0.3, reg_alpha=4, random_state=123).fit(X_train, y_train)
pred_xgb = cl_xgb.predict(X_test)
accuracy_score(y_test, pred_xgb)

cl_xgb.feature_importances_
pd.Series(cl_xgb.get_booster().get_score(), index=[i for i in cl_xgb.get_booster().get_score()]).sort_values(ascending=False)

# Feature importance
plt.figure(figsize=(10, 7))
sns.barplot(
    x=pd.Series(cl_xgb.get_booster().get_score(), index=[i for i in cl_xgb.get_booster().get_score()]).sort_values(ascending=False).values,
    y=pd.Series(cl_xgb.get_booster().get_score(), index=[i for i in cl_xgb.get_booster().get_score()]).sort_values(ascending=False).index,
    color='b')
plt.title('Feature importance, XGBoost')
plt.xlabel('F score')

# Feature importance 2
xgb.plot_importance(cl_xgb)
plt.rcParams['figure.figsize'] = [10, 7]
plt.show()

# Results
print('-------------------------------')
print('RESULTS: Accuracy validation-set approach')
print('-------------------------------')
print('Logistic regression:                  ', round(accuracy_score(y_test, pred_val_lg), 3))
print('Logistic regression forward selection:', round(accuracy_score(y_test, pred_fwd_lg), 3))
print('Random forest:                        ', round(accuracy_score(y_test, pred_rf), 3))
print('XGBoost:                              ', round(accuracy_score(y_test, pred_xgb), 3))
print('-------------------------------')


####################################
# Cross-validation approach
####################################

# --------------------------
# Logistic regression
# --------------------------

# Compare cv score to validation-set score
np.mean(cross_val_score(linear_model.LogisticRegression(), X_train, y_train, cv=10, scoring='accuracy'))
accuracy_score(y_train, regr_val_lg.predict(X_train))
f1_score(y_train, regr_val_lg.predict(X_train))


# --------------------------
# Forward selection
# --------------------------

models_fwd_cv_lg = pd.DataFrame(columns=['model', 'score', 'variables'])

tic = time.time()
predictors = []

for i in range(1,len(X.columns)+1):    
    models_fwd_cv_lg.loc[i] = forwardCrossValidation(predictors, 10)
    predictors = models_fwd_cv_lg.loc[i]['variables']

toc = time.time()
print('Total elapsed time:', (toc-tic)/60, 'minutes.')

# Plot results
plt.figure(figsize=(10, 7))
sns.lineplot(x=models_fwd_cv_lg.index, y='score', data=models_fwd_cv_lg)
plt.xlabel('# predictors')
plt.ylabel('Accuracy')
plt.title('10-fold cross-validation - Logistic regression')
"""
Optimal accuracy is reached for 7 predictors
"""


# --------------------------
# XGBoost
# --------------------------

params_cv = {
    'objective':'binary:logistic', 
    'colsample_bytree': 0.3, 
    'learning_rate': 0.1, 
    'max_depth': 4, 
    'alpha': 4
}

xgb_cv_results = xgb.cv(params=params_cv, dtrain=dtrain, num_boost_round=3000, nfold=10, metrics='logloss', early_stopping_rounds=100, as_pandas=True, seed=123)
xgb_cv_results

# Plot results
plt.figure(figsize=(10, 7))
sns.lineplot(x=xgb_cv_results.index, y='train-logloss-mean', data=xgb_cv_results, label='train data')
sns.lineplot(x=xgb_cv_results.index, y='test-logloss-mean', data=xgb_cv_results, label='test data')
plt.xlabel('number of trees')
plt.ylabel('Log loss')


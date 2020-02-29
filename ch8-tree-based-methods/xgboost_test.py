
#%% -----------------------------------------
# Import packages
# -------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

import sklearn.ensemble as ens
import xgboost as xgb

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, mean_squared_error, make_scorer

#%% -----------------------------------------
# Settings
# -------------------------------------------

os.chdir('/Users/viktor.eriksson2/Documents/python_files/miscellanious/stanford-statistical-learning')
os.getcwd()

sns.set()


#%% -----------------------------------------
# Regression
# -------------------------------------------

boston = pd.read_csv('data/boston.csv')
boston.head(10)
boston.isna().sum()

X = boston.drop(columns=['medv'])
y = boston['medv']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=3)


# Comparing XGBoost with normal boosting and bagging

# Fitting
regr_xgb = xgb.XGBRegressor(n_estimators=5000, max_depth=6, learning_rate=0.08, random_state=0).fit(X_train, y_train)
regr_boo = ens.GradientBoostingRegressor(max_features='auto', n_estimators=5000, max_depth=6, learning_rate=0.08, random_state=0).fit(X_train, y_train)
regr_bg = ens.RandomForestRegressor(max_features=X_train.shape[1], n_estimators=5000, max_depth=6, random_state=0).fit(X_train, y_train)

# print(regr_xgb)

# Predictions
pred_xgb = regr_xgb.predict(X_test)
pred_boo = regr_boo.predict(X_test)
pred_bg = regr_bg.predict(X_test)

# RMSEs
print('The RMSE of XGBoost is: {}'.format(np.sqrt(mean_squared_error(y_test, pred_xgb))))
print('The RMSE of boosting is: {}'.format(np.sqrt(mean_squared_error(y_test, pred_boo))))
print('The RMSE of bagging is: {}'.format(np.sqrt(mean_squared_error(y_test, pred_bg))))


# Cross-validation
mse = make_scorer(mean_squared_error)
RMSEs = []
for i in np.arange(10, 251, step=5):
    # Models
    model_xgb = xgb.XGBRegressor(n_estimators=i, max_depth=6, learning_rate=0.08, random_state=0)
    model_boo = ens.GradientBoostingRegressor(max_features='auto', n_estimators=i, max_depth=6, learning_rate=0.08, random_state=0)
    model_bag = ens.RandomForestRegressor(max_features=X_train.shape[1], n_estimators=i, max_depth=6, random_state=0)

    # CV scores
    scores_xbg = cross_val_score(model_xgb, X_train, y_train, cv=5, scoring=mse)
    scores_boo = cross_val_score(model_boo, X_train, y_train, cv=5, scoring=mse)
    scores_bag = cross_val_score(model_bag, X_train, y_train, cv=5, scoring=mse)

    # Results
    RMSEs += [[i, np.sqrt(np.mean(scores_xbg)), np.sqrt(np.mean(scores_boo)), np.sqrt(np.mean(scores_bag))]]

res_df = pd.DataFrame(RMSEs, columns=['nr_trees', 'RMSE_xbg', 'RMSE_boo', 'RMSE_bag'])

# Plot results
plt.figure(figsize=(10, 7))
sns.lineplot(x='nr_trees', y='RMSE_xbg', data=res_df, label='XGBoost, depth=6')
sns.lineplot(x='nr_trees', y='RMSE_boo', data=res_df, label='Boosting, depth=6')
sns.lineplot(x='nr_trees', y='RMSE_bag', data=res_df, label='Bagging, depth=6')
plt.xlabel('number of trees')
plt.ylabel('RMSE')
plt.title('5-fold crossvalidation for different number of trees')
"""
Comments:
- Bagging doesn't improve notably when number of trees are increased
- Boosting stabilizes around 50 trees
- XGBoost stabilizes around 100 trees
"""

#%% -----------------------------------------
# Regression
# -------------------------------------------

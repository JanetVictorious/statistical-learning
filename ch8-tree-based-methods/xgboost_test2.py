
# Import packages
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load data
boston = load_boston()
boston.keys()
boston.feature_names
boston.target
print(boston.DESCR)

dt1 = pd.DataFrame(boston.data)
dt1.columns = boston.feature_names
dt1['PRICE'] = boston.target

dt1.info()
dt1.isna().sum()
pd.DataFrame(dt1.describe()).T

# Predictors and response
X = dt1.drop(columns=['PRICE'])
y = dt1['PRICE']

# Create train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Convert data to optimized data structure for better performance
dmatrix_train = xgb.DMatrix(data=X_train, label=y_train)
dmatrix_test = xgb.DMatrix(data=X_test, label=y_test)

# Create regression
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1, max_depth=5, alpha=10, n_estimators=1000)

# Fit model
xg_reg.fit(X_train, y_train)

# Predictions
xg_pred = xg_reg.predict(X_test)

# RMSE
print('The RMSE: %f' % (np.sqrt(mean_squared_error(y_test, xg_pred))))


# Cross-validation approach
params = {
    'objective':'reg:squarederror', 
    'colsample_bytree': 0.3, 
    'learning_rate': 0.1, 
    'max_depth': 5, 
    'alpha': 10
}

cv_results = xgb.cv(dtrain=dmatrix_train, params=params, nfold=5, num_boost_round=1000, early_stopping_rounds=10, metrics='rmse', as_pandas=True, seed=123)
cv_results

# Final boosting metric test results
cv_results.iloc[-1,2]

# Plot results
plt.figure(figsize=(10, 7))
sns.lineplot(x=cv_results.index, y='train-rmse-mean', data=cv_results, label='train data')
sns.lineplot(x=cv_results.index, y='test-rmse-mean', data=cv_results, label='test data')
plt.xlabel('number of trees')
plt.ylabel('RMSE')

# Visualize
xg_reg = xgb.train(params=params, dtrain=dmatrix_test, num_boost_round=100)

xgb.plot_tree(xg_reg, num_trees=3)
plt.rcParams['figure.figsize'] = [50, 10]
plt.show()

plt.figure(figsize=(10,7))
xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()

# Evaluate results for different alphas
alphas = 10**np.arange(-3, 2, step=0.1)

RMSe = []
for a in alphas:
    # Create regression
    xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1, max_depth=4, reg_alpha=a, n_estimators=1000).fit(X_train, y_train)

    # RMSE
    rmse = np.sqrt(mean_squared_error(y_test, xg_reg.predict(X_test)))

    # Results
    RMSe += [[a, rmse]]

rmse_df = pd.DataFrame(RMSe, columns=['alpha', 'rmse'])
rmse_df

# Plot results
plt.figure(figsize=(10, 7))
sns.lineplot(x='alpha', y='rmse', data=rmse_df)


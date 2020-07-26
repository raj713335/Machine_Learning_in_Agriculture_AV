import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost


# Read the data
X_train = pd.read_csv('DATA/train.csv', index_col='ID')
X_test = pd.read_csv('DATA/test.csv', index_col='ID')




# Remove rows with missing target, separate target from predictors
y = X_train.Crop_Damage
X_train.drop(['Crop_Damage'], axis=1, inplace=True)


# Understanding the shape of tabular data

print(X_train.head())
print(X_test.head())

print(X_train.shape)
print(X_test.shape)



# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y, train_size=0.99, test_size=0.01,
                                                      random_state=0)



# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])






# Imputation

from sklearn.impute import SimpleImputer

# Fill in the lines below: imputation
my_imputer = SimpleImputer(strategy='median') # Your code here
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Fill in the lines below: imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns


print(imputed_X_train.shape)
print(imputed_X_valid.shape)





from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


# Define the model
my_model_3 = XGBRegressor(n_estimators=1000, learning_rate=0.15)

# Fit the model
my_model_3.fit(imputed_X_train, y_train,
             early_stopping_rounds=5,
             eval_set=[(imputed_X_valid, y_valid)],
             verbose=False) # Your code here

# Get predictions
predictions_3 = my_model_3.predict(X_valid)

# Calculate MAE
mae_3 = mean_absolute_error(predictions_3,y_valid)

print(mae_3)








# Predict on the test data

from sklearn.impute import SimpleImputer

# Fill in the lines below: imputation
my_imputer = SimpleImputer(strategy='median') # Your code here
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_test))


# Fill in the lines below: imputation removed column names; put them back
imputed_X_train.columns = X_test.columns



# Get test predictions
preds_test = my_model_3.predict(imputed_X_train)


import numpy as np

preds_test=(np.around(preds_test,decimals=0))
preds_test=preds_test.astype(int)


# Save test predictions to file
output = pd.DataFrame({'ID': X_test.index,
                       'Crop_Damage': preds_test})
output.to_csv('submissionXX.csv', index=False)




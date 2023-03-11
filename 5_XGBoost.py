import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


MELBOURNE_FILE_PATH = 'input/melbourne-housing-snapshot/melb_data.csv'
data = pd.read_csv(MELBOURNE_FILE_PATH)
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]
y = data.Price

X_train, X_valid, y_train, y_valid = train_test_split(X, y)
my_model = XGBRegressor()
my_model.fit(X_train, y_train)
predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

# adding number of estimators
my_model = XGBRegressor(n_estimators=500, early_stopping_rounds=5)
my_model.fit(X_train, y_train,  
             eval_set=[(X_valid, y_valid)],
             verbose=False)
predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

# adding learning rate
my_model = XGBRegressor(n_estimators=500, early_stopping_rounds=5, learning_rate=0.05)
my_model.fit(X_train, y_train,  
             eval_set=[(X_valid, y_valid)],
             verbose=False)
predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

# adding learning rate
my_model = XGBRegressor(n_estimators=500, early_stopping_rounds=5, learning_rate=0.05, n_jobs=4)
my_model.fit(X_train, y_train,  
             eval_set=[(X_valid, y_valid)],
             verbose=False)
predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

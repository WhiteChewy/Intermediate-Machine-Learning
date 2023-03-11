import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

MELBOURNE_FILE_PATH = 'input/melbourne-housing-snapshot/melb_data.csv'


def score_dataset(X_train: pd.DataFrame, X_valid: pd.DataFrame, y_train: pd.DataFrame, y_valid: pd.DataFrame):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

def drop_columns(X_train, X_valid, y_train, y_valid):
    # Getting names of columns with missing values
    cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

    # Drop columns in training and validation data
    reduced_X_train = X_train.drop(cols_with_missing, axis=1)
    reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

    print("MAE from Approach 1 (Drop columns with missing values):")
    print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))


def imputation_score(X_train: pd.DataFrame, X_valid: pd.DataFrame, y_train: pd.DataFrame , y_valid: pd.DataFrame):
    my_imputer = SimpleImputer()
    imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
    imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

    # Imputation removed column names; put them back
    imputed_X_train.columns = X_train.columns
    imputed_X_valid.columns = X_valid.columns
    
    print("MAE from Approach 2 (Imputation):")
    print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))


def imputation_extended_score(X_train: pd.DataFrame, X_valid: pd.DataFrame, y_train: pd.DataFrame, y_valid: pd.DataFrame):
    # avoiding changes of original data
    X_train_plus = X_train.copy()
    X_valid_plus = X_valid.copy()
    # Getting names of columns with missing values
    cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
    # making new columns that indicates what will be imputed
    for col in cols_with_missing:
        X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
        X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()
    # imputation
    my_imputer = SimpleImputer()
    imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
    imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))
    # put back names
    imputed_X_train_plus.columns = X_train_plus.columns
    imputed_X_valid_plus.columns = X_valid_plus.columns
    print("MAE from Approach 3 (An Extension to Imputation):")
    print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))
    X_train.shape



if __name__ == '__main__':
    mlb_data = pd.read_csv(MELBOURNE_FILE_PATH)
    y = mlb_data.Price

    melb_predictors = mlb_data.drop(['Price'], axis=1)
    X = melb_predictors.select_dtypes(exclude=['object'])

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
    # drop_columns(X_train, X_valid, y_train, y_valid)
    # imputation_score(X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid)
    imputation_extended_score(X_train, X_valid, y_train, y_valid)

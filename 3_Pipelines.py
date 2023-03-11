import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

MELBOURNE_FILE_PATH = 'input/melbourne-housing-snapshot/melb_data.csv'

data = pd.read_csv(MELBOURNE_FILE_PATH)
y = data.Price
X = data.drop(['Price'], axis=1)

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Select categorical columns with relatively low cardinalty
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                                                               X_train_full[cname].dtype == 'object']

numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

print(X_train.head())

'''
STEP 1. Определить шаги для подготовки данных
'''
# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')
# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])
#Build preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols),
    ]
)
'''
STEP 2. Определяем модель
'''
model = RandomForestRegressor(n_estimators=100, random_state=0)
'''
STEP 3. Создаем и определяем пайплайн
Наконец именно на этом шаге мы используем класс Pipeline чтобы определить пайплайн который связывает шаги подготовки
данных и шаги моделирования.

Вот несколько важных замечаний:
- С пайплайном мы подготоавливаем данные для обучения и заполняем модель в одной строчке кода. (В обратном случае, без
пайплайна, нам нужно производить вменение, быстрое кодирование и обучение модели в разных разделенных шагах. Особенно
нечитаемо становится если нам необходимо работать И с численными и дискретными данными!)
- С пайплайном мы помещаем необработанные сущности X_valid в команду predict() и пайплайн автоматически подготавливает
эти сущности ДО того как выполнить предсказания. (Однако, без пайплайна, мы должны помнить что нужно подготовить оценочные 
данные перед выполнением предсказаний)
'''
# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model),
])
# Preprocessing of training data, fit model
my_pipeline.fit(X_train, y_train)
# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)
# Evluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE: ', score)
from itertools import compress as mask_arr
import numpy as np
import pandas as pd
from DataCorruption.DataCorruptor import DataCorruptor
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
import multiprocessing as mp
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer


from sklearn.feature_selection import RFE
from sklearn.svm import SVR
estimator = LogisticRegression


def get_pipeline(X, model=None):
    """Get a sklearn pipeline that is adjusted to the dataset X """
    numeric_features = X.select_dtypes(include="number").columns.to_list()
    categorical_features = X.select_dtypes(include="object").columns.to_list()

    if model is None:
        model = LogisticRegression()
    # TODO: Make this funtion parametrisable so it takes numeric/categorical transofmers as parameters
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    return Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)])


def measure_error_auc(clf, X_test, y_test, feature_cols, column_to_test):
    data_corruptor = DataCorruptor(X_test, feature_cols, log=False)
    total_cells = X_test.shape[0] * X_test.shape[1]
    res = []
    for n in range(X_test.shape[0]):
        #tmp = data_corruptor.get_dataset_with_corrupted_cell_in_column(column_to_test)
        corrupted_score = clf.score(X_test, y_test)
        res.append([(n / X_test.shape[0]), corrupted_score])
    df = pd.DataFrame(res, columns=['%Corrupted', 'Score'])

    # print('Area under the curve {}'.format(np.trapz(df['Score'],df['%Corrupted'])))
    return np.trapz(df['Score'], df['%Corrupted'])


def load_data():
    df = pd.read_csv('../Amit/data.csv')
    df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == "M" else 0)
    y = df['diagnosis']
    X = df.drop(['diagnosis', 'id'], axis=1)

    return X, y


def load_clean_airbnb_data():
    df = pd.read_csv('../Amit/Airbnb/clean_train.csv')
    df['Rating'] = df['Rating'].apply(lambda x: 1 if x == "Y" else 0)
    df = df.reset_index()
    y = df['Rating']
    X = df.drop(['Rating', 'index'], axis=1)

    return X, y


def load_dirty_airbnb_data():
    df = pd.read_csv('../Amit/Airbnb/dirty_test.csv')
    df['Rating'] = df['Rating'].apply(lambda x: 1 if x == "Y" else 0)
    df = df.reset_index()

    y = df['Rating']
    X = df.drop(['Rating', 'index'], axis=1)
    return X, y

X,y = load_clean_airbnb_data()

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.20, random_state=42)

#X_train, y_train = load_clean_airbnb_data()
#X_test, y_test = load_dirty_airbnb_data()
pipeline = get_pipeline(X_train)
airbnb_cols = X_train.columns
results = []

#### BASELINE CASES #####
pipeline.fit(X_train, y_train)
print("Baseline performance with all columns",pipeline.score(X_test, y_test))

for col in airbnb_cols:
    fitted_pipeline = pipeline.fit(X_train, y_train)
    result = measure_error_auc(fitted_pipeline, X_test, y_test, airbnb_cols, col)
    print('Trying out: {} : {}'.format(col,result))

    results.append([col, result])

print(pd.DataFrame(results, columns=['column', 'score']).sort_values(by='score', ascending=False))


top_5_auc_feature_ranking = []
top_10_auc_feature_ranking = []

top_10_rfe = []


# X,y = load_data()
#
# columns = X.columns.tolist()
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.20, random_state=42)
#
# results =[]
# pipeline = get_pipeline(X_train)
# for col in columns:
#     print('Trying out: '+ col)
#     fitted_pipeline = pipeline.fit(X_train, y_train)
#     results_df = measure_error_auc(fitted_pipeline, X_test, y_test, columns, col)
#     results.append([col,results_df])
#
#
#
# print(pd.DataFrame(results, columns=['column','score']).sort_values(by='score', ascending=False))
#
#
# print()
# print("")
# print()
#
#

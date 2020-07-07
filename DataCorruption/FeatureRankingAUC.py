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



def measure_error_auc(clf, X_test, y_test, feature_cols,column_to_test):
    data_corruptor = DataCorruptor(X_test, feature_cols,log=False)
    total_cells = X_test.shape[0] * X_test.shape[1]
    res = []
    for n in range(X_test.shape[0]):
        corrupted_score = clf.score(data_corruptor.get_dataset_with_corrupted_cell_in_column(column_to_test), y_test)
        res.append([(n / X_test.shape[0]), corrupted_score])
    df = pd.DataFrame(res, columns=['%Corrupted', 'Score'])

    # print('Area under the curve {}'.format(np.trapz(df['Score'],df['%Corrupted'])))
    return np.trapz(df['Score'], df['%Corrupted'])


def load_data():
    df = pd.read_csv('../amit/data.csv')
    df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == "M" else 0)
    y = df['diagnosis']
    X = df.drop(['diagnosis', 'id'], axis=1)

    return X, y

def load_airbnb_data():
    df = pd.read_csv('./amit/Airbnb/')

X,y = load_data()

columns = X.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.50, random_state=42)

results =[]
for col in columns:
    fitted_pipeline = get_pipeline(X_train).fit(X_train, y_train)
    results.append([col,measure_error_auc(fitted_pipeline,X_test,y_test,columns,col)])

print(pd.DataFrame(results, columns=['column','score']).sort_values(by='score', ascending=False))


print()
print("")
print()



from itertools import compress as mask_arr
import numpy as np
import pandas as pd
from DataCorruption.DataCorruptor import DataCorruptor
from sklearn.model_selection import KFold
import multiprocessing as mp
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
import traceback


class Error_Robustness_Scorer:

    def __init__(self, X, y, estimator=None):

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.01, random_state=42)
        self.feature_cols = X.columns

        self.pipeline = self.get_pipeline(self.X_train, estimator).fit(self.X_train, self.y_train)

    def get_pipeline(self, X, model=None):
        """Get a sklearn pipeline that is adjusted to the dataset X """
        numeric_features = X.select_dtypes(include="number").columns.to_list()
        categorical_features = X.select_dtypes(include="object").columns.to_list()

        if model is None:
            model = LogisticRegression(C=0.001)
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

    def measure_error_auc(self):
        try:
            data_corruptor = DataCorruptor(self.X_test, self.feature_cols, log=False)
            total_cells = self.X_test.shape[0] * self.X_test.shape[1]
            res = []
            for n in range(total_cells):
                corrupted_score = self.pipeline.score(data_corruptor.get_dataset_with_corrupted_cell(), self.y_test)
                res.append([(n / total_cells), corrupted_score])
            df = pd.DataFrame(res, columns=['%Corrupted', 'Score'])
        except Exception as e:
            print(e)
            print(traceback.format_exc())
        # print('Area under the curve {}'.format(np.trapz(df['Score'],df['%Corrupted'])))
        return np.trapz(df['Score'], df['%Corrupted'])


def load_clean_airbnb_data():
    print("Loaded clean AirBnB dataset")
    df = pd.read_csv('../Amit/Airbnb/clean_train.csv')
    df['Rating'] = df['Rating'].apply(lambda x: 1 if x == "Y" else 0)
    df = df.reset_index()
    y = df['Rating']
    X = df.drop(['Rating', 'index'], axis=1)
    # X = X[top_10]
    return X, y

X,y = load_clean_airbnb_data()

print(Error_Robustness_Scorer(X,y).measure_error_auc())
print(Error_Robustness_Scorer(X,y).measure_error_auc())
print(Error_Robustness_Scorer(X,y).measure_error_auc())
print(Error_Robustness_Scorer(X,y).measure_error_auc())
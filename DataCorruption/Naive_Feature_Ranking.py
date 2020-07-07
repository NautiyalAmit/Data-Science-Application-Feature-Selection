import numpy as np
import pandas as pd
from DataCorruption.DataCorruptor import DataCorruptor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer


class Naive_Error_Ranking:

    def __init__(self, X, y, pipeline):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.20, random_state=42)

        self.pipeline = pipeline
        self.feature_columns = X.columns

        self.pipeline.fit(self.X_train, self.y_train)
        #TODO: Take corruption params
        self.data_corruptor = DataCorruptor(self.X_test, X.columns)

        self._get_baseline_score()

    def _get_baseline_score(self):
        self.clean_test_baseline = self.pipeline.score(self.X_test, self.y_test)
        print("Baseline score for this model and pipeline: {}".format(self.clean_test_baseline))

    def __call__(self):
        print('Feature Ranking Error')
        res_ = []
        print()
        for idx, column in enumerate(self.feature_columns):
            corrupted_score = self.pipeline.score(self.data_corruptor.get_dataset_with_corrupted_col(column,'_introduce_outlier','_insert_empty_string'),
                                                  self.y_test)
            loss = corrupted_score - self.clean_test_baseline
            res_.append([column, corrupted_score, loss])
            print(column + " model score: %.6f" % corrupted_score)

        return res_


def get_pipeline(X, model=None):
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


def load_clean_airbnb_data():
    print("Loaded clean AirBnB dataset")
    df = pd.read_csv('../Amit/Airbnb/clean_train.csv')
    df['Rating'] = df['Rating'].apply(lambda x: 1 if x == "Y" else 0)
    df = df.reset_index()
    y = df['Rating']
    X = df.drop(['Rating', 'index'], axis=1)

    return X, y


X, y = load_clean_airbnb_data()
pipeline = get_pipeline(X)

NER = Naive_Error_Ranking(X, y, pipeline)

print(pd.DataFrame(NER(), columns=['ColumnName', 'CorruptedScore', "Loss"]).sort_values(by='Loss', ascending=False))

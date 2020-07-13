import numpy as np
import pandas as pd
from DataCorruption.DataCorruptor import DataCorruptor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


class Naive_Error_Ranking:

    def __init__(self, X, y, pipeline):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.20, random_state=42)

        self.pipeline = pipeline
        self.feature_columns = X.columns

        self.pipeline.fit(self.X_train, self.y_train)
        # TODO: Take corruption params
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
            corrupted_score = self.pipeline.score(
                self.data_corruptor.get_dataset_with_corrupted_col(column, '_insert_nan',
                                                                   '_insert_empty_string'),
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
        model = LogisticRegression(C=10)
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


top_10 = ['Bathrooms', 'Bedrooms', 'Beds', 'LocationName', 'NumGuests', 'NumReviews', 'Price', 'latitude', 'longitude',
          'zipcode']


def load_clean_airbnb_data():
    print("Loaded clean AirBnB dataset")
    df = pd.read_csv('../Amit/Airbnb/clean_train.csv')
    df['Rating'] = df['Rating'].apply(lambda x: 1 if x == "Y" else 0)
    df = df.reset_index()
    y = df['Rating']
    X = df.drop(['Rating', 'index'], axis=1)
    # X = X[top_10]
    return X, y


def load_dirty_airbnb_data():
    df = pd.read_csv('../Amit/Airbnb/duplicates/dirty_test.csv')
    df['Rating'] = df['Rating'].apply(lambda x: 1 if x == "Y" else 0)
    df = df.reset_index()

    y = df['Rating']
    X = df.drop(['Rating', 'index'], axis=1)
    # X = X[top_10]

    return X, y


def load_data():
    print("Loaded clean cancer dataset")

    df = pd.read_csv('../Amit/data.csv')
    df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == "M" else 0)
    y = df['diagnosis']
    X = df.drop(['diagnosis', 'id'], axis=1)

    # X = X[['texture_mean', 'area_mean', 'fractal_dimension_mean', 'symmetry_se', 'texture_worst', 'perimeter_worst',
    #       'smoothness_worst', 'compactness_worst', 'concavity_worst', 'fractal_dimension_worst']]

    return X, y


clean_X, clean_y = load_clean_airbnb_data()
pipeline = get_pipeline(clean_X)

NER = Naive_Error_Ranking(clean_X, clean_y, pipeline)

top_k_df = pd.DataFrame(NER(), columns=['ColumnName', 'CorruptedScore', "Loss"]).sort_values(by='Loss', ascending=False)
# top_k_df = pd.read_pickle('ranking_pickle')

dirty_X, dirty_y = load_dirty_airbnb_data()


def experiment_with_ranking(X, y, split=False, clean_train_X=None, clean_train_y=None):
    results_ = []

    if split:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42)
        for top_k in range(len(top_k_df)):
            top_k += 1
            top_k_columns = top_k_df.head(top_k)["ColumnName"].values

            tmp_X = X_train[top_k_columns]
            tmp_y = y_train
            pipeline = get_pipeline(tmp_X).fit(tmp_X, tmp_y)

            score = pipeline.score(X_test[top_k_columns], y_test)
            res = (len(top_k_columns), score)
            results_.append(res)

    else:
        for top_k in range(len(top_k_df)):
            top_k += 1
            top_k_columns = top_k_df.head(top_k)["ColumnName"].values

            tmp_X = clean_train_X[top_k_columns]
            tmp_y = clean_train_y
            pipeline = get_pipeline(tmp_X).fit(tmp_X, tmp_y)

            score = pipeline.score(X[top_k_columns], y)
            res = (len(top_k_columns), score)
            results_.append(res)

    return pd.DataFrame(results_, columns=['topk', 'score']).sort_values(by='score', ascending=False)


print("================================")
print("Airbnb Clean Training and test ")
print(experiment_with_ranking(clean_X, clean_y, split=True))
# one = experiment_with_ranking(clean_X, clean_y, split=True)
print("================================")
print("Airbnb Clean Training and dirty test ")
print(experiment_with_ranking(dirty_X, dirty_y, clean_train_X=clean_X, clean_train_y=clean_y))
# two = experiment_with_ranking(dirty_X, dirty_y, clean_train_X=clean_X, clean_train_y=clean_y)
print()

print("================================")
print(" Cancer Clean Training and test ")

X, y = load_data()
pipeline = get_pipeline(X)

NER = Naive_Error_Ranking(X, y, pipeline)
top_k_df = pd.DataFrame(NER(), columns=['ColumnName', 'CorruptedScore', "Loss"]).sort_values(by='Loss', ascending=False)

print(experiment_with_ranking(X, y, split=True))

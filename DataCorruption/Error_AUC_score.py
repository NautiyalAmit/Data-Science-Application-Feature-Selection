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
import time

class Error_Robustness_Scorer:

    def __init__(self, X, y, estimator=None):

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.10, random_state=42)
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
        pool = mp.Pool(mp.cpu_count() - 1)
        results = []

        def log_result(x):
            #print('Logging ', x)
            results.append(x)

        try:
            data_corruptor = DataCorruptor(self.X_test, self.feature_cols, log=False)
            total_cells = self.X_test.shape[0] * self.X_test.shape[1]
            res = []
            for n in range(total_cells):
                # corrupted_score = self.pipeline.score(data_corruptor.get_dataset_with_corrupted_cell(), self.y_test)
                d = data_corruptor.get_dataset_with_corrupted_cell().copy()
                t = self.y_test.copy()
                try:
                    pool.apply_async(self.measure, args=(self.pipeline,d, t,float(n/total_cells)),
                                     callback=log_result)
                except Exception as err:
                    print(err)
                # Close the pool for new tasks
            pool.close()

            # Wait for all tasks to complete at this point
            pool.join()
            # res.append([(n / total_cells), corrupted_score])
            df = pd.DataFrame(results, columns=['%Corrupted', 'Score'])


        except Exception as e:
            print(e)
            print(traceback.format_exc())
        # print('Area under the curve {}'.format(np.trapz(df['Score'],df['%Corrupted'])))
        return np.trapz(df['Score'], df['%Corrupted'])

    @staticmethod
    def measure(p, d, t,percentage):
        return percentage,p.score(d,t)


def load_clean_airbnb_data():
    print("Loaded clean AirBnB dataset")
    df = pd.read_csv('../Amit/Airbnb/clean_train.csv')
    df['Rating'] = df['Rating'].apply(lambda x: 1 if x == "Y" else 0)
    df = df.reset_index()
    y = df['Rating']
    X = df.drop(['Rating', 'index'], axis=1)
    # X = X[top_10]
    return X, y


top_k_df = pd.read_pickle('ranking_pickle')
X, y = load_clean_airbnb_data()
res = []
for top_k in range(len(top_k_df)):
    top_k += 1
    top_k_columns = top_k_df.head(top_k)["ColumnName"].values
    start_time = time.time()

    scr = [top_k, Error_Robustness_Scorer(X[top_k_columns], y).measure_error_auc(),(time.time() - start_time)]
    print(scr)
    res.append(scr)

print(pd.DataFrame(res, columns=['column', 'score','duration']).sort_values(by='score', ascending=False))

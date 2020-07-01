from itertools import compress as mask_arr
import numpy as np
import pandas as pd
from DataCorruption.DataCorruptor import DataCorruptor
from sklearn.model_selection import KFold
import multiprocessing as mp
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer


# mp.set_start_method('spawn', force=True)


# This function returns you a pippeline for specified features representation
def measure_error_auc(clf, X_test, y_test, feature_cols):
    data_corruptor = DataCorruptor(X_test, feature_cols,log=False)
    total_cells = X_test.shape[0] * X_test.shape[1]
    res = []
    for n in range(total_cells):
        corrupted_score = clf.score(data_corruptor.get_dataset_with_corrupted_cell(), y_test)
        res.append([(n / total_cells), corrupted_score])
    df = pd.DataFrame(res, columns=['%Corrupted', 'Score'])

    # print('Area under the curve {}'.format(np.trapz(df['Score'],df['%Corrupted'])))
    return np.trapz(df['Score'], df['%Corrupted'])


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


def split_fit_corrupt_measure(X,y, train_index, test_index, list_without_feature):
    X_train, _X_test = X[list_without_feature].iloc[train_index], X[list_without_feature].iloc[test_index]
    y_train, _y_test = y[train_index], y[test_index]
    #print(_X_test.shape,_y_test.shape)
    fitted_pipeline = get_pipeline(X_train).fit(X_train, y_train)

    # print([x[:5] for x in list_without_feature],)
    score= measure_error_auc(fitted_pipeline,
                             _X_test,
                             _y_test,
                             list_without_feature)
    #print(score)
    return score

def error_backward_selection(X, y):
    res_dict = {}
    kf = KFold(n_splits=10)
    initial_features = X.columns.tolist()
    selected_features = initial_features[:]  # Copy
    print("Having {} folds and {} CPU cores".format(10,mp.cpu_count() - 1))
    pool = mp.Pool(mp.cpu_count() - 1)

    # CV of the base feature representation
    base_res = []

    def log_result(x):
        print('Logging ', x)
        base_res.append(x)

    for train_index, test_index in kf.split(X):
        #split_fit_corrupt_measure(X,y, train_index, test_index, X.columns.tolist())
        pool.apply_async(split_fit_corrupt_measure, args=(X,y, train_index, test_index, X.columns.tolist()),
                         callback=log_result)

    pool.close()
    pool.join()
    print("Baseline average error AUC score after CV {}".format(np.array(base_res).mean()))

    top_score = np.array(base_res).mean()
    maximum_score_after_removal = -np.inf
    while top_score >= maximum_score_after_removal:
        tmp_results = []
        # feature list
        for feature in selected_features:
            mask = [0 if x is feature else 1 for x in selected_features]
            list_without_feature = list(mask_arr(selected_features, mask))
            pool = mp.Pool(mp.cpu_count() - 1)

            results = []
            def log_result(x):
                #print('Logging ',x)
                results.append(x)

            # Cross Validate
            for train_index, test_index in kf.split(X):
                pool.apply_async(split_fit_corrupt_measure, args=(X,y, train_index, test_index, list_without_feature),
                                 callback=log_result)

            # Close the pool for new tasks
            pool.close()

            # Wait for all tasks to complete at this point
            pool.join()
            print("{} removed".format(feature), np.array(results).mean())
            new_score = np.array(results).mean()

            res_dict[repr(list_without_feature)] = new_score
            tmp_results.append([feature, new_score])

        print(tmp_results)
        argmax =np.array(tmp_results)[:,1].astype(np.float).argmax()
        maximum_score_after_removal = tmp_results[argmax][1]

        if maximum_score_after_removal > top_score:
            top_score=maximum_score_after_removal
            feature_removed = tmp_results[argmax][0]
            selected_features.remove(feature_removed)
        else:

            print("Done.")
            print("Top {} out of {} features with the score {} are :".format(len(selected_features),
                                                                             len(initial_features), top_score))
            print(selected_features)
            break
        print("Currently best score is: {} \n Looking for better representation. ".format(top_score))



def load_data():
    df = pd.read_csv('../amit/data.csv')
    df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == "M" else 0)
    y = df['diagnosis']
    X = df.drop(['diagnosis', 'id'], axis=1)

    return X, y


X, y = load_data()

feature_cols = X.columns.to_list()

error_backward_selection(X, y)

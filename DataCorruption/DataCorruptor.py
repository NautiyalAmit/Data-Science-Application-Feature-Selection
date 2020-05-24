import pandas as pd
import numpy as np
from numpy.random import choice
import random
from random import randrange

np.random.seed(10)
RAND=random.randint(1, 1000)
#RANDOM=np.random.random()
    #RANDOM_NORMAL=np.random.normal(row[column_target], sigma, 1)


class NumericDataCorruptor:
    
    
    #print("########CLASS############",RAND)
    def __init__(self, data, feature_stats, feature_cols, log=False):
        self.data = data.copy()
        self.feature_stats = feature_stats
        self.feature_cols = feature_cols

        self.log = log
        self.probability_of_error = 0.95

        self.corrupted_cells_mask = pd.DataFrame(0, index=np.arange(data.shape[0]), columns=feature_cols)


    def _switch_column_values(self, row, column_target):
        """This function is swiching values between two columns """
        cols = row.index.tolist()
        column_target_index = cols.index(column_target)

        if column_target_index == 0:
            column_source_index = 1
        elif column_target_index == len(cols) - 1:
            column_source_index = column_target_index - 1
        else:
            column_source_index = column_target_index - 1

        column_source = cols[column_source_index]
        replace_value = row[column_source]
        if self.log:
            print(">>> Replacing values between {}: {} and {}: {}".format(column_target, row[column_target]
                                                                          , column_source, replace_value))
        row[column_source] = row[column_target]
        row[column_target] = replace_value
        return row

    def _insert_nan(self, row, column_target):
        if self.log:
            print(">>> Replacing value with NaN")
        row[column_target] = np.nan
        return row

    def _add_noise(self, row, column_target):
        mu, sigma = self.feature_stats.loc[column_target][['mean', 'std']].to_numpy()
        noise = np.random.normal(row[column_target], sigma, 1)
        if self.log:
            print('>>> Adding noise to {} => {}'.format(row[column_target], noise[0]))
        row[column_target] = noise[0]
        return row

    def _introduce_outlier(self, row, column_target):
        """
        It takes the row and a column name and returns the row with a value for that column exchanged by an outlier.
        Outlier is defined as value one standard deviation further from the min/max
        :param row:
        :param column_target:
        :return:
        """
        # print('introduce outliers')
        sigma, maximum, minimum = self.feature_stats.loc[column_target][['std', 'max', 'min']].to_numpy()
        if  np.random.random() < 0.5:
            #print("############INTRODUCE OUTLIER IF ########",RAND)

            outlier = maximum + (np.random.random() * sigma)
        else:
            
            #print("############INTRODUCE OUTLIER ELSE########",RAND)
            outlier = minimum - (np.random.random() * sigma)
        if self.log:
            print('>>> Adding outliers insted of {} => {}'.format(row[column_target], outlier))
        row[column_target] = outlier

        return row

    def _corrupt_value_by_column(self, row, col_name):
        #print("########CORRUPT VAL BY COLn############",RAND)
        if np.random.random() < self.probability_of_error:
            #TODO: here add functions to corrupt categroical data
            draw = choice([self._switch_column_values,
                           self._add_noise, self._insert_nan,
                           self._introduce_outlier], 1,
                          p=[0.25, 0.25, 0.25, .25])[0]
            return draw(row, col_name)
        else:
            return row

    def get_dataset_with_corrupted_col(self, col_name, error_proba=0.95):
        """
        This function return DataFrame with a given collumn corrupted to a degree controlled by error_proba value.
        :param col_name:
        :param error_proba:
        :return:
        """
        if col_name not in self.feature_cols:
            raise ValueError("Column name is not present in the data")
        self.probability_of_error = error_proba

        print('Corrutping %.2f percent of : %s' % (error_proba, col_name))
        self.data = self.data.apply(self._corrupt_value_by_column, axis=1, args=(col_name,))  # Iterate over all cols
        return self.data

    def get_random_indices(self):
        """This function returns a pair for integer indices that are not yet corrupted."""
        col_ix, row_ix = randrange(self.data.shape[1]), randrange(self.data.shape[0])
        # While the selected cell is already corrupted, keep "rolling the dice" until we find one that is not corrupted.
        while self.corrupted_cells_mask.iat[row_ix,col_ix] == 1:
            col_ix, row_ix = randrange(self.data.shape[1]), randrange(self.data.shape[0])
        self.corrupted_cells_mask.iat[row_ix, col_ix] = 1
        return col_ix, row_ix

    def get_dataset_with_corrupted_cell(self):
        col_idx, row_idx = self.get_random_indices()

        row_with_corrupted_cell = self._corrupt_value_by_column(self.data.iloc[row_idx],
                                                                self.feature_cols[col_idx])
        self.data.iloc[row_idx] = row_with_corrupted_cell

        return self.data


df = pd.DataFrame([[30, 20, 0.1], [10, 50, 0.5], [15, 30, 0.2]],
                  columns=['A', 'B', 'C'])

feature_stats = df.describe().T[['mean', 'std', 'max', 'min']]

data_corruptor = NumericDataCorruptor(df, feature_stats, df.columns.tolist())

print(data_corruptor.get_dataset_with_corrupted_cell())
print(data_corruptor.get_dataset_with_corrupted_cell())
print(data_corruptor.get_dataset_with_corrupted_cell())
print(data_corruptor.corrupted_cells_mask)

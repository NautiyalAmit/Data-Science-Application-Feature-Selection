import warnings
import pandas as pd
import numpy as np
from numpy.random import choice
import nlpaug.augmenter.char as nac

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

warnings.filterwarnings('ignore')

swapRandom = nac.RandomCharAug(action="swap")
replaceTwoCharsBasedOnKeyboard = nac.KeyboardAug()
deleteRandomChar = nac.RandomCharAug(action="delete")

np.random.seed(0)  # Setting seed globally


# r = np.random.RandomState(0) TODO: Setting the seed for the class locally without impacting global numpy seed


class DataCorruptor:

    def __init__(self, data, feature_cols, feature_stats=None, log=False):
        # np.random.seed(0)

        if feature_stats is None:
            # TODO: Take the cardinlal statistics (like most common value), into account while corupting data
            self.feature_stats = data.describe().T[['mean', 'std', 'max', 'min']]
        else:
            self.feature_stats = feature_stats

        self.data = data.copy()

        self.feature_cols = feature_cols

        self.log = log
        self.probability_of_error = 0.95

        self.corrupted_cells_mask = pd.DataFrame(0, index=np.arange(data.shape[0]), columns=feature_cols)

    def _insert_empty_string(self, row, column_target):
        if self.log:
            print(">>> Replacing value with empty string : {}".format(row[column_target]))
        row[column_target] = ""
        return row

    def _delete_random_char(self, row, column_target):
        if self.log:
            print(">>> Deleting random char: {}".format(row[column_target]))
        row[column_target] = deleteRandomChar.augment(row[column_target])
        return row

    def _replace_char_close_on_keyboard(self, row, column_target):
        if self.log:
            print(">>> Replacing char with sth close on keyboard {}".format(row[column_target]))
        row[column_target] = replaceTwoCharsBasedOnKeyboard.augment(row[column_target])
        return row

    def _swap_random_char(self, row, column_target):
        if self.log:
            print(">>> Swaping random char: {}".format(row[column_target]))
        row[column_target] = swapRandom.augment(row[column_target])
        return row

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
        if np.random.random() < 0.5:
            # print("############INTRODUCE OUTLIER IF ########",RAND)

            outlier = maximum + (np.random.random() * sigma)
        else:

            # print("############INTRODUCE OUTLIER ELSE########",RAND)
            outlier = minimum - (np.random.random() * sigma)
        if self.log:
            print('>>> Adding outliers insted of {} => {}'.format(row[column_target], outlier))
        row[column_target] = outlier

        return row

    def _corrupt_value_by_column(self, row, col_name):

        if is_numeric_dtype(self.data[col_name]):
            draw = choice([#self._switch_column_values,
                          # self._add_noise,
                        self._insert_nan,
                        #self._introduce_outlier
            ], 1,
                          p=[ 1])[0]
            return draw(row, col_name)

        elif is_string_dtype(self.data[col_name]):
            draw = choice([self._insert_empty_string,
                           #self._delete_random_char,
                           self._replace_char_close_on_keyboard
                           #self._swap_random_char
             ], 1, p=[.5,.5])[0]

            return draw(row, col_name)

    def get_dataset_with_corrupted_col(self, col_name, error_proba=0.15):
        """
        This function return DataFrame with a given collumn corrupted to a degree controlled by error_proba value.
        :param col_name:
        :param error_proba:
        :return:
        """
        if col_name not in self.feature_cols:
            raise ValueError("Column name is not present in the data")
        self.probability_of_error = error_proba
        if self.log:
            print('Corrutping %.2f percent of : %s' % (error_proba, col_name))
        self.data = self.data.apply(self._corrupt_value_by_column, axis=1, args=(col_name,))  # Iterate over all cols
        return self.data

    def get_random_indices(self):
        """This function returns a pair for integer indices that are not yet corrupted."""
        #col_ix, row_ix = np.random.randint(self.data.shape[1]), np.random.randint(self.data.shape[0])
        # While the selected cell is already corrupted, keep "rolling the dice" until we find one that is not corrupted.
        #while self.corrupted_cells_mask.iat[row_ix, col_ix] == 1:
        #    col_ix, row_ix = np.random.randint(self.data.shape[1]), np.random.randint(self.data.shape[0])
        #return col_ix, row_ix

        non_corrupted_row_col_pairs =[]
        for col_idx in range(self.data.shape[1]):
           non_corrupted_cell_row_idx_list = self.data.iloc[:,col_idx].index[self.corrupted_cells_mask.iloc[:,col_idx] == 0].tolist()
           non_corrupted_row_col_pairs_for_col_idx = list(map(lambda row_index: (self.data.index.get_loc(row_index), col_idx), non_corrupted_cell_row_idx_list))
           non_corrupted_row_col_pairs.extend(non_corrupted_row_col_pairs_for_col_idx)

        rand_idx = np.random.randint(len(non_corrupted_row_col_pairs), size=1)[0]
        return non_corrupted_row_col_pairs[rand_idx]


    def get_dataset_with_corrupted_cell(self):
        row_idx,col_idx = self.get_random_indices()

        row_with_corrupted_cell = self._corrupt_value_by_column(self.data.iloc[row_idx],
                                                                self.feature_cols[col_idx])
        self.data.iloc[row_idx] = row_with_corrupted_cell
        self.corrupted_cells_mask.iat[self.data.index.get_loc(row_idx), col_idx] = 1

        return self.data


    def get_random_row_index(self,col_idx):

        non_corrupted_cell_row_idx_list = self.data.iloc[:,col_idx].index[self.corrupted_cells_mask.iloc[:,col_idx] == 0].tolist()
        row_idx = np.random.choice(non_corrupted_cell_row_idx_list, 1)[0]
        return row_idx

    def get_dataset_with_corrupted_cell_in_column(self,col_name):
        if col_name not in self.feature_cols:
            raise ValueError("Column name is not present in the data")
        col_idx = self.feature_cols.index(col_name)
        row_idx = self.get_random_row_index(col_idx)
        row_with_corrupted_cell = self._corrupt_value_by_column(self.data.loc[row_idx],
                                                                self.feature_cols[col_idx])
        self.data.loc[row_idx] = row_with_corrupted_cell

        self.corrupted_cells_mask.iat[self.data.index.get_loc(row_idx), col_idx] = 1
        return self.data

#print('Quick smoke test')    
#df = pd.DataFrame([[30, 20, 0.1, 'lmao'], [10, 50, 0.5, 'omfg'], [15, 30, 0.2, 'wtfp']], columns=['A', 'B', 'C', "D"])

#data_corruptor = DataCorruptor(df, df.columns.tolist())

# print(data_corruptor.get_dataset_with_corrupted_cell())
# print(data_corruptor.get_dataset_with_corrupted_cell())

#data_corruptor.get_dataset_with_corrupted_cell()
#data_corruptor.get_dataset_with_corrupted_cell()
#print(data_corruptor.get_dataset_with_corrupted_cell())
#print(data_corruptor.corrupted_cells_mask)

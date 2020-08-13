import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from pandas.api.types import is_numeric_dtype, is_string_dtype
from scipy.stats import chi2_contingency
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn import preprocessing

# We change a setting just to avoid an annoying warning
pd.set_option('display.max_rows', 100)

# We are goint to define 5 functions
# Number 1
def value_counts_csv(df):
    
    """
    This function returns a .csv for every column of a pandas DataFrame. The goal is to show the frequency and the relative frequency of every non-null value of the column.
    """
    
    assert isinstance(df, pd.DataFrame)
    
    if not os.path.exists('value_counts_csv'):
        os.makedirs('value_counts_csv')
    
    for col in df.columns:
        value_counts = pd.DataFrame(df[col].value_counts(dropna = False, ascending = False))
        value_counts['Percentage'] = value_counts.iloc[:,0]/sum(value_counts.iloc[:,0])
        
        value_counts.reset_index(drop = False, inplace = True)
        value_counts.rename(columns = {'index': col, col: 'Count'}, inplace = True)
        
        value_counts.to_csv('./value_counts_csv/'+col+'.csv', index = False)

# Number 2        
def count_nulls(df):
    
    """
    This function return a .csv for a given pandas DataFrame. The goal is to show the frequency and the relative frequency of null values for every column.
    """
    
    assert isinstance(df, pd.DataFrame)
    
    if not os.path.exists('./value_counts_csv/Nulls_folder'):
        os.makedirs('./value_counts_csv/Nulls_folder')
    
    nulls = pd.DataFrame(df.isna().sum())
    nulls.reset_index(drop = False, inplace = True)
    nulls.rename(columns = {'index': 'Column_Name', nulls.columns[1]:'Count'}, inplace = True)
    
    nulls['Percentage'] = nulls['Count']/df.shape[0]
        
    nulls.to_csv('./value_counts_csv/Nulls_folder/Nulls.csv', index = False)
    
# Number 3
def hist_boxplot(df):
    
    """
    This function return a .png file for every column of a given pandas DataFrame. The goal is to give a look at the distributions of the features. In case of a categorical feature, only a histogram is provided. If the feature is numerical, also a boxplot is shown.
    """
    
    assert isinstance(df, pd.DataFrame)
    
    if not os.path.exists('plots/distribution'):
        os.makedirs('plots/distribution')
        
    for col in df.columns:
        if df[col].dtype == np.object:
            plt.figure(col)
            plt.hist(df[col].dropna())
            plt.savefig('plots/distribution/' + col)
            plt.close(col)
        else:
            fig, ax = plt.subplots(1,2, figsize=(20,10))
            plt.sca(ax[0])
            plt.hist(df[col].dropna())
                
            plt.sca(ax[1])
            df.boxplot(column=col)
                
            fig.savefig('plots/distribution/' + col)
            plt.close(fig)
            
# Number 4
def regressor_gridCV(X_train, y_train, reg, 
                      X_test = None, y_test = None, cv = 5,
                      scoring = 'neg_mean_squared_error', params = {}, model_name = "model",
                     random_state=123):
    
    """
    The goal of this function is to train and validate a binary classifier using the Cross-Validation method together with GridSearch. It returns a .csv with the results of the Cross-Validation scores. Moreover, the function test the performance of the model on a test dataset (if this is provided) and it prints the result.
    """
    
    grid = GridSearchCV(reg, params, cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state), scoring = scoring, refit = True, n_jobs=-1)
    
    model = grid.fit(X_train, y_train)
    print("The best parameters of grid are: ", model.best_params_, 
          "\nThe best estimator is: ", model.best_estimator_)
    
    if not os.path.exists('models/cv_results'):
        os.makedirs('models/cv_results')
    
    cvres = model.cv_results_
    
    if params != {}:    
        dataframe = pd.DataFrame(cvres["params"])
        dataframe.insert(0, "mean_test_score", cvres["mean_test_score"])
                        
    else: 
        dataframe = pd.DataFrame({"mean_test_score":cvres["mean_test_score"]})
            
    dataframe.to_csv("./models/cv_results/" + model_name + ".csv", index = False) 
    
    if X_test is not None:
        results = model.predict(X_test)
    
        if y_test is not None:
            print("The results on the test are: ")
            print("MSE = {0:.10f}".format(mean_squared_error(y_test, results)))
            print("RMSE = {0:.10f}".format(mean_squared_error(y_test, results)**0.5))
            print("MAE = {0:.10f}".format(mean_absolute_error(y_test, results)))
            print("R2 = {0:.10f}".format(r2_score(y_test, results)))
            
    return(model)

# Number 5
def columns_list_except(df, avoid_columns):
    """
    The goal of this function is to return a list with the names of the columns of a DataFrame except the ones that you want to avoid explicitely
    """
    col_names = [col for col in df.columns if col not in list(avoid_columns)]
    return col_names

# Number 6
def joining_fields(df, left_column, right_column, name_new_field):
    """
    The function take two columns in input and it joins their values through the character '_'
    """
    df[name_new_field] = df[left_column].astype(str) + "_" + df[right_column].astype(str)
    return df

# Number 7
def comparing_columns(df_left, df_right, col_left, col_right):
    """
    The function counts how many values of the pandas series df_right[col_right] you can find in the pandas series df_left[col_left]
    """
    values = set(df_left[col_left])
    df_right['Match'] = df_right[col_right].isin(values)
    print(df_right['Match'].value_counts())
    return df_right['Match']
    
# Number 8
def plotting_random_grid(sample_size, df_sampled, df_filtered, key_col, col_x, col_y, x_ticks):
    """
    Firstly, the function makes a sample of the DataFrame df_sampled according to key_col. 
    After that, it uses every value of the sample to filter df_filtered and to plot a xy-chart of col_x and col_y, where col_x and col_y belongs to df_filtered. 
    The number of points represented on the x-axis is given by x_ticks
    Sample size should be a multiple of 3 in order to obtatin a suitable visualization
    """
    if df_sampled.name == df_filtered.name:
        last_directory_part = df_sampled.name
    else:
        last_directory_part = df_sampled.name + "-" + df_filtered.name
    current_directory = 'plots/arima/' + last_directory_part
    if not os.path.exists(current_directory):
        os.makedirs(current_directory)
        
    sample_key_val = df_sampled[key_col].sample(n=sample_size, random_state=234)
    fig, axs = plt.subplots((sample_size // 3), 3, figsize=(20, 80))
    
    i = 0
    for sample_val in range(0, sample_size):
        j = sample_val % 3
        key_val = sample_key_val.iloc[sample_val]
        df_example = df_filtered[df_filtered[key_col] == key_val]
        axs[i,j].plot(df_example[col_x], df_example[col_y])
        axs[i,j].set_xticks(range(0, x_ticks))
        fig.savefig(current_directory + "/" + key_col + "-" + col_x + "-" + col_y)
        plt.close(fig)
        if j == 2:
            i = i+1

# Number 9
def building_unit_lags(df, df_lag, pk, time_col, value_col, lag_size):
    """
    This function creates a lag column for a dataset df_lag in input. The size of the lag step is given by lag_size
    """
    for i in range(1, lag_size+1):
        df_to_merge = df.copy()
        df_to_merge[time_col] = df_to_merge[time_col] + i
        df_to_merge[value_col + '_lag_' + str(i)] = df_to_merge[value_col]
        df_lag = df_lag.merge(df_to_merge[[value_col + '_lag_' + str(i), time_col, pk]],
                                     left_on=[time_col, pk],
                                     right_on=[time_col, pk],
                                     how='left', suffixes=(None, None))
    return df_lag

# Number 10
def creating_submission_lag(df_submission, df_xgb_sel, month):
    """
    This function creates the lags columns on the dataframe we need to submit, so that we can apply our model and create a predictions column
    """
    df_submission_lag = df_xgb_sel.copy()
    df_submission_lag = df_submission_lag[df_submission_lag['date_block_num'] == month]
    df_submission_lag = df_submission_lag[['joined_fields', 'item_cnt_day']]
   
    df_submission = df_submission.merge(df_submission_lag, how='left',
                                        suffixes=('_left', 'right'), 
                                        left_on='joined_fields', right_on='joined_fields')
    return df_submission

# Number 11
def creating_submission_file(df_test, df_xgb_sel, month, rename_old, rename_new):
    """
    This function apply the creating_submission_lag function to a dataset and renames the columns according to the input
    """
    df_submission = creating_submission_lag(df_submission=df_test,
                                        df_xgb_sel=df_xgb_sel, month=month)
        
    df_submission = df_submission.rename(columns={rename_old: rename_new})
    return df_submission

# Number 12
def numeric_encoding(df, col):
    """
    Given a categorical column of a dataframe, the function performs a LabelEncoder transformation on it
    """
    le = preprocessing.LabelEncoder()
    le.fit(df[col])
    col_numeric = le.transform(df[col])
    df[col] = col_numeric
    return df

# Number 13
def modifying_submission_format(df_submission, predictions):
    """
    This function needs as input the result of the creating_submission_file function. Thererefore, it transforms the dataframe in the format
    requested by Kaggle
    """
    df_submission['item_cnt_month'] = predictions
    df_submission_final = df_submission.copy()
    df_submission_final['ID'] = df_submission_final.index
    df_submission_final = df_submission_final[['ID', 'item_cnt_month']]
    return [df_submission, df_submission_final]

# Number 14
def parsing_trials_hp(trials, field):
    """
    By this function we parse the trials file given by hyperopt to obtain a visualization of the performance of the model with the tested hyperparameters
    """
    field_value = [t['misc']['vals'][field] for t in trials]
    loss = [t['result']['loss'] for t in trials]
    plt.xlabel(field)
    plt.ylabel('Loss')
    plt.title('Loss by ' + field)
    plt.scatter(field_value, loss)
    
# Number 15    
def current_time():
    """
    Function to print the current_time
    """
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    
# Number 16
def extracting_trials_hp(param_list, trials, loss=None):
    """
    Given the results of a hyperopt search (the trials.trials object), we extract the most relevant parameters. We save them to a dictionary.
    """
    param_dict = {}
    for param in param_list:
        param_list = [t['misc']['vals'][param] for t in trials]
        param_dict[param] = param_list
    if loss:
        param_list = [t['result']['loss'] * 100 for t in trials]
        param_dict['loss'] = param_list
    return param_dict

# Number 17
def extracting_parameters_xgb(xgb_model, model_name, fields, sep=';'):
    """
    Given an xgb model, we extract the desired parameters (listed in the fields input) and we save them on a csv file
    """
    params = xgb_model.get_params()
    dict_dataframe = {}
    for num, field in enumerate(fields):
        dict_dataframe[field] = params[field]
    df_params = pd.DataFrame(dict_dataframe, index=[0])
    df_params.to_csv('models/hyperparameters/' + model_name + "_params", index=False, sep=sep)
    

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from pandas.api.types import is_numeric_dtype, is_string_dtype
from scipy.stats import chi2_contingency
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.model_selection import StratifiedKFold







pd.set_option('display.max_rows', 100)


def value_counts_csv(df):
    assert isinstance(df, pd.DataFrame)
    
    if not os.path.exists('value_counts_csv'):
        os.makedirs('value_counts_csv')
    
    for col in df.columns:
        value_counts = pd.DataFrame(df[col].value_counts(dropna = False, ascending = False))
        value_counts['Percentage'] = value_counts.iloc[:,0]/sum(value_counts.iloc[:,0])
        
        value_counts.reset_index(drop = False, inplace = True)
        value_counts.rename(columns = {'index': col, col: 'Count'}, inplace = True)
        
        value_counts.to_csv('./value_counts_csv/'+col+'.csv', index = False)

        
def count_nulls(df):
    assert isinstance(df, pd.DataFrame)
    
    if not os.path.exists('./value_counts_csv/Nulls_folder'):
        os.makedirs('./value_counts_csv/Nulls_folder')
    
    nulls = pd.DataFrame(df.isna().sum())
    nulls.reset_index(drop = False, inplace = True)
    nulls.rename(columns = {'index': 'Column_Name', nulls.columns[1]:'Count'}, inplace = True)
    
    nulls['Percentage'] = nulls['Count']/df.shape[0]
        
    nulls.to_csv('./value_counts_csv/Nulls_folder/Nulls.csv', index = False)
    

def hist_boxplot(df):
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
            

def classifier_gridCV(X_train, y_train, clf, 
                      X_test = None, y_test = None, cv = 5,
                      scoring = 'accuracy', params = {}, model_name = "model",
                     random_state=123):
    
    grid = GridSearchCV(clf, params, cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state), scoring = scoring, refit = True, n_jobs=-1)
    
    model = grid.fit(X_train, y_train)
    print("The best parameters of grid are: ", model.best_params_, 
          "\nThe best estimator is: ", model.best_estimator_)
    
    if not os.path.exists('Models/CV_results'):
        os.makedirs('Models/CV_results')
    
    cvres = model.cv_results_
    
    if params != {}:    
        dataframe = pd.DataFrame(cvres["params"])
        dataframe.insert(0, "mean_test_score", cvres["mean_test_score"])
                        
    else: 
        dataframe = pd.DataFrame({"mean_test_score":cvres["mean_test_score"]})
            
    dataframe.to_csv("./Models/CV_results/CV_results_"+model_name+".csv", index = False) 
    
    if X_test is not None:
        results = model.predict(X_test)
    
        if y_test is not None:
            print("The results on the test are: ")
            print("Precision = {}".format(precision_score(y_test, results, average='macro')))
            print("Recall = {}".format(recall_score(y_test, results, average='macro')))
            print("Accuracy = {}".format(accuracy_score(y_test, results)))
            
    return(model)

# numerical vs numerical: Pearson and Spearman [-1,1]
# categorical vs numerical: eta correlation (anova test) [0,1]
# categorical vs categorical: cramer's v (chi-squared test) [0,1]
# check if our metrics are symmetric: correlation and cramer yes

def correlation_matrix(df, method_numeric='pearson'):
    
    """
    Comment here
    
    """
    l = df.shape[1]
    matrix = np.zeros(shape=(l,l))
    matrix_mask = np.zeros(shape=(l,l))

    for j in range(l):
        col_1 = df.iloc[:,j]
        name_col_1 = df.columns[j]
        
        for i in range(j,l):
            col_2 = df.iloc[:,i]
            name_col_2 = df.columns[i]
            
            if is_numeric_dtype(col_1) and is_numeric_dtype(col_2):
                correlation = col_1.corr(col_2)
                matrix[j,i] = matrix[i,j] = correlation
                matrix_mask[j,i] = matrix_mask[i,j] = 1
                
            elif is_string_dtype(col_1) and is_string_dtype(col_2):
                frequency_table = pd.crosstab(col_1, col_2)
                
                chi2 = chi2_contingency(frequency_table)[0]
                
                total_frequencies = df.shape[0]
                phi2 = chi2/total_frequencies
                r, k = frequency_table.shape
                cramer = np.sqrt(phi2/min((r-1),(k-1)))
                matrix[i,j] = matrix[j,i] = cramer
                matrix_mask[j,i] = matrix_mask[i,j] = 2
             
            elif is_string_dtype(col_1) and is_numeric_dtype(col_2):
                anova_model = ols(name_col_2 + str(' ~ C(') + name_col_1 + str(')'), df).fit()
                anova_results = anova_lm(anova_model)
                eta2 = anova_results['sum_sq'][0]/(anova_results['sum_sq'][0]+anova_results['sum_sq'][1])
                matrix[i,j] = matrix[j,i] = eta2
                matrix_mask[j,i] = matrix_mask[i,j] = 3
            
            elif is_numeric_dtype(col_1) and is_string_dtype(col_2):
                anova_model = ols(name_col_1 + str(' ~ C(') + name_col_2 + str(')'), df).fit()
                anova_results = anova_lm(anova_model)
                eta2 = anova_results['sum_sq'][0]/(anova_results['sum_sq'][0]+anova_results['sum_sq'][1])
                matrix[i,j] = matrix[j,i] = eta2
                matrix_mask[j,i] = matrix_mask[i,j] = 3
            
            else: 
                matrix[i,j] = matrix[j,i] = np.nan
                
                
    return((matrix, matrix_mask))

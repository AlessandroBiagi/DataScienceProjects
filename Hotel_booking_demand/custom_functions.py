import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV

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
                      X_test = None, y_test = None, cv = 3, scoring = 'accuracy', params = {}, model_name = "model"):
    
    grid = GridSearchCV(clf, params, cv = cv, scoring = scoring, refit = True)
    
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
            print("Precision = {}".format(precision_score(y_test, results, average='macro')))
            print("Recall = {}".format(recall_score(y_test, results, average='macro')))
            print("Accuracy = {}".format(accuracy_score(y_test, results)))
            
    return(model)

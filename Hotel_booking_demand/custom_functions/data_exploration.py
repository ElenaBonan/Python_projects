import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os 
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, accuracy_score , roc_auc_score
from sklearn.model_selection import GridSearchCV
from pandas.api.types import is_numeric_dtype, is_string_dtype
from scipy.stats import chi2_contingency
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from sklearn.model_selection import StratifiedKFold

def value_counts_csv(df):
    """ 
    Given a dataframe it returns a csv for every columns where for every different values there is the number 
    of observations. The results are saved in a folder value_counts
    """
    assert isinstance(df, pd.DataFrame)
    if not os.path.exists('value_counts_csv'):
        os.makedirs('value_counts_csv')
    path = "./value_counts_csv"
    for col in df.columns:
        df_new = pd.DataFrame(df[col].value_counts(dropna = False, ascending = False))
        df_new['Percentage'] =  df_new.iloc[:,0]/sum( df_new.iloc[:,0])
        df_new.reset_index(inplace = True)
        df_new.rename(columns={ df_new.columns[1]: "Count" }, inplace = True)
        df_new.rename(columns={ df_new.columns[0]: col }, inplace = True)
        df_new.to_csv(path + "/" +col+ ".csv", index = False)
        
        

def count_nulls(df):
    """ Function to get the null values of the dataframe. It saved the results in the folder nulls inside value_counts"""
    assert isinstance(df, pd.DataFrame)
    if not os.path.exists('value_counts_csv/nulls'):
        os.makedirs('value_counts_csv/nulls')
    df_new = pd.DataFrame(df.isna().sum())
    df_new["Percentage"]=  df_new.iloc[:,0]/ df.shape[0]
    df_new.reset_index(inplace = True)
    df_new.rename(columns={ df_new.columns[0]: "Column" }, inplace = True)
    df_new.rename(columns={ df_new.columns[1]: "Count" }, inplace = True)
    df_new.to_csv('./value_counts_csv/nulls/nulls.csv', index = False)
    
def distributions(df):
    """ Function to get the distribution of the columns"""
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
            f, ax = plt.subplots(1,2, figsize=(20,10))
            plt.sca(ax[0])
            plt.hist(df[col].dropna())
            plt.sca(ax[1])
            df.boxplot(column=col)
            f.savefig('plots/distribution/' + col)
            plt.close(f)
            
def cat_dummies(df):
    object_columns = df.columns[df.dtypes == object].tolist()
    df =  pd.get_dummies(df, prefix = object_columns, columns = object_columns)
    return df
            
def classifier_gridCV(X_train, y_train, X_test = None,  y_test = None, cv = 5, scoring = "accuracy", params = {}, clf = xgb.XGBClassifier(), model_name = "model", random_state = 123):
    grid = GridSearchCV(clf,params,scoring = scoring, cv = StratifiedKFold(n_splits=cv, random_state = random_state, shuffle = True), n_jobs = -1)
    model = grid.fit(X_train, y_train)
    print("The best parameters from grid are:" , model.best_params_ )
    print("The parameters of the best model are: ", model.best_estimator_)
    path = "models/grid_CV"
    if not os.path.exists(path):
        os.makedirs(path)
    cvres = model.cv_results_
    if params != {}:
        dataframe = pd.DataFrame(cvres["params"])
        dataframe.insert(0, "mean_test_score", cvres["mean_test_score"])
    else :
        dataframe = pd.DataFrame( {"mean_test_score" : cvres["mean_test_score"] } )
    dataframe.to_csv( path+ "/" + model_name+ "_cv" ".csv", index = False)
    if X_test is not None:
        result = model.predict(X_test)
        if  y_test is not None:
            print("The results on the test are:")
            print("Precision = {}".format(precision_score(y_test, result, average='macro')))
            print("Recall = {}".format(recall_score(y_test, result, average='macro')))
            print("Accuracy = {}".format(accuracy_score(y_test, result)))
    return(model)

def correlation_matrix(df, method_numeric="pearson"):
    """ 
    This function return a matrix with the correlation between the variables. We used       the following methods:
    numeric vs numeric we use the method_numeric in input that can be pearson or             spearman
    categorical vs categoriacal we use the Cramer V
    numerical vs categorical the eta squared 
    """
    k = len(df.columns)
    matrix = np.zeros(shape=(k,k))
    matrix_mask= np.zeros(shape=(k,k))
    for j in range(k):
        col1 =  df.iloc[:,j]
        for i in range(j,k):
            col2 =  df.iloc[:,i]
            if  is_numeric_dtype(col1) and is_numeric_dtype(col2):
                correlation = col1.corr(col2)
                matrix[j,i]= matrix[i,j]= correlation
                matrix_mask[j,i] = matrix_mask[i,j] = 1
            elif is_string_dtype(col1) and is_string_dtype(col2):
                freq_table = pd.crosstab(col1,col2)
                chi2 = chi2_contingency(freq_table)[0]
                observations = df.shape[0]
                phi2 = chi2 / observations
                a,b = freq_table.shape
                cram = np.sqrt( phi2/ min((a-1),(b-1)))
                matrix[j,i]= matrix[i,j]= cram 
                matrix_mask[j,i] = matrix_mask[i,j] = 2
            elif (is_string_dtype(col1) and is_numeric_dtype(col2)):
                arg = df.columns[i] + ' ~ C(' +  df.columns[j]+ ')'
                model = ols( arg, df).fit()
                anovaResults = anova_lm(model)
                eta2 = anovaResults['sum_sq'][0]/(anovaResults['sum_sq'][0]+anovaResults['sum_sq'][1])
                matrix[j,i]= matrix[i,j]= eta2
                matrix_mask[j,i] = matrix_mask[i,j] = 3
            elif (is_numeric_dtype(col1) and is_string_dtype(col2)):
                arg = df.columns[j] + ' ~ C(' +  df.columns[i]+  ')' 
                model = ols( arg, df).fit()
                anovaResults = anova_lm(model)
                eta2 = anovaResults['sum_sq'][0]/(anovaResults['sum_sq'][0]+anovaResults['sum_sq'][1])
                matrix[j,i]= matrix[i,j]= eta2
                matrix_mask[j,i] = matrix_mask[i,j] = 3
            else:
                matrix[j,i]= matrix[i,j]= np.nan
    return((matrix,matrix_mask))
            
            
     
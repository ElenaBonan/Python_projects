import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os 
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, accuracy_score , roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from pandas.api.types import is_numeric_dtype, is_string_dtype
from scipy.stats import chi2_contingency
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from sklearn.model_selection import StratifiedKFold

def classifier_gridCV(X_train, y_train, X_test = None,  y_test = None, cv = 5, scoring = "accuracy", params = {}, clf = xgb.XGBClassifier(), model_name = "model", random_state = 123):
    """
    This function it used for a classification problem and performs the followings operations:
    - look for the best hyperparameters using cross-validated grid-search (using the model and grid specify ) 
    - print the best parameters and results obtained in the training 
    - save the the scoring obtained with all the paramereter tryed in a csv inside models/GridCv
    - if a test set is provided it prints the results obtained in the test 
    """
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


def regressor_gridCV(X_train, y_train, X_test = None,  y_test = None, cv = 3, scoring = "neg_mean_squared_error", params = {}, reg = xgb.XGBRegressor(), model_name = "model", random_state = 123):
    """
    This function it used for a regression problem and performs the followings operations:
    - look for the best hyperparameters using cross-validated grid-search (using the model and grid specified) 
    - print the best parameters and results obtained in the training 
    - save the the scoring obtained with all the paramereter tryed in a csv inside models/GridCv
    - if a test set is provided it prints the results obtained in the test 
    """
    grid = GridSearchCV(reg,params,scoring = scoring, cv = StratifiedKFold(n_splits=cv, random_state = random_state, shuffle = True), n_jobs = -1)
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
            print("Root squared error = {}".format(mean_squared_error(y_test, result)))
            print("Mean absolute error = {}".format(mean_absolute_error(y_test, result)))
            print("R2 score = {}".format(r2_score(y_test, result)))
    return(model)

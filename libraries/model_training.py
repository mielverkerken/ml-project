import matplotlib.pyplot as plt
import util.helpers as H

from util.constants import * 
from util.visanim import animate_sample

from libraries.data_split import StratifiedGroupKFold
from libraries.analyse import *

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import cross_val_score

from IPython.display import HTML, display
import pandas as pd
import math

def tune_pipeline(x_data, y_data, model, scaler, tuned_param,sorted_labels, groups=None, n_jobs=-1,  k=5, verbose=True, graph=True, confusion_matrix=True, learning_curve=True):
  pipe = Pipeline([
                   ("scale", scaler),
                   ("model", model)
  ], verbose=False)

  splitter = StratifiedGroupKFold(k)
    
  scoring = {'mapk': H.mapk_scorer, 'topk': H.top3_acc_scorer, 'accuracy': make_scorer(accuracy_score)}

  CV = GridSearchCV(pipe, tuned_param, cv=splitter, scoring=scoring, refit="mapk", verbose=1, return_train_score=True, n_jobs=n_jobs)
  CV.fit(x_data, y_data, groups)

  mapk_means = CV.cv_results_['mean_test_mapk']
  mapk_stds = CV.cv_results_['std_test_mapk']
  topk_means = CV.cv_results_['mean_test_topk']
  topk_stds = CV.cv_results_['std_test_topk']
  accuracy_means = CV.cv_results_['mean_test_accuracy']
  accuracy_stds = CV.cv_results_['std_test_accuracy']
  train_mapk_means = CV.cv_results_['mean_train_mapk']
  train_topk_means = CV.cv_results_['mean_train_topk']
  train_accuracy_means = CV.cv_results_['mean_train_accuracy']


  if verbose:
    print("Best parameters set found on development set: ",CV.best_params_)
    print("Grid scores on training data set:")
    print("mapk, topk, accuracy")
    print()
    for mapk_m, mapk_s, params, topk_m, topk_s, acc_m, acc_s in zip(mapk_means, mapk_stds, CV.cv_results_['params'], topk_means, topk_stds, accuracy_means, accuracy_stds):
        print("%0.3f (+/-%0.03f), %0.3f (+/-%0.03f), %0.3f (+/-%0.03f) for %r" % (mapk_m, mapk_s * 2, topk_m, topk_s *2, acc_m, acc_s *2, params))

  print(f"Mean map@3 of best model: {CV.best_score_}")


  if graph:
    plt.figure()
    plt.plot(np.log10(tuned_param['model__C']),train_mapk_means,'g-',label="mapk (T)")
    plt.plot(np.log10(tuned_param['model__C']),mapk_means,'r-',label="mapk (V)")
    plt.plot(np.log10(tuned_param['model__C']),train_topk_means,'g--',label="topk (T)")
    plt.plot(np.log10(tuned_param['model__C']),topk_means,'r--',label="topk (V)")
    plt.plot(np.log10(tuned_param['model__C']),train_accuracy_means,'g-.',label="accuracy (T)")
    plt.plot(np.log10(tuned_param['model__C']),accuracy_means,'r-.',label="accuracy (V)")
    plt.xlabel("log_{10} of C")
    plt.ylabel("score")
    plt.legend()    


  if confusion_matrix:
    plot_confusion_matrix(x_data, y_data, groups, CV, sorted_labels, k)
    
  if learning_curve:
    plot_learning_curve(x_data, y_data, groups, CV, k=k)

  return CV


def feature_election(x_data, y_data, CV, groups, feature_range, plot=True):
    f_score_len = len(list(feature_range))
    f_scores = np.zeros(f_score_len)

    for i, num_features in enumerate(feature_range):
        # Select k best features based on training data
        selector = SelectKBest(f_classif, k=num_features).fit(x_data, y_data)
        # extract these features from training data
        x_new=selector.transform(x_data)
        # build and CV logistic regression model with these features
        f_scores[i] = cross_val_score(CV.best_estimator_, x_new, y_data, cv=StratifiedGroupKFold(5), groups=groups, scoring=H.mapk_scorer ).mean()
        print(str(num_features) + " Features gives Cross Validation Score (map@3) : ",f_scores[i])
                  
    if plot:
        plt.figure()
        plt.plot(list(feature_range), f_scores)
        plt.show()

    i = np.argmax(f_scores)
    opt_features = list(feature_range)[i]

    print()
    print("Optimal performance of ",f_scores[i],", for ",opt_features," features")
    
    return f_scores, i, opt_features   

def tune_pipeline_2(x_data, y_data, model, scaler, tuned_param, sorted_labels, feature_selection, imputer, fileName, groups=None, n_jobs=-1,  k=5, n_original=False, verbose=True, confusion_matrix=True, learning_curve=False):
    pipe = Pipeline([
      ("scale", scaler),
      ("imputer", imputer),
      ("feature_selection", feature_selection),
      ("model", model)
    ], verbose=False)

    splitter = StratifiedGroupKFold(k, n_original)
      
    scoring = {'mapk': H.mapk_scorer, 'topk': H.top3_acc_scorer, 'accuracy': make_scorer(accuracy_score)}

    CV = GridSearchCV(pipe, tuned_param, cv=splitter, scoring=scoring, refit="mapk", verbose=1, return_train_score=True, n_jobs=n_jobs)
    CV.fit(x_data, y_data, groups)

    print("Best parameters set found on development set: ",CV.best_params_)
    print(f"Mean map@3 of best model: {CV.best_score_}")   

    results_to_df(CV, tuned_param, show=verbose, fileName=fileName)

    if confusion_matrix:
      plot_confusion_matrix(x_data, y_data, groups, CV, sorted_labels, k)

    if learning_curve:
        plot_learning_curve(x_data, y_data, groups, CV, k=k, n_original=n_original)

    return CV 

def results_to_df(CV, tuned_parameters, show=True, fileName=None):
    cv_results_test = {
      'mean_test_mapk': CV.cv_results_['mean_test_mapk'], 
      'std_test_mapk': CV.cv_results_['std_test_mapk'],
      'mean_test_topk': CV.cv_results_['mean_test_topk'],
      'std_test_topk': CV.cv_results_['std_test_topk'],
      'mean_test_accuracy': CV.cv_results_['mean_test_accuracy'],
      'std_test_accuracy': CV.cv_results_['std_test_accuracy'],
    } 

    cv_results_train = {
      'mean_train_mapk': CV.cv_results_['mean_train_mapk'], 
      'std_train_mapk': CV.cv_results_['std_train_mapk'],
      'mean_train_topk': CV.cv_results_['mean_train_topk'],
      'std_train_topk': CV.cv_results_['std_train_topk'],
      'mean_train_accuracy': CV.cv_results_['mean_train_accuracy'],
      'std_train_accuracy': CV.cv_results_['std_train_accuracy'],
    }

    keys = []

    for param in tuned_parameters:
      keys.append(param)
      cv_results_test[param] = CV.cv_results_[f"param_{param}"].data
      cv_results_train[param] = CV.cv_results_[f"param_{param}"].data

    test_result = pd.DataFrame(cv_results_test)
    train_result = pd.DataFrame(cv_results_train)
    results = pd.merge(test_result, train_result, on=keys)

    if (show):
      print("Validation Results:")
      display(test_result)
      print("Train Results:")
      display(train_result)

    if (fileName):
      results.to_csv(f'../results/{fileName}.csv', index = None, header=True)
    
    return train_result, test_result, results

def get_n_numbers(n, start, end):
    return np.array([math.ceil(x) for x in np.linspace(0,1,n+1) * (end - start) + start])[1:]
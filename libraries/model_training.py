import numpy as np
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
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.base import BaseEstimator, ClassifierMixin

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
    scoring = {'topk': H.top3_acc_scorer_new,
               'mapk':  H.mapk_scorer_new,
               'accuracy': make_scorer(accuracy_score)}


    CV = GridSearchCV(pipe, tuned_param, cv=splitter, scoring=scoring, verbose=1, n_jobs=n_jobs, refit='mapk', return_train_score = True)
    CV.fit(x_data, y_data, groups)

    print("Best parameters set found on development set: ",CV.best_params_)
    print(f"Mean map@3 of best model: {CV.best_score_}")   

    results_to_df(CV, tuned_param, show=verbose, fileName=fileName)

    if confusion_matrix:
        plot_confusion_matrix(x_data, y_data, groups, CV, sorted_labels, splitter=splitter)

    if learning_curve:
        plot_learning_curve(x_data, y_data, groups, CV, scoring=scoring, splitter=splitter)

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
      results.to_csv(f'../results/{fileName}.csv', index=None, header=True)

    return train_result, test_result, results

def get_n_numbers(n, start, end):
    return np.array([math.ceil(x) for x in np.linspace(0,1,n+1) * (end - start) + start])[1:]


class FeatureSelection_Supervised(TransformerMixin, BaseEstimator):
    def __init__(self, model_feature=None, threshold=None):
        self.threshold = threshold
        self.ss = None
        self.model_feature = model_feature

    def fit(self, X, y):
        self.model_feature.fit(X, y)
        if hasattr(self.model_feature, 'coef_'):
            print(f"Mean importance: {self.model_feature.coef_.mean()}")
        if hasattr(self.model_feature, 'feature_importances_'):
            print(f"Mean importance: {self.model_feature.feature_importances_.mean()}")
        if not (hasattr(self.model_feature, 'coef_') or hasattr(self.model_feature, 'feature_importances_')):
            raise Exception('model_feature should contain feature_importances_ or coef_')

        self.ss = SelectFromModel(self.model_feature, prefit=False, threshold=self.threshold).fit(X, y)
        return self

    def transform(self, X):
        x_new = self.ss.transform(X)
        print(f"Number of features (initial): {x_new.shape[1]} ({X.shape[1]})")
        return x_new

class Ensemble2class(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(self, model1, model2):
        """
        Called when initializing the classifier
        """
        self.model1 = model1
        self.model2 = model2

    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """
        # First model
        y1 = np.copy(y)
        y1[y1 == 8] = 3
        y1[y > 8] = y1[y > 8] - 1
        self.model1.fit(X, y1)

        # Second model
        mask = np.logical_or(y == 8, y == 3)
        y2 = np.copy(y)[mask]
        y2[y2 == 3] = 0
        y2[y2 == 8] = 1
        self.model2.fit(X[mask], y2)
        self.classes_, y = np.unique(y, return_inverse=True)
        return self

    def predict(self, X, y=None):
        # pdb.set_trace()
        return np.flip(np.argsort(self.predict_proba(X), axis=1), axis=1)[:, 0]

    def predict_proba(self, X):
        X_ = np.zeros((len(X),18))
        pred1 = self.model1.predict_proba(X)
        mask = np.flip(np.argsort(pred1, axis=1), axis=1)[:, 0] == 3
        X_[:,0:3] = pred1[:, 0:3]
        X_[:,4:8] = pred1[:, 4:8]
        X_[:,9:18] = pred1[:, 8:17]
        pred2 = self.model2.predict_proba(X[mask])
        X_[mask, 3] = pred2[:, 0] * pred1[mask, 3]
        X_[mask, 8] = pred2[:, 1] * pred1[mask, 3]
        return X_
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


def optimize_params(X, y, params_space, validation_split=0.2):
     """Estimate a set of 'best' model parameters."""
     # Split X, y into train/validation
     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, stratify=y)

    # Estimate XGB params
    def objective(_params):
        _clf = XGBClassifier(n_estimators=10000,
                             max_depth=int(_params['max_depth']),
                             learning_rate=_params['learning_rate'],
                             min_child_weight=_params['min_child_weight'],
                             subsample=_params['subsample'],
                             colsample_bytree=_params['colsample_bytree'],
                             gamma=_params['gamma'])
        _clf.fit(X_train, y_train,
                 eval_set=[(X_train, y_train), (X_val, y_val)],
                 eval_metric='auc',
                 early_stopping_rounds=30)
        y_pred_proba = _clf.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_true=y_val, y_score=y_pred_proba)
        return {'loss': 1. - roc_auc, 'status': STATUS_OK}

    trials = Trials()
    return fmin(fn=objective,
                space=params_space,
                algo=tpe.suggest,
                max_evals=100,
                trials=trials,
                verbose=0)


class OptimizedXGB(BaseEstimator, ClassifierMixin):
    """XGB with optimized parameters.

    Parameters
    ----------
    custom_params_space : dict or None
        If not None, dictionary whose keys are the XGB parameters to be
        optimized and corresponding values are 'a priori' probability
        distributions for the given parameter value. If None, a default
        parameters space is used.
    """
    def __init__(self, custom_params_space=None):
        self.custom_params_space = custom_params_space

    def fit(self, X, y, validation_split=0.3):
        """Train a XGB model.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data.

        y : ndarray, shape (n_samples,) or (n_samples, n_labels)
            Labels.

        validation_split : float (default: 0.3)
            Float between 0 and 1. Corresponds to the percentage of samples in X which will be used as validation data to estimate the 'best' model parameters.
        """
        # If no custom parameters space is given, use a default one.
        if self.custom_params_space is None:
            _space = {
                'learning_rate': hp.uniform('learning_rate', 0.0001, 0.05),
                'max_depth': hp.quniform('max_depth', 8, 15, 1),
                'min_child_weight': hp.quniform('min_child_weight', 1, 5, 1),
                'subsample': hp.quniform('subsample', 0.7, 1, 0.05),
                'gamma': hp.quniform('gamma', 0.9, 1, 0.05),
                'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 0.7, 0.05)
            }
        else:
            _space = self.custom_params_space

        # Estimate best params using X, y
        opt = optimize_params(X, y, _space, validation_split)

        # Instantiate `xgboost.XGBClassifier` with optimized parameters
        best = XGBClassifier(n_estimators=10000,
                             max_depth=int(opt['max_depth']),
                             learning_rate=opt['learning_rate'],
                             min_child_weight=opt['min_child_weight'],
                             subsample=opt['subsample'],
                             gamma=opt['gamma'],
                             colsample_bytree=opt['colsample_bytree'])
        best.fit(X, y)
        self.best_estimator_ = best
        return self

    def predict(self, X):
        """Predict labels with trained XGB model.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)

        Returns
        -------
        output : ndarray, shape (n_samples,) or (n_samples, n_labels)
        """
        if not hasattr(self, 'best_estimator_'):
            raise NotFittedError('Call `fit` before `predict`.')
        else:
            return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        """Predict labels probaiblities with trained XGB model.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)

        Returns
        -------
        output : ndarray, shape (n_samples,) or (n_samples, n_labels)
        """
        if not hasattr(self, 'best_estimator_'):
            raise NotFittedError('Call `fit` before `predict_proba`.')
        else:
            return self.best_estimator_.predict_proba(X)
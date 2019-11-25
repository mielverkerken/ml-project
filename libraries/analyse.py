from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np
import matplotlib.pyplot as plt
from libraries.data_split import StratifiedGroupKFold
from sklearn.model_selection import learning_curve
import util.helpers as H
from sklearn.metrics import make_scorer
import matplotlib.cm as cm

def _plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues, print_cm=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if print_cm:
        print(cm)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('ConfusionMartrix.png', bbox_inches='tight')
    # fig.show()

def plot_confusion_matrix(X, Y, group, cv, labels, print_cm=True, splitter=StratifiedGroupKFold(5)):

    y_tot_true, y_tot_pred = np.empty(0, dtype=int), np.empty(0, dtype=int)
    for train_index, test_index in splitter.split(X, Y, group):
        x_train, x_test, y_train, y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
        cv.best_estimator_.fit(x_train, y_train)
        y_pred = cv.best_estimator_.predict(x_test)
        y_tot_true = np.concatenate((y_tot_true, y_test), axis=0)
        y_tot_pred = np.concatenate((y_tot_pred, y_pred), axis=0)
    _plot_confusion_matrix(y_tot_true, y_tot_pred, labels, normalize=True, title='Confusion matrix, with normalization', print_cm=print_cm)

def plot_learning_curve(X, y, group, cv, scoring={'mapk': H.mapk_scorer_new}, title=None, splitter=StratifiedGroupKFold(5), shuffle=True):
    colormap_train = cm.Reds(np.flip(np.linspace(0.7, 1, len(scoring))))
    colormap_test = cm.Greens(np.flip(np.linspace(0.7, 1, len(scoring))))

    plt.figure()
    for ind, score_func in enumerate(scoring):
        train_sizes, train_scores, valid_scores = learning_curve(cv.best_estimator_, X, y,
                                                                 groups=group,
                                                                 train_sizes=np.linspace(0.1, 1.0, 5),
                                                                 cv=splitter,
                                                                 verbose=False, scoring=scoring[score_func],
                                                                 n_jobs=-1, shuffle=shuffle)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(valid_scores, axis=1)
        test_scores_std = np.std(valid_scores, axis=1)
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color=colormap_train[ind])
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color=colormap_test[ind])
        plt.plot(train_sizes, train_scores_mean, 'o-', color=colormap_train[ind],
                 label=f"Training score {score_func}")
        plt.plot(train_sizes, test_scores_mean, 'o-', color=colormap_test[ind],
                 label=f"Cross-validation score {score_func}")
        print(f"mean validation scores {score_func} in learning curve: {np.mean(valid_scores, axis=1)}")
        if title:
            plt.title(title)

    # Reordering legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.gca().legend(handles=handles[0::2]+handles[1::2], prop={'size': 8})

    plt.grid()
    plt.xlabel("Training examples")
    plt.ylabel("Precision")
    plt.savefig(f'LearningCurve.png', bbox_inches='tight')
    # plt.show()

def top_n_logistic_model_coefficients(CV, X):
    for i in range(CV.best_estimator_['model'].coef_.shape[0]):
        feature_weights = zip(X.columns, CV.best_estimator_['model'].coef_[i])
        print("***label{}***".format(i))
        for a, b in sorted(feature_weights, key = lambda t: t[1], reverse=True)[0:5]:
            print(a,b)
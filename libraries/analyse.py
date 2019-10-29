from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np
import matplotlib.pyplot as plt
from libraries.data_split import StratifiedGroupKFold
from sklearn.model_selection import learning_curve
import util.helpers as H

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
    if print:
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
    return ax

def plot_confusion_matrix(X, Y, group, cv, labels, k=5, print=True):
    splitter = StratifiedGroupKFold(k)
    y_tot_true, y_tot_pred = np.empty(0, dtype=int), np.empty(0, dtype=int)
    for train_index, test_index in splitter.split(X, Y, group):
        x_train, x_test, y_train, y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
        cv.fit(x_train, y_train, group)
        y_pred = cv.predict(x_test)
        y_tot_true = np.concatenate((y_tot_true, y_test), axis=0)
        y_tot_pred = np.concatenate((y_tot_pred, y_pred), axis=0)
    _plot_confusion_matrix(y_tot_true, y_tot_pred, labels, normalize=True, title='Confusion matrix, with normalization', print_cm=print)

def plot_learning_curve(X, y, group, cv, k=5):
    train_sizes, train_scores, valid_scores = learning_curve(cv, X, y, groups=group, train_sizes=np.linspace(0.3, 1.0, 8), cv=StratifiedGroupKFold(k), scoring=H.mapk_scorer)
    plt.figure()
    plt.plot(train_sizes, train_scores, 'g-', label="train")
    plt.plot(train_sizes, valid_scores, 'r-', label="validate")
    plt.xlabel("Training examples (%)")
    plt.ylabel("map@3")
    plt.legend()
    plt.show()
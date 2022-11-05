import pandas as pd

from sklearn.metrics import (f1_score, accuracy_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay,
                            precision_recall_curve, PrecisionRecallDisplay, roc_curve, RocCurveDisplay)


def get_f1_score(labels, preds, average='macro'):
    """
    Get the f1 score for the given label and prediction set
    """

    f1 = f1_score(labels, preds, average=average)

    return f1

def get_accuracy(labels, preds, raw_correct=True):
    """
    Get the accuracy of the predictions. If raw correct is true, simply return number of correct predictions.
    """

    return accuracy_score(labels, preds, normalize=raw_correct)

def get_precision_recall(labels, preds, average="macro"):
    """
    Get the precision and recall of the model predictions.
    """

    recall = recall_score(labels, preds, average=average)
    precision = precision_score(labels, preds, average=average)

    return (precision, recall)


def get_confusion_matrix_df(labels, preds, label_dict):
    """
    Generates a confusion matrix in dataframe form given the true labels and predictions.
    """  
    cm_df = pd.DataFrame(confusion_matrix(labels, preds), 
                    index=[i[1] for i in label_dict.items()], 
                    columns=[i[1] for i in label_dict.items()])
    return cm_df

def plot_confusion_matrix_display(labels, preds, figsize=(15, 8)):
    """
    Gets the Confusion Matrix Display object from Scikit learn.
    The object takes care of all plotting.
    """

    conf_mat = ConfusionMatrixDisplay.from_predictions(labels, preds)

    return conf_mat

def plot_roc_curve(labels, preds, probs):
    """
    Gets the ROC curve display object from scikit learn, which can be used to plot.
    """

    fpr, tpr, _ = roc_curve(labels, probs[:, 1])
    roc = RocCurveDisplay(fpr=fpr, tpr=tpr)

    return roc

def get_feature_importance(train_features, trained_model):
    """
    Get the importances of each feature in making a correct prediction.
    Returns a pandas dataframe.
    """

    fi = pd.DataFrame(
        {
            'Feature':train_features,
            'Importance':trained_model.feature_importances_
        }
    ).sort_values('Importance', ascending=False)

    return fi

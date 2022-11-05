from sklearn.metrics import f1_score

def get_f1_score(labels, preds, average='macro'):
    """
    Get the f1 score for the given label and prediction set
    """

    f1 = f1_score(labels, preds, average=average)

    return f1

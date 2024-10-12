from sklearn.metrics import accuracy_score, auc, precision_score, recall_score, f1_score, confusion_matrix, roc_curve

def accuracy(true: list, pred: list):
    return accuracy_score(true, pred)

def auc(true: list, pred: list):
    return

def precision(true: list, pred: list, avg: str):
    return precision_score(true, pred, average=avg)

def recall(true: list, pred: list, avg: str):
    return recall_score(true, pred, average=avg)

def f1(true: list, pred: list, avg: str):
    return f1_score(true, pred, average=avg)

def conf_mat(true: list, pred: list):
    confusion_matrix(true, pred)

def far95():
    pass

def roc_curve():
    return fpr, tpr, thresholds
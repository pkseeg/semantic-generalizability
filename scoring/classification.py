from sklearn.metrics import precision_recall_fscore_support

def f1(ytrues, yhats):
    p, r, f, s = precision_recall_fscore_support(ytrues, yhats, average=["micro"])
    return f
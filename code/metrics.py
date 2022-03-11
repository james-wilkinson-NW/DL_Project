import numpy as np
from sklearn.metrics import roc_curve

def topK(probs, truths, k = 5):
    '''

    returns which fraction of the time the model predicted the ground truth within the top K predictions

    Args:
        k: threshold to define success. If truth is in top "k" predictions
        probs: 2d array of 1d vectors of probabilities, one vector per sample
        truths: 1d list of the indexes of the ground truths. one index per sample

    Returns: fraction (0->1) of the samples where the ground truth was in the top K predictions
    '''

    top_ids = np.argpartition(probs, -k)[:, -k:]

    success_count = 0
    for i in range(len(probs)):
        if truths[i] in top_ids[i]:
            success_count += 1

    return float(success_count)/len(probs)


def EER_metric(probs, truths):
    '''

    Args:
        probs: 2d array of 1d vectors of probabilities, one vector per sample
        truths: 1d list of the indexes of the ground truths. one index per sample

    Returns: macro average EER across all classes

    '''

    # turn truths into one-hot
    probs = np.array(probs)
    y_true = np.zeros(shape=probs.shape)
    for i in range(len(y_true)):
        y_true[i][truths[i]] = 1

    # Compute ROC curve and ROC area for each class
    n_classes = len(probs[0])
    EERs = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], probs[:, i])
        fnr = 1-tpr
        EER_class = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        EERs.append(EER_class)

    return sum(EERs)/len(EERs) # macro average


if __name__ == '__main__': # basic testing
    ps = [[0, 0.1, 0.2, 0.1, 0.6],
          [0.5, 0.2, 0.2, 0.1, 0],
          [0.3, 0.4, 0.2, 0.1, 0],
          [0.2, 0.2, 0.5, 0.1, 0],
          [0, 0.1, 0.2, 0.6, 0.1],
          [0, 0.1, 0.2, 0.6, 0.1]]

    T = [4, 0, 2, 1, 3, 2]
    print(EER_metric(ps, T))

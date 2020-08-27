from sklearn.metrics import roc_curve, auc
from matplotlib import plot as plt

def ROC_AUC(ground_truth, prediction):
    
    """
    Parameters
    ----------
    ground_truth : numpy.array.ravel()
        vector of binary values corresponding to ground truth segmentation.
    prediction : numpy.array.ravel()
        vector of probabilities between 0-1 corresponding to predicted segmentation.

    Returns
    -------
    None.
    """
    
    fpr, tpr, _ = roc_curve(ground_truth,prediction)
    roc_auc = auc(fpr,tpr)
    
    fig, ax = plt.subplots(1,1)
    ax.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    ax.legend(loc="lower right")
    
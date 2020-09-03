from sklearn.metrics import auc
from matplotlib import pyplot as plt

def ROC_AUC(FPR_list, TPR_list):
    
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
    roc_auc = auc(FPR_list,TPR_list)
    
    fig, ax = plt.subplots(1,1)
    ax.plot(FPR_list, TPR_list, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    ax.legend(loc="lower right")
    
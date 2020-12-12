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
    try: 
        roc_auc = auc(FPR_list,TPR_list)
    except:
        print("FPR,TPR not monotonic. Continuing without calculating AUC")
    
    fig, ax = plt.subplots(1,1)
    ax.plot(FPR_list, TPR_list, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('1 - Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_title('Receiver-Operating Characteristic')
    ax.legend(loc="lower right")
    
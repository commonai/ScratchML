import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions as plt_decision_region

def plot_loss(loss_array):
    """plots the loss using matplotlib
    
    Args:
        loss_array (list): list of values
    """
    plt.plot(loss_array, label='line 1', linewidth=2)
    plt.xlabel('n iterations')
    plt.ylabel('loss')

def plot_decision_regions(X, Y, classifier, xlabel="", ylabel=""):
    """Plots decision boundary and region of classifier predictions
    
    Args:
        X (np.array): (n_samples, n_features)
        Y (np.arary): (n_samples)
        classifier: classifer, must have predict method
        xlabel (str, optional): Defaults to "".
        ylabel (str, optional): Defaults to "".
    """
    plt_decision_region(X=X, y=Y, clf=classifier)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

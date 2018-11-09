import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions as plt_decision_region

def plot_loss(loss_array):
    plt.plot(loss_array, label='line 1', linewidth=2)
    plt.xlabel('n iterations')
    plt.ylabel('loss')

def plot_decision_regions(X, Y, classifier, xlabel="", ylabel=""):
    plt_decision_region(X=X, y=Y, clf=classifier)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

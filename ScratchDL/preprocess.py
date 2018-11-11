import numpy as np

def one_hot_encode(data):
    """turns data into onehot encoding
    
    Args:
        data (np.array): (n_samples,)
    
    Returns:
        np.array: shape (n_samples, n_classes)
    """
    n_classes = np.unique(data).shape[0]
    onehot = np.zeros((data.shape[0], n_classes))
    for i, val in enumerate(data.astype(int)):
        onehot[i, val] = 1.
    return onehot

import os
import struct
import numpy as np
from sklearn import datasets


def load_iris():
    """Loads the iris dataset form scikit-learn"""

    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    Y = iris.target
    return X, Y


def load_digits():
    """downloads the MNIST dataset

    Returns:
        [type]: [description]
    """
    digits = datasets.load_digits()
    X = digits.data
    Y = digits.target
    return X, Y


def load_mnist():
    """Load normalized MNIST data"""
    dir_path = os.path.dirname(__file__)
    labels_path = \
        os.path.join(dir_path, "data", 'train-labels-idx1-ubyte')
    images_path = \
        os.path.join(dir_path, "data", 'train-images-idx3-ubyte')

    test_labels_path = \
        os.path.join(dir_path, "data", 't10k-labels-idx1-ubyte')
    test_images_path = \
        os.path.join(dir_path, "data", 't10k-images-idx3-ubyte')

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(test_labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        test_labels = np.fromfile(lbpath,
                                  dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2

    with open(test_images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        test_images = np.fromfile(imgpath,
                                  dtype=np.uint8).reshape(len(test_labels), 784)
        test_images = ((images / 255.) - .5) * 2

    return images, labels, test_images, test_labels

import numpy as np
from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split

def load_dataset(dataset_name="fashion_mnist"):
    """ Load and preprocess dataset. """
    if dataset_name == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    elif dataset_name == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    else:
        raise ValueError("Unsupported dataset. Choose from ['mnist', 'fashion_mnist'].")

    # Normalize and reshape images
    X_train, X_test = X_train.reshape(-1, 784) / 255.0, X_test.reshape(-1, 784) / 255.0

    # Split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

    # One-hot encoding
    num_classes = len(np.unique(y_train))
    y_train, y_val, y_test = np.eye(num_classes)[y_train], np.eye(num_classes)[y_val], np.eye(num_classes)[y_test]

    return X_train, y_train, X_val, y_val, X_test, y_test

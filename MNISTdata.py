import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_data():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X /= 256.0
    y = y.astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=0)    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=10000, random_state=0)
    train_data = (X_train, y_train)
    test_data = (X_test, y_test)
    validation_data = (X_val, y_val)

    # data above is stored as tuple of inputs and outputs
    # change to a list of matching input/output tuples
    if False:
        train_data = [*zip(*train_data)]
        validation_data = [*zip(*validation_data)]
        test_data = [*zip(*test_data)]
    
    return (train_data, validation_data, test_data)

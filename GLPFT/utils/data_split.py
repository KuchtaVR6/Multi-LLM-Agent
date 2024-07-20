import numpy as np


def train_validation_split(X, y, test_size=0.1, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    # Shuffle indices
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    # Determine split point
    split_index = int(X.shape[0] * (1 - test_size))

    # Split indices into train and test sets
    train_indices = indices[:split_index]
    validation_indices = indices[split_index:]

    # Split data into training and testing sets
    X_train, X_validation = X[train_indices], X[validation_indices]
    y_train, y_validation = y[train_indices], y[validation_indices]

    return X_train, X_validation, y_train, y_validation
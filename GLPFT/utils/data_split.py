import numpy as np


def train_validation_split(inputs, targets, test_size=0.1):
    # Ensure inputs and targets are lists of the same length
    assert len(inputs) == len(targets), "Inputs and targets must be the same length"

    # Calculate the number of validation samples
    num_samples = len(inputs)
    num_val_samples = int(num_samples * test_size)

    # Generate random indices
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    # Split the indices for training and validation
    val_indices = indices[:num_val_samples]
    train_indices = indices[num_val_samples:]

    # Split the inputs and targets using the indices
    inputs_train = [inputs[i] for i in train_indices]
    inputs_val = [inputs[i] for i in val_indices]
    targets_train = [targets[i] for i in train_indices]
    targets_val = [targets[i] for i in val_indices]

    return inputs_train, inputs_val, targets_train, targets_val
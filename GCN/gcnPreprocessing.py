import sklearn.model_selection
import numpy as np


def get_y_from_indices(y, mask, indices):
    """Construct vector y and its corresponding mask from label indices.

    This method constructs a vector y and a mask m from some indices given.
    Both returned vectors will be boolean and have shape (n,). y will
    contain 1 at positions where the corresponding node in the GCN is positive
    (disease gene) and mask will contain 1 at positions of the indices.

    Parameters:
    ----------
    y:                  The targets for all nodes
    mask:               The mask for all known nodes
    indices:            The indices selected for the set that is
                        to be constructed.
                        All elements of indices must be in the range (0, y.shape)

    Returns:
    A tuple of two numpy vectors. The first contains the labels and the
    second contains the corresponding mask.
    """
    assert (y.shape[0] == mask.shape[0])
    # construct the mask
    m = np.zeros_like(mask)
    m[indices] = 1
    # construct y
    y_sub = np.zeros_like(y)
    y_sub[indices] = y[indices]
    
    return y_sub, m

def train_test_split(y, mask, val_size):
    """Split mask and targets into train and validation sets (stratified).

    This method contructs mask and targets for training and validation
    from the complete mask and targets. The proportion of nodes used
    for validation is determined by `val_size`.
    The split will be stratified and the returned arrays have the
    same dimensions as the input arrays.

    Parameters:
    ----------
    y:                  The targets for all nodes.
    mask:               The mask for the known nodes
    val_size:           The proportion (or absolute size) of nodes
                        used for validation
    
    Returns:
    Four arrays of length of the input arrays, namely train targets,
    train mask, validation targets and validation mask.
    """
    assert (y.shape[0] == mask.shape[0])
    mask_idx = np.where(mask == 1)[0]
    train_idx, val_idx = sklearn.model_selection.train_test_split(mask_idx,
                                                                  test_size=val_size,
                                                                  stratify=y[mask==1, 0]
    )
    # build the train/validation masks
    m_train, y_train = get_y_from_indices(y, mask, train_idx)
    m_val, y_val = get_y_from_indices(y, mask, val_idx)
    return y_train, m_train, y_val, m_val

def cross_validation_sets(y, mask, folds):
    """Builds labels and masks for k-fold cross validation.

    Constructs four different sets for k different folds. The four sets
    are all boolean vectors of length n, where n is the number of nodes
    in the GCN and the shape of y and mask.
    The four sets per fold are training labels, testing labels, training
    mask and testing mask, respectively.
    The sets are splitted in a stratified manner, for instance in a 5-fold
    CV, there should be 1/5 of all the positives in the test set.

    Parameters:
    ----------
    y:                  The targets/labels for all nodes. y has to be a
                        binary vector and contain a 1 for positive nodes
                        and 0 for negative nodes.
    mask:               Similar to y, mask has to be a binary vector,
                        containing a 1 for all nodes that are labelled
                        (no matter if positive or negative).
                        y and mask have to have the same first dimension.
    folds:              The number of folds to split the data to, i.e. k

    Returns:
    A list of length `folds` or k. Each of the elements in the list is a tuple
    of length four, containing the training labels, testing labels, training
    mask and testing mask, respectively.
    """
    label_idx = np.where(mask == 1)[0] # get indices of labelled genes
    kf = sklearn.model_selection.StratifiedKFold(n_splits=folds, shuffle=True)
    splits = kf.split(label_idx, y[label_idx])
    k_sets = []
    for train, test in splits:
        # get the indices in the real y and mask realm
        train_idx = label_idx[train]
        test_idx = label_idx[test]
        # construct y and mask for the two sets
        y_train, train_mask = get_y_from_indices(y, mask, train_idx)
        y_test, test_mask = get_y_from_indices(y, mask, test_idx)
        k_sets.append((y_train, y_test, train_mask, test_mask))
    return k_sets
import numpy as np


class RollingWindowSplit(object):
    """
    Class generating the training/test indices for a rolling window
    cross validation, simulating for example daily retraining of a model.
    Useful for time-series problems.
    
    Note
    ----
    Complies with Scikit-Learn conventions for cross-validation iterators such
    as `sklearn.model_selection.ShuffleSplit`.
    """

    def __init__(self, train_set_size, test_set_size=1):
        """
        Initialisation function.
        
        Parameters
        ----------
        
        train_set_size: Integer
            Absolute size of the training set in number of rows.
        
        test_set_size: Integer, optional, default = 1
            Absolute size of the test set in number of rows.
    
        """

        self.train_set_size = train_set_size
        self.test_set_size = test_set_size

    def split(self, X, y=None, groups=None):
        """
        Generator function yielding training and test indices for a rolling
        window cross-validation.
        
        Parameters
        ----------
        
        X: np.array/pd.DataFrame
            Feature matrix.
        
        y: object
            Always ignored, exists for compatibility.
        
        groups: object
            Always ignored, exists for compatibility.
        
        Yields
        ------
        
        train_indices: np.array
            np.array of integers corresponding to the training indices for
            the current iteration.
 
         train_indices: np.array
            np.array of integers corresponding to the test indices for
            the current iteration.           
            
        """

        start = self.train_set_size
        stop = len(X) - self.test_set_size + 1
        for i in range(start, stop):
            train_indices = np.arange(i - self.train_set_size, i)
            test_indices = np.arange(i, i + self.test_set_size)
            yield train_indices, test_indices

import numpy as np
# pandas might be needed if get_dummies or other pd functions are used inside mocks in future.
# For now, only numpy is strictly necessary for np.zeros, np.unique, np.full.
# import pandas as pd

class MockClassifier:
    def __init__(self):
        self.n_classes_ = 1 # Default for binary or non-multilabel
        self.y_shape_ = None
        self.classes_ = [0] # Default classes

    def fit(self, X, y):
        self.y_shape_ = y.shape
        if len(y.shape) > 1 and y.shape[1] > 1: # multilabel-indicator
            self.n_classes_ = y.shape[1]
            self.classes_ = np.arange(self.n_classes_)
        elif y.ndim == 1: # binary or multiclass
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
        else: # Single class in multilabel or other edge cases e.g. y.shape = (n,1)
            self.classes_ = np.unique(y.ravel()) # Flatten to handle (n,1)
            self.n_classes_ = len(self.classes_)
            if self.n_classes_ == 0 : # if y was empty or all NaNs resulting in no classes
                self.n_classes_ = 1
                self.classes_ = [0] # Fallback for empty y

        return self

    def predict(self, X):
        # Ensure fit has been called and y_shape_ is not None
        if self.y_shape_ is None:
            raise RuntimeError("The classifier has not been fitted yet.")

        if len(self.y_shape_) > 1 and self.y_shape_[1] > 1: # multilabel-indicator
            preds = np.zeros((len(X), self.n_classes_))
            if self.n_classes_ > 0:
                 preds[:, 0] = 1 # Predict the first class for all samples
            return preds
        else: # binary or multiclass
            # Predict the first class encountered during fit, or 0 if no classes were seen
            # This ensures predict always returns something based on what fit observed.
            first_class_to_predict = self.classes_[0] if len(self.classes_) > 0 else 0
            return np.full(len(X), first_class_to_predict)


    def score(self, X, y):
        # Return a dummy score
        return 0.5

    def get_params(self, deep=True):
        # scikit-learn's clone function expects this method.
        return {}

class AnotherMockClassifier:
    def __init__(self, param=None):
        self.param = param
        self.n_classes_ = 1
        self.y_shape_ = None
        self.classes_ = [0]


    def fit(self, X, y):
        self.y_shape_ = y.shape
        if len(y.shape) > 1 and y.shape[1] > 1:
            self.n_classes_ = y.shape[1]
            self.classes_ = np.arange(self.n_classes_)
        elif y.ndim == 1:
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
        else:
            self.classes_ = np.unique(y.ravel())
            self.n_classes_ = len(self.classes_)
            if self.n_classes_ == 0 :
                self.n_classes_ = 1
                self.classes_ = [0]
        return self

    def predict(self, X):
        if self.y_shape_ is None:
            raise RuntimeError("The classifier has not been fitted yet.")

        if len(self.y_shape_) > 1 and self.y_shape_[1] > 1:
            preds = np.zeros((len(X), self.n_classes_))
            if self.n_classes_ > 0:
                preds[:, 0] = 1
            return preds
        else:
            first_class_to_predict = self.classes_[0] if len(self.classes_) > 0 else 0
            return np.full(len(X), first_class_to_predict)

    def score(self, X, y):
        return 0.6

    def get_params(self, deep=True):
        return {"param": self.param}

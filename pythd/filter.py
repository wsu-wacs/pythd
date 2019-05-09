"""
Filter functions for MAPPER.

Original code by Xiu Huan Yap <yap.4@wright.edu>
Rewritten and modified by Kyle Brown <brown.718@wright.edu>
"""
from abc import ABC, abstractmethod
import numpy as np
import scipy as sp

class BaseFilter(ABC):
    @abstractmethod
    def get_values(self, arg):
        """Given a set of points, get the filter values.
        
        Parameters
        ----------
        arg : numpy.ndarray
            The points to transform
        
        Returns
        -------
        numpy.ndarray
            The filter values
        """
        pass
    
    def __call__(self, arg):
        return self.get_values(arg)

class TrainableFilter(BaseFilter):
    @abstractmethod
    def fit(self, arg):
        pass

    @abstractmethod
    def reset(self):
        pass

class CustomFilter(BaseFilter):
    """Use a Python function as a filter
    
    Parameters
    ----------
    f : function
        A function that takes a numpy array and returns the filtered numpy array
    """
    def __init__(self, f):
        self.f = f
    
    def get_values(self, arg):
        return f(arg)

class IdentityFilter(BaseFilter):
    """Identity map filter
    
    This filter just returns the passed value. It is included for convenience
    as it is often useful on low-dimensional datasets.
    """
    def __init__(self):
        pass
    
    def get_values(self, arg):
        return arg

class ComponentFilter(BaseFilter):
    """Use one or more components of the data as the filter
    
    Parameters
    ----------
    comp : int or list
        Either a single index or a list of indices. The filter will return
        only the columns indicated by this value.
    """
    def __init__(self, comp):
        if isinstance(comp, int):
            self.comp = [comp]
        else:
            self.comp = comp
    
    def get_values(self, arg):
        return arg[:, self.comp]

class CombinedFilter(BaseFilter):
    """Combine two or more filters into one filter.
    
    Parameters
    ----------
    *args
        The filters to combine. They will be combined in the order passed.
    """
    def __init__(self, *args):
        self.filters = args
    
    def get_values(self, arg):
        return np.concatenate([f(arg) for f in self.filters], axis=1)

class ScikitLearnFilter(TrainableFilter):
    """Filter functions used in scikit-learn
    
    Parameters
    ----------
    cls: class
        The class of the filter. Should implement the fit_transform method.
    *args
        Positional arguments passed to the scikit-learn class
    **kwargs
        Keyword arguments passedt to the scikit-learn class
    """
    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs
        
        self.reset()
    
    def fit(self, arg):
        self.filt = self.cls(*self.args, **self.kwargs)
        self.filt.fit(arg)
        self.is_fit = True
    
    def reset(self):
        self.filt = self.cls(*self.args, **self.kwargs)
        self.is_fit = False
        
    def get_values(self, arg):
        if not self.is_fit:
            self.fit(arg)
        return filt.transform(arg)

class EccentricityFilter(TrainableFilter):
    """Filter function giving a single eccentricity value.
    
    Parameters
    ----------
    metric : str
        Which distance metric to use. Should match those used by scipy.spatial.distance.pdist
    method : str
        Type of eccentricity to compute. Possible values are:
        * mean - Eccentricity is defined as distance from the mean point
        * medoid - Eccentricity defined as distance from the medoid (point with average minimum distance to all others)
    *args
        Positional arguments passed to scipy's pdist and cdist
    **kwargs
        Keyword arguments passed to scipy's pdist and cdist
    """
    def __init__(self, metric="euclidean", method="mean", *args, **kwargs):
        if method not in ["mean", "medoid"]:
            raise ValueError(f"Unsupported eccentricity method: {method}")

        self.metric = metric.lower()
        self.method = method.lower()
        self.args = args
        self.kwargs = kwargs
        
        self.reset()
    
    def reset(self):
        self.is_fit = False
    
    def fit(self, arg):
        if self.method == "mean":
            self.centroid = np.mean(arg, axis=0, keepdims=True)
        elif self.method == "medoid":
            # Get square distance matrix
            self.dist = sp.spatial.distance.pdist(arg, metric=self.metric, *self.args, **self.kwargs)
            self.dist = sp.spatial.distance.squareform(self.dist)
            # Centroid index
            idx = np.argmin(self.dist.sum(axis=0))
            self.centroid = arg[[idx], :]
        
        self.is_fit = True
    
    def get_values(self, arg):
        if not self.is_fit:
            self.fit(arg)
        
        return sp.spatial.distance.cdist(arg, self.centroid)
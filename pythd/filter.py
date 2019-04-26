"""
Filter functions for MAPPER.

Original code by Xiu Huan Yap <yap.4@wright.edu>
Rewritten and modified by Kyle Brown <brown.718@wright.edu>
"""
from abc import ABC, abstractmethod

class BaseFilter(ABC):
    @abstractmethod
    def get_values(self, arg):
        pass
    
    def __call__(self, arg):
        return self.get_values(arg)

class TrainableFilter(BaseFilter):
    @abstractmethod
    def reset(self):
        pass

class ScikitLearnFilter(TrainableFilter):
    """Filter functions used in scikit-learn
    """
    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs
        
        self.reset()
    
    def reset(self):
        self.fit = False
        self.filt = self.cls(*self.args, **self.kwargs)
    
    def get_values(self, arg):
        
        return filt.fit_transform(arg)

class CustomFilter(BaseFilter):
    """Use a Python function as a filter
    """
    def __init__(self, f):
        self.f = f
    
    def get_values(self, arg):
        return f(arg)

class IdentityFilter(BaseFilter):
    """Identity map filter
    """
    def __init__(self):
        pass
    
    def get_values(self, arg):
        return arg

class ComponentFilter(BaseFilter):
    """Use one or more components of the data as the filter
    """
    def __init__(self, comp):
        self.comp = comp
    
    def get_values(self, arg):
        return arg[:, self.comp]

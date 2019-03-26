"""
Coverings for MAPPER.

Original code by Xiu Huan Yap <yap.4@wright.edu>
Rewritten and modified by Kyle Brown <brown.718@wright.edu>
"""
from abc import ABC, abstractmethod

class BaseCover(ABC):
    pass

class _1DBins:
    """
        Helper class consisting of 1-dimensional bins
    """
    def __init__(self, num_intervals, minv, maxv, overlap):
        self.num_intervals = num_intervals
        self.minv = minv
        self.maxv = maxv
        self.range = float(self.maxv - self.minv)
        self.overlap = overlap
        
        self.rhat = self.range / self.num_intervals
        self.r = self.rhat * (1 + self.overlap / (1.0 - self.overlap))
        self.eps = self.r * 0.5
        
        self.bins = self._get_bins()

    def _get_bins(self):
        bins = []
        for i in range(self.num_intervals):
            c = self.minv + i*self.rhat
            b = (max(c - self.eps, self.minv), min(c + self.eps, self.maxv))
            bins.append(b)
        return bins
        
class IntervalCover1D(BaseCover):
    def __init__(self, bins):
        self.bins = bins
        
    @classmethod
    def EvenlySpaced(self, num_intervals, minv, maxv, overlap):
        pass
    
    
        
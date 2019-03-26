"""
Coverings for MAPPER.

Original code by Xiu Huan Yap <yap.4@wright.edu>
Rewritten and modified by Kyle Brown <brown.718@wright.edu>
"""
from abc import ABC, abstractmethod

class BaseCover(ABC):
    @abstractmethod
    def get_open_set_membership(self, point):
        """Get the open sets a single point belongs to.
        """
        pass
    
    def get_open_set_membership_dict(self, filter_values):
        """Get the open sets a sequence of points belongs to.
        """
        d = {}
        for i, value in enumerate(filter_values):
            nums = self.get_open_set_membership(value)
            for j in nums:
                if j not in d:
                    d[j] = [i]
                else:
                    d[j].append(i)
        return d

class _1DBins:
    """
        Helper class consisting of 1-dimensional bins
    """
    def __init__(self, bins):
        self.bins = bins
        self.num_intervals = len(bins)
        self.minv = bins[0][0]
        self.maxv = bins[-1][1]
    
    @classmethod
    def EvenlySpaced(cls, num_intervals, minv, maxv, overlap):
        """Specify evenly-spaced intervals with given overlap
        """
        rhat = float(maxv - minv) / num_intervals
        r = rhat * (1.0 + overlap / (1.0 - overlap))
        eps = r * 0.5
        
        bins = []
        for i in range(num_intervals):
            c = minv + i*rhat
            b = (max(c - eps, minv), min(c + eps, maxv))
            bins.append(b)
        
        return cls(bins)
    
    def get_bins_value_is_in(self, value):
        containing = []
        for i in range(self.num_intervals):
            a, b = self.bins[i]
            if a <= value and value <= b:
                containing.append(i)
        return containing
        
class IntervalCover1D(BaseCover):
    def __init__(self, bins):
        self.bins = bins
        
    @classmethod
    def EvenlySpaced(cls, num_intervals, minv, maxv, overlap):
        bins = _1DBins.EvenlySpaced(num_intervals, minv, maxv, overlap)
        return cls(bins)

    def get_open_set_membership(self, value):
        return self.bins.get_bins_value_is_in(value)
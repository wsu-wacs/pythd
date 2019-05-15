"""
Coverings for MAPPER.

Original code by Xiu Huan Yap <yap.4@wright.edu>
Rewritten and modified by Kyle Brown <brown.718@wright.edu>
"""
from abc import ABC, abstractmethod
import copy
import itertools
import pickle

import numpy as np

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
    
    def __repr__(self):
        return f"_1DBins({self.bins!r})"
        
    def __str__(self):
        return f"1DBins from {self.minv} to {self.maxv} with {self.num_intervals} bins"
    
    def __copy__(self):
        cls = self.__class__
        new_obj = cls(self.bins)
        new_obj.__dict__.update(self.__dict__)
        return new_obj
    
    def __deepcopy__(self, memo):
        new_bins = copy.deepcopy(self.bins, memo)
        cls = self.__class__
        new_obj = cls(new_bins)
        return new_obj
    
    @classmethod
    def EvenlySpaced(cls, num_intervals, minv, maxv, overlap):
        """Specify evenly-spaced intervals with given overlap
        
        Parameters
        ----------
        num_intervals : int
            Number of intervals to include in this dimension
        minv : float
            Left endpoint of the first (lowest) interval
        maxv : float
            Right endpoint of the last (largest) interval
        overlap : float
            Proportion of overlap of the intervals, a value between 0 and 1
        """
        if num_intervals < 1:
            raise ValueError("Must have at least one interval.")
        if minv > maxv:
            raise ValueError("Left endpoint can not be larger than right endpoint.")
        if overlap < 0.0 or overlap > 1.0:
            raise ValueError("Overlap must be a value between 0 and 1.")
        
        rhat = float(maxv - minv) / num_intervals
        r = rhat * (1.0 + overlap / (1.0 - overlap))
        eps = r * 0.5
        
        bins = []
        for i in range(num_intervals):
            c = minv + i*rhat + rhat*0.5
            b = (max(c - eps, minv), min(c + eps, maxv))
            bins.append(b)
        
        return cls(bins)
    
    def change_size(self, expand=True, amount=None, proportion=0.1, do_copy=False):
        """Expand or shrink the size of each bin by a given amount
        
        Parameters
        ----------
        expand : bool
            If True, expands bins. If False, shrinks bins
        amount : float (optional)
            The amount to change the size of each bin by. If not given,
            then the parameter proportion is used instead
        proportion : float
            Expand each bin by this proportion of its size
        do_copy : bool
            If True, make a deep copy of this _1DBins object and expand that

        Returns
        -------
        _1DBins
            The _1DBins object with expanded/contracted bins
        """
        obj = self
        if do_copy:
            obj = copy.deepcopy(self)
        
        for i, bin in enumerate(obj.bins):
            a, b = bin
            if amount is not None:
                amt = amount*0.5
            else:
                w = b - a
                amt = 0.5*proportion*w
            
            if expand:
                a -= amt
                b += amt
            else:
                a += amt
                b -= amt
                if b < a:
                    b = a
            
            obj.bins[i] = (a, b)
        
        return obj
    
    def expand(self, amount=None, proportion=0.1, do_copy=False):
        """Increase the size of the bins.
        
        For parameters and return value, see change_size()
        """
        return self.change_size(expand=True, amount=amount, proportion=proportion, do_copy=do_copy)
    
    def contract(self, amount=None, proportion=0.1, do_copy=False):
        """Shrink each of the bins.
        
        For parameters and return value, see change_size()
        """
        return self.change_size(expand=False, amount=amount, proportion=proportion, do_copy=do_copy)
    
    def get_bins_value_is_in(self, value):
        """Get the interval IDs of the intervals a point falls in.
        
        Parameters
        ----------
        value : float or int
            The value to test
        
        Returns
        -------
        list
            List of bin IDs that the point falls in
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"Value must be a numeric type, not {type(value)}. Value: {value}")

        containing = []
        for i in range(self.num_intervals):
            a, b = self.bins[i]
            # First bin continues infinitely to the left
            if i == 0 and value <= b:
                containing.append(i)
            # Last bin continues infinitely to the right
            elif (i + 1) == self.num_intervals and a <= value:
                containing.append(i)
            elif a <= value and value <= b:
                containing.append(i)

        return containing
        
class IntervalCover1D(BaseCover):
    """A one-dimensional interval cover."""
    def __init__(self, bins):
        self.bins = bins
        
    @classmethod
    def EvenlySpaced(cls, num_intervals, minv, maxv, overlap):
        bins = _1DBins.EvenlySpaced(num_intervals, minv, maxv, overlap)
        return cls(bins)
    
    @classmethod
    def EvenlySpacedFromValues(cls, f_x, num_intervals, overlap):
        minv = f_x.min()
        maxv = f_x.max()
        bins = _1DBins.EvenlySpaced(num_intervals, minv, maxv, overlap)
        return cls(bins)

    def get_open_set_membership(self, value):
        if isinstance(value, (list, np.ndarray)):
            value = value[0]
        return self.bins.get_bins_value_is_in(value)

class IntervalCover(BaseCover):
    """Hypercube cover in k-dimensions.
    
    This is a cover consisting of generalized intervals, 
    which are the cartesian product of k intervals of the
    form [a,b].
    """
    def __init__(self, bbins):
        self.bbins = bbins
    
    def __copy__(self):
        cls = self.__class__
        new_obj = cls(self.bbins)
        new_obj.__dict__.update(self.__dict__)
        return new_obj
    
    def __deepcopy__(self, memo):
        bbins = copy.deepcopy(self.bbins, memo)
        cls = self.__class__
        new_obj = cls(bbins)
        return new_obj
        
    @classmethod
    def EvenlySpaced(cls, num_intervals, minvs, maxvs, overlaps):
        """
        Construct the cover using evenly spaced intervals.
        
        Parameters
        ----------
        num_intervals : int or list
            The number of intervals to include in each dimension
        minvs : list
            Minimum values for each dimension
        maxvs : list
            Maximum values for each dimension
        overlaps : float or list
            Overlap values for each dimension
        """
        dim = len(minvs)
        
        if isinstance(num_intervals, int):
            num_intervals = [num_intervals for i in range(dim)]
        if isinstance(overlaps, (int, float)):
            overlaps = [overlaps for i in range(dim)]
        
        bbins = []
        for i in range(dim):
            bins = _1DBins.EvenlySpaced(num_intervals[i], minvs[i],
                                        maxvs[i], overlaps[i])
            bbins.append(bins)
        
        return cls(bbins)
    
    @classmethod
    def EvenlySpacedFromValues(cls, f_x, num_intervals, overlaps):
        minvs = f_x.min(axis=0)
        maxvs = f_x.max(axis=0)
        return cls.EvenlySpaced(num_intervals, minvs, maxvs, overlaps)
        
    def get_open_set_membership(self, value):
        """Get the open sets a point falls in.
        
        Parameters
        ----------
        value : numpy.ndarray
            The point (filter value)
        
        Returns
        -------
        list
            A list of hashable indices of the open sets this point falls in.
            For this cover, the indices will be tuples of integers giving the
            indices of intervals in each dimension.
        """
        bins_in = [self.bbins[i].get_bins_value_is_in(v) for i, v in enumerate(value)]
        return list(itertools.product(*bins_in))

    def change_size(self, expand=True, amounts=None, proportions=0.1, do_copy=False):
        """Change the size of the open sets
        
        Parameters
        ----------
        expand : bool
            True to increase the size, False to decrease the size
        amounts : float or list or tuple (optional)
            The amount to change size by in each dimension. If not given then
            proportions is used instead
        proportions : float or list or tuple
            The proportion to change size by in each dimension. Only used if
            amounts is not given
        do_copy : bool
            Whether to make a deep copy of this IntervalCover, or to change size
            in place
        """
        obj = self
        if do_copy:
            obj = copy.deepcopy(self)

        if not isinstance(amounts, (list, tuple)):
            amounts = [amounts for bins in self.bbins]
        if not isinstance(proportions, (list, tuple)):
            proportions = [proportions for bins in self.bbins]
        
        for i, bins in enumerate(obj.bbins):
            obj.bbins[i] = bins.change_size(expand=expand, amount=amounts[i], proportion=proportions[i], do_copy=do_copy)
        
        return obj
    
    def expand(self, amounts=None, proportions=0.1, do_copy=False):
        return self.change_size(expand=True, amounts=amounts, proportions=proportions, do_copy=do_copy)
    
    def contract(self, amounts=None, proportions=0.1, do_copy=False):
        return self.change_size(expand=False, amounts=amounts, proportions=proportions, do_copy=do_copy)

import unittest
import numpy as np
import random

from pythd.cover import _1DBins, IntervalCover

class TestBins(unittest.TestCase):
    def setUp(self):
        self.no_overlap = _1DBins.EvenlySpaced(10, 0.0, 1.0, 0.0)
        self.overlap = _1DBins.EvenlySpaced(10, 0.0, 1.0, 0.5)

    def test_bin_edges(self):
        self.assertEqual(self.no_overlap.bins[0][0], 0.0)
        self.assertEqual(self.no_overlap.bins[-1][-1], 1.0)
        self.assertEqual(self.overlap.bins[0][0], 0.0)
        self.assertEqual(self.overlap.bins[-1][-1], 1.0)
    
    def test_no_overlap(self):
        for i in (list(range(100)) + [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]):
            n = random.uniform(0.0, 1.0)
            bins = self.no_overlap.get_bins_value_is_in(n)
            
            self.assertEqual(len(bins), 1)

    def test_overlap(self):
        for i in range(100):
            n = random.uniform(0.0, 1.0)
            bins = self.overlap.get_bins_value_is_in(n)
            
            self.assertIn(len(bins), [1,2])
            
            if len(bins) == 2:
                self.assertEqual(bins[0]+1, bins[1])

class TestIntervalCover(unittest.TestCase):
    def setUp(self):
        self.data = np.random.rand(1000, 2)
        self.no_overlap = IntervalCover.EvenlySpacedFromValues(self.data, 5, 0.0)
        self.overlap = IntervalCover.EvenlySpacedFromValues(self.data, 5, 0.5)
    
    def test_no_overlap(self):
        for point in self.data:
            bins = self.no_overlap.get_open_set_membership(point)
            self.assertEqual(len(bins), 1)
    
    def test_overlap(self):
        for point in self.data:
            bins = self.overlap.get_open_set_membership(point)
            self.assertIn(len(bins), [1,2,3,4])
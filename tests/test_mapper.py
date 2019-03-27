import unittest
import numpy as np

from pythd.mapper import MAPPER
from pythd.filter import IdentityFilter
from pythd.cover import IntervalCover1D
from pythd.clustering import HierarchicalClustering

class TestMAPPER(unittest.TestCase):
    def setUp(self):
        self.x = np.random.rand(50, 1)
        minv = self.x.min()
        maxv = self.x.max()

        self.filter = IdentityFilter()
        self.cover = IntervalCover1D.EvenlySpaced(10, minv, maxv, 0.5)
        self.clustering = HierarchicalClustering()
    
    def test_mapper(self):
        mapper = MAPPER(filter=self.filter, cover=self.cover, clustering=self.clustering)
        mapper.compute_1_skeleton(self.x)
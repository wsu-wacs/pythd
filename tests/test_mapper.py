import unittest
import numpy as np
import functools

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
        self.cover = IntervalCover1D.EvenlySpaced(10, minv, maxv, 0.7)
        self.clustering = HierarchicalClustering()
    
    def test_mapper_1_skeleton(self):
        mapper = MAPPER(filter=self.filter, cover=self.cover, clustering=self.clustering)
        mapper_result = mapper.run(self.x)
        one_skeleton = mapper_result.compute_1_skeleton()
        
        self.assertEqual(len(one_skeleton), 2)
        
        # All data points should be present in the nodes
        points = [v for v in one_skeleton[0].values()]
        points = functools.reduce(lambda a,b: a|b, points)
        self.assertEqual(self.x.shape[0], len(points))
        
        # Edges should be between existing nodes
        for a,b in one_skeleton[-1]:
            self.assertIn(a, one_skeleton[0])
            self.assertIn(b, one_skeleton[0])
    
    @unittest.expectedFailure
    def test_mapper_bad_skeleton(self):
        # Should throw an error when an invalid k is given for the k-skeleton
        mapper = MAPPER(filter=self.filter, cover=self.cover, clustering=self.clustering)
        mapper_result = mapper.run(self.x)
        mapper_result.compute_k_skeleton(-1)
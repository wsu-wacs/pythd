import unittest
import numpy as np

from pythd.filter import IdentityFilter, ComponentFilter

class TestIdentityFilter(unittest.TestCase):
    def setUp(self):
        self.filter = IdentityFilter()
        self.x = np.random.rand(10, 20)
    
    def test_get_values(self):
        y = self.filter.get_values(self.x)
        
        self.assertTrue((self.x == y).all())

    def test_call(self):
        y = self.filter(self.x)
        
        self.assertTrue((self.x == y).all())

class TestComponentFilter(unittest.TestCase):
    def setUp(self):
        self.x = np.random.rand(10, 20)
        self.nrow, self.ncol = self.x.shape

    def test_one_component(self):
        for i in range(self.ncol):
            f = ComponentFilter(i)
            y = f(self.x)
            
            self.assertTrue((self.x[:, i] == y).all())

    def test_multi_component(self):
        for i in [[0,1,2], [3,4,5], [0], [0,19]]:
            f = ComponentFilter(i)
            y = f(self.x)
            
            self.assertTrue((self.x[:, i] == y).all())
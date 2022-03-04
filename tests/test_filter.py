import unittest
import numpy as np
from sklearn.manifold import TSNE

from pythd.filter import IdentityFilter, ComponentFilter, CombinedFilter, EccentricityFilter, ScikitLearnFilter

try:
    import umap
    has_umap = True
except:
    has_umap = False

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
            
            self.assertTrue((self.x[:, [i]] == y).all())

    def test_multi_component(self):
        for i in [[0,1,2], [3,4,5], [0], [0,19]]:
            f = ComponentFilter(i)
            y = f(self.x)
            
            self.assertTrue((self.x[:, i] == y).all())

class TestCombinedFilter(unittest.TestCase):
    def setUp(self):
        self.x = np.random.rand(50, 3)
        self.id = IdentityFilter()
        self.filter = CombinedFilter(ComponentFilter(0), ComponentFilter(1), ComponentFilter(2))
    
    def test_identity(self):
        id_data = self.id(self.x)
        filter_data = self.filter(self.x)
        self.assertTrue((id_data == filter_data).all())

class TestEccentricityFilter(unittest.TestCase):
    def setUp(self):
        self.x = np.random.rand(50, 2)
    
    def test_mean(self):
        filt = EccentricityFilter(method="mean")
        ecc = filt(self.x)
        
        self.assertEqual(ecc.shape[0], self.x.shape[0])
        self.assertEqual(ecc.shape[1], 1)
        
        self.assertTrue((ecc >= 0).all())
    
    def test_medoid(self):
        filt = EccentricityFilter(method="medoid")
        ecc = filt(self.x)
        
        self.assertEqual(ecc.shape[0], self.x.shape[0])
        self.assertEqual(ecc.shape[1], 1)
        
        self.assertTrue((ecc >= 0).all())
        self.assertEqual(ecc.min(), 0.0)

class TestScikitFilters(unittest.TestCase):
    def setUp(self):
        self.x = np.random.rand(50, 10)
        
    def test_tsne(self):
        filt = ScikitLearnFilter(TSNE, n_components=2, init='random', learning_rate='auto')
        f_x = filt(self.x)
        
        self.assertEqual(f_x.shape[0], self.x.shape[0])
        self.assertEqual(f_x.shape[1], 2)
    
    @unittest.skipUnless(has_umap, "requires umap")
    def test_umap(self):
        import umap
        filt = ScikitLearnFilter(umap.UMAP, n_components=2)
        f_x = filt(self.x)
        
        self.assertEqual(f_x.shape[0], self.x.shape[0])
        self.assertEqual(f_x.shape[1], 2)
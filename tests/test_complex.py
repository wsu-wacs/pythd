import os
import unittest
import itertools

from pythd.complex import SimplicialComplex

def all_simplices(simplices):
    for simplex in simplices:
        for i in range(1, len(simplex)+1):
            for seq in itertools.combinations(simplex, i):
                yield seq

class TestSimplicialComplex(unittest.TestCase):
    def setUp(self):
        self.complex = SimplicialComplex()
        self.simplices = [(1,2,3), (1,4), (2,4)]
        for simplex in self.simplices:
            self.complex.add_simplex(simplex, data=simplex)
    
    def tearDown(self):
        if os.path.exists("temp.pickle"):
            os.remove("temp.pickle")
        if os.path.exists("temp.json"):
            os.remove("temp.json")
        
    def test_is_simplex(self):
        # Verify that all the right simplices are in there
        for seq in all_simplices(self.simplices):
            self.assertTrue(self.complex.is_simplex(seq))

        # Verify that things that are not simplices are not in there
        self.assertFalse(self.complex.is_simplex((1,2,4)))
        self.assertFalse(self.complex.is_simplex((1,3,4)))
        self.assertFalse(self.complex.is_simplex((5,)))
        self.assertFalse(self.complex.is_simplex((3,4)))
    
    def test_id(self):
        for seq in all_simplices(self.simplices):
            node = self.complex.get_node(seq)
            self.assertEqual(node.id, seq[-1])
    
    def test_order(self):
        for seq in all_simplices(self.simplices):
            node = self.complex.get_node(seq)
            if node.parent is not None:
                self.assertEqual(node.get_order(), 
                                 node.parent.get_order()+1)
    
    def test_cofaces(self):
        cofaces = frozenset(self.complex.get_cofaces((1,3)))
        self.assertEqual(cofaces, frozenset([(1,3),(1,2,3)]))
        
        cofaces = frozenset(self.complex.get_cofaces((1,2)))
        self.assertEqual(cofaces, frozenset([(1,2),(1,2,3)]))
        
        cofaces = frozenset(self.complex.get_cofaces((3,)))
        self.assertEqual(cofaces, frozenset([(3,),(1,3),(2,3),(1,2,3)]))
    
    def test_save(self):
        self.complex.save_pickle(fname="temp.pickle")
        complex = SimplicialComplex.load_pickle(fname="temp.pickle")
        for k in range(1, 4):
            self.assertEqual(complex.get_k_simplices(k=k), self.complex.get_k_simplices(k=k))
        
        self.complex.save_json(fname="temp.json")
        complex = SimplicialComplex.load_json(fname="temp.json")
        for k in range(1, 4):
            self.assertEqual(complex.get_k_simplices(k=k), self.complex.get_k_simplices(k=k))

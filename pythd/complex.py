"""
Classes to represent and manipulate simplicial complexes

By Kyle Brown <brown.718@wright.edu>
"""
import itertools

class SimplicialTreeNode(object):
    """Represents a single node (simplex) in a simplicial tree"""
    def __init__(self, id, parent, data=None):
        self.id = id
        self.data = data
        self.parent = parent
        self.children = {}
    
    def has_child(self, id):
        return id in self.children
    
    def get_child(self, id):
        return self.children[id]
    
    def add_child(self, node):
        self.children[node.id] = node
    
    def get_parent(self):
        return self.parent

class SimplicialComplex(object):
    """A simplicial complex, stored as a simplex tree."""
    def __init__(self, simplices=None):
        self.simplex_tree = SimplicialTreeNode(-1, None, None)
        self.cousins = {}
    
    def _add_simplex_base(self, seq, data=None):
        """Single step to insert a single simplex into the tree"""
        id = seq[-1]
        node = self.simplex_tree
        depth = 0
        for i in seq:
            if node.has_child(i):
                child_node = node.get_child(i)
            else:
                child_node = SimplicialTreeNode(i, node, None)
                node.add_child(child_node)
                
                if depth > 0:
                    cousin_idx = (depth, i)
                    if cousin_idx not in self.cousins:
                        self.cousins[cousin_idx] = [child_node]
                    else:
                        self.cousins[cousin_idx].append(child_node)

            node = child_node
            depth += 1
        node.data = data
        
    def add_simplex(self, simplex, data=None):
        """Add a single simplex to the simplex tree.
        
        This method will insert a single simplex and all of its
        faces into the simplex tree.

        Parameters
        ----------
        simplex: iterable
            The node ids making up the simplex to add
        data
            Arbitrary data to associate with the simplex
        """
        simplex = sorted(simplex)
        k = len(simplex)
        # add all 0-simplex faces, then 1-simplex faces, up to k
        for i in range(1, k+1):
            for seq in itertools.combinations(simplex, i):
                if i == k:
                    self._add_simplex_base(seq, data=data)
                else:
                    self._add_simplex_base(seq, data=None)
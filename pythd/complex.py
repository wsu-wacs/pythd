"""
Classes to represent and manipulate simplicial complexes

By Kyle Brown <brown.718@wright.edu>
"""
import itertools

class SimplicialTreeNode(object):
    """Represents a single node (simplex) in a simplicial tree"""
    def __init__(self, id, parent, data=None, order=0, **kwargs):
        self.id = id
        self.data = data
        self.parent = parent
        self.children = {}
        self.dict = {}
        self.dict.update(kwargs)
        if parent is not None:
            self.order = parent.order + 1
        else:
            self.order = order
    
    def has_child(self, id):
        return id in self.children
    
    def get_child(self, id):
        return self.children[id]
    
    def get_children(self):
        return self.children
    
    def add_child(self, node):
        self.children[node.id] = node
    
    def get_data(self):
        return self.data
    
    def get_order(self):
        return self.order
    
    def get_parent(self):
        return self.parent
    
    def get_simplex(self):
        """Get the sequence of nodes making up the simplex corresponding to the node"""
        simplex = (self.id,)
        if self.parent is not None and self.parent.id != -1:
            simplex = self.parent.get_simplex() + simplex
        return simplex
    
    def __getitem__(self, key):
        return self.dict[key]
    
    def __setitem__(self, key, value):
        self.dict[key] = value

class SimplicialComplex(object):
    """A simplicial complex, stored as a simplex tree."""
    def __init__(self, simplices=None):
        self.simplex_tree = SimplicialTreeNode(-1, None, None, order=-1)
        self.cousins = {}
    
    def _add_simplex_base(self, seq, data=None, **kwargs):
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
        if data is not None: # prevent overwriting data with None
            node.data = data
        node.dict.update(kwargs)
    
    def _get_k_simplices_base(self, node, k=0, include_data=False):
        """Recursive helper function for get_k_simplices()"""
        k_simplices = []
        if node.get_order() == k: # base case: we're at the right depth
            if include_data:
                k_simplices.append((node.get_simplex(), node.get_data(), node.dict))
            else:
                k_simplices.append(node.get_simplex())
        else: # otherwise travel down the tree
            for simplex, child in node.get_children().items():
                k_simplices += self._get_k_simplices_base(child, k=k, 
                                                          include_data=include_data)
        return k_simplices
        
    def get_k_simplices(self, k=0, include_data=False):
        """Get k-simplices in the complex
        
        Parameters
        ----------
        k : int
            The order of the simplices to retrieve
        include_data : bool
            Whether to include the optional data associated to each simplex"""
        return self._get_k_simplices_base(self.simplex_tree, k=k, include_data=include_data)
        
    def add_simplex(self, simplex, data=None, **kwargs):
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
        # add all 0-simplex faces, then 1-simplex faces, and so on up to k
        for i in range(1, k):
            for seq in itertools.combinations(simplex, i):
                self._add_simplex_base(seq, data=None)
        self._add_simplex_base(simplex, data=data, **kwargs)
    
    def is_simplex(self, simplex):
        """Check if a simplex is in this complex.
        
        Parameters
        ----------
        simplex: iterable
            A sequence of integers defining the simplex
        
        Returns
        -------
        bool
            True if the simplex is in the complex
            False otherwise
        """
        simplex = sorted(simplex)
        node = self.simplex_tree
        for i in simplex:
            if i not in node.children:
                return False
            node = node.children[i]
        return True

    def get_node(self, simplex):
        """Get a SimplicialTreeNode from the complex.
        
        Parameters
        ----------
        simplex : iterable
           
        Returns
        -------
        SimplicialTreeNode
            
        """
        simplex = sorted(simplex)
        node = self.simplex_tree
        for i in simplex:
            if i not in node.children:
                raise KeyError(f"Simplex {simplex} not in the complex")
            node = node.children[i]
        return node
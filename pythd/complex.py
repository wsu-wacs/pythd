"""
Classes to represent and manipulate simplicial complexes

By Kyle Brown <brown.718@wright.edu>
"""
import itertools

class SimplicialTreeNode(object):
    """
    Represents a single node (simplex) in a simplicial tree
    
    Attributes
    ----------
    id : int
        The integer id of the 0-simplex this node represents
    parent : SimplicialTreeNode
        The parent node of this one. Used to quickly move up the tree
    data : object
        An arbitrary object associated with this node
    order : int
        The order of the simplex this node represents
    **kwargs
        Any other data to associate with this node, stored as a dictionary.
    """
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
        """Return True if the node has a child with the given 0-simplex"""
        return id in self.children
    
    def get_child(self, id):
        """Get the child of this node with the given 0-simplex"""
        return self.children[id]
    
    def get_children(self):
        """Get the children dict of this node"""
        return self.children
    
    def add_child(self, node):
        """Add a node to this node as a child"""
        self.children[node.id] = node
    
    def get_data(self):
        """Get the data associated to this node"""
        return self.data
    
    def get_order(self):
        """Get the order (dimension) of the simplex represented by this node"""
        return self.order
    
    def get_parent(self):
        """Get the parent of this node"""
        return self.parent
    
    def get_simplex(self):
        """
        Get the sequence of nodes making up the simplex corresponding to the node
        
        Returns
        -------
        tuple
            A sequence of integers consisting of the 0-simplices of this node
        """
        simplex = (self.id,)
        if self.parent is not None and self.parent.id != -1:
            simplex = self.parent.get_simplex() + simplex
        return simplex
    
    def __getitem__(self, key):
        """Get some data associated to this node."""
        return self.dict[key]
    
    def __setitem__(self, key, value):
        """Associate some data with this node."""
        self.dict[key] = value

class SimplicialComplex(object):
    """
    A simplicial complex, stored as a simplex tree.
    
    A simplex tree is a kind of trie used to store an abstract simplicial
    complex. Nodes of the tree represent faces of the simplicial complex,
    with the depth of the node being the same as the dimension of the
    simplex it corresponds to.
    
    Attributes
    ----------
    simplex_tree : SimplicialTreeNode
        The root node of the simplex tree, whose children are the 0-simplices
    max_order : int
        The dimension of the largest simplex in the tree.
    cousins : dict
        An auxiliary structure used to keep track of identical 0-simplices 
        at each depth in the tree. This speeds up some operations.
    
    References
    ----------
    [1] Boissonnat, Jean-Daniel, and Clément Maria. "The simplex tree: 
        An efficient data structure for general simplicial complexes." In 
        European Symposium on Algorithms, pp. 731-742. Springer, Berlin, 
        Heidelberg, 2012.
    """
    def __init__(self):
        self.simplex_tree = SimplicialTreeNode(-1, None, None, order=-1)
        self.max_order = -1
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
        """Get k-simplices in the complex.
        
        Parameters
        ----------
        k : int
            The order of the simplices to retrieve
        include_data : bool
            Whether to include the optional data (and dict) associated to each simplex
        
        Returns
        -------
        list
            A list of the k-simplices in the tree. If include_data is False, each
            element of the list is a tuple giving the 0-simplices making up the simplex.
            If include_data is True, each element is a tuple of the form (simplex, data, dict)
            where simplex is the tuple of 0-simplices, data is optional data associated with
            the simplex, and dict is the dictionary associated with the simplex.
        """
        return self._get_k_simplices_base(self.simplex_tree, k=k, include_data=include_data)
        
    def add_simplex(self, simplex, data=None, **kwargs):
        """Add a single simplex to the simplex tree.
        
        This method will insert a single simplex and all of its
        faces into the simplex tree. Optionally, one can associate some
        data with the simplex.

        Parameters
        ----------
        simplex: iterable
            The node ids making up the simplex to add
        data
            Arbitrary data to associate with the simplex
        **kwargs
            Named data to associate with the simplex, as in a dictionary
        """
        simplex = sorted(simplex)
        k = len(simplex)
        self.max_order = max(self.max_order, k)
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
            A sequence of integers defining the simplex by its 0-simplices
        
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
            A sequence of integers defining the simplex by its 0-simplices
           
        Returns
        -------
        SimplicialTreeNode
            The node in the tree. If the node is not present, raises a KeyError
        """
        simplex = sorted(simplex)
        node = self.simplex_tree
        for i in simplex:
            if i not in node.children:
                raise KeyError(f"Simplex {simplex} not in the complex")
            node = node.children[i]
        return node
    
    def _get_node_children_simplices(self, node, simplex=()):
        s = [simplex]
        for n in node.get_children().values():
            s = s + self._get_node_children_simplices(n, simplex + (n.id,))
        return s
    
    def get_cofaces(self, simplex):
        """Get the cofaces of a simplex
        
        This will get the cofaces of a given simplex. The cofaces are
        all simplices in the complex that the simplex is a face of.
        
        Parameters
        ----------
        simplex : iterable
    
        Returns
        -------
        list
        """
        simplex = sorted(simplex)
        id = simplex[-1]
        depth = len(simplex)-1
        cofaces = []
        
        for i in range(depth, self.max_order + 1):
            idx = (i,id)
            if idx in self.cousins:
                for n in self.cousins[idx]:
                    n_simplex = n.get_simplex()
                    
                    if frozenset(simplex) <= frozenset(n_simplex):
                        cofaces = cofaces + self._get_node_children_simplices(n, n_simplex)
        return cofaces
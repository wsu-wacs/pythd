import copy
from itertools import combinations
import json
import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist
import pandas as pd

from pythd.mapper import MAPPER
from pythd.clustering import HierarchicalClustering

class THD:
    """
    Class for running a THD
    
    Parameters
    ----------
    dataset : numpy.ndarray or pandas.DataFrame
        The base dataset to run the THD on
    filt : pythd.filter.BaseFilter
        The filter function to use
    cover : pythd.cover.BaseCover
        The covering to use
    clustering : pythd.clustering.BaseClustering
        The clustering algorithm to use
    group_threshold : int
        The group threshold to use. Any connected components with
        less data points than this will not be used
    full_df : numpy.ndarray or pandas.DataFrame (optional)
        The full dataset used to build the THD.
        This can be given to allow coloring by columns not used in building the THD
    """
    def __init__(self, dataset, filt, cover,
                 clustering=HierarchicalClustering(),
                 contract_amount=0.1,
                 group_threshold=100,
                 full_df=None):
        self.dataset = pd.DataFrame(dataset)
        if full_df is not None:
            self.full_df = pd.DataFrame(full_df)
        else:
            self.full_df = self.dataset
        self.filter = filt
        self.base_cover = copy.deepcopy(cover)
        self.clustering = clustering
        self.contract_amount = contract_amount
        self.group_threshold = group_threshold
        self.reset()
        
    def reset(self):
        """Reset the THD to be able to run it from the start."""
        self.cover = copy.deepcopy(self.base_cover)
        self.root = THDJob(self.dataset, self.filter, self.cover,
                     rids=list(range(self.dataset.shape[0])),
                     clustering=self.clustering,
                     contract_amount=self.contract_amount,
                     group_threshold=self.group_threshold,
                     full_df=self.full_df)
        self.jobs = [self.root]
        self.is_run = False

    def run(self, verbose=False):
        """Run the THD.
        
        Returns
        -------
        pythd.thd.THDGroup
            The root group of the completed THD.
        """
        def vprint(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)

        while self.jobs:
            vprint(len(self.jobs), "jobs remaining")
            job = self.jobs.pop()
            job.run(verbose=True)
            vprint("Group {} finished ({} rids)".format(job.group.get_name(), len(job.group.rids)))
            self.jobs += job.child_jobs
        self.is_run = True
        return self.get_results()
    
    def get_results(self):
        """Get the root group of the THD.
        
        Returns
        -------
        pythd.thd.THDGroup
            The root group of the completed THD.
        """
        return self.root.group
    
    def get_dict(self):
        """Get a dictionary representation of the THD configuration and results.
        
        The dictionary is suitable for JSON serialization."""
        o =  {
            "data_shape": tuple(map(int, self.dataset.shape)),
            "filter_type": type(self.filter).__name__,
            "cover": self.base_cover.get_dict(),
            "clustering": self.clustering.get_dict(),
            "group_threshold": self.group_threshold
        }
        if self.is_run:
            o["groups"] = self.root.group.get_all_dicts()
        return o
    
    def save_json(self, fname, **kwargs):
        """Save the THD configuration and results to a JSON file.
        
        Parameters
        ----------
        fname : str
            The name of the file to save to.
        
        **kwargs
            Keyword arguments passed to the json.dump method.
        """
        with open(fname, "w") as f:
            d = self.get_dict()
            json.dump(d, f, **kwargs)

class THDJob:
    """
    THD code for running on one group

    Parameters
    ----------
    dataset : numpy.ndarray or pandas.DataFrame
        The whole dataset to run on
    filt : pythd.filter.BaseFilter
        The filter function to use
    cover : pythd.cover.BaseCover
        The covering to use
    rids : list
        Row indices for this job
    clustering : pythd.clustering.BaseClustering
        The clustering algorithm to use
    group_threshold : int
        The minimum size of a child group to be further decomposed
    contract_amount : float
        Amount to shrink the cover by if a split isn't observed
    parent : THDGroup
        The parent group of this group. None if this is the root node
    """
    def __init__(self, dataset, filt, cover, rids, 
                 clustering=HierarchicalClustering(),
                 group_threshold=100,
                 contract_amount=0.1,
                 parent=None,
                 full_df=None):
        self.dataset = dataset
        if full_df is not None:
            self.full_df = full_df
        else:
            self.full_df = self.dataset
        self.filt = filt
        self.cover = copy.deepcopy(cover)
        self.rids = rids
        self.subset = self.dataset.iloc[self.rids, :]
        self.clustering = clustering
        self.group_threshold = group_threshold
        self.contract_amount = contract_amount
        self.parent = parent
        self.reset()
    
    def reset(self):
        self.mapper = MAPPER(filter=self.filt, cover=self.cover, clustering=self.clustering)
        self.child_jobs = []
        self.components = []
        self.groups = []
        self.job_groups = []
        self.is_run = False
    
    def run(self, verbose=False):
        if self.is_run:
            self.reset()
        # MAPPER -> get topological network and connected components
        self.result = self.mapper.run(self.subset.values, rids=self.rids)
        self.network = self.result.compute_k_skeleton(k=1)
        self.components = self.network.get_connected_components()
        
        self.group = THDGroup(self.dataset, self.rids, self.network, full_df=self.full_df)
        if self.parent:
            self.parent.add_child(self.group)
        # row IDs for each group
        for nids in self.components:
            rids = set()
            for nid in nids:
                node = self.network.get_node((nid,))
                rids = rids | node['points_orig']
            group = list(rids)
            self.groups.append(group)
            if len(group) > self.group_threshold:
                self.job_groups.append(group)

        # Determine child groups to decompose on
        if len(self.job_groups) > 0:
            if len(self.job_groups) == 1:
                new_cover = self.cover.contract(proportions=self.contract_amount, do_copy=True)
            else:
                new_cover = self.cover
            
            for group in self.job_groups:
                    job = THDJob(self.dataset, self.filt, new_cover, rids=group,
                                 clustering=self.clustering,
                                 group_threshold=self.group_threshold,
                                 contract_amount=self.contract_amount,
                                 parent=self.group,
                                 full_df=self.full_df)
                    self.child_jobs.append(job)
        self.is_run = True

class THDGroup:
    """
    Class representing a THD group in a completed THD
    
    Parameters
    ----------
    dataset : numpy.ndarray or pandas.DataFrame
        The entire dataset that the THD was run on
    rids : sequence
        The row ids of the group
    network : pythd.complex.SimplicialComplex
        The simplicial complex produced by MAPPER on the group
    """
    def __init__(self, dataset, rids, network, full_df=None):
        self.dataset = dataset
        if full_df is not None:
            self.full_df = full_df
        else:
            self.full_df = self.dataset
        self.rid_list = list(map(int, rids))
        self.rids = set(self.rid_list)
        self.num_rows = len(self.rids)
        self.network = network
        self.children = []
        self.depth = 0
        self.id = 0
        self.parent_id = 0
        self.dcs = {0: 1}
        self.parent = None
        # Values used for coloring
        self.density = len(self.rids) / self.dataset.shape[0]
        self.value = self.density
        self.network_size = len(self.network.get_k_simplices())
        # Cluster distance
        self.computed_distance = False
        self.distance_method = None
        self.cluster_method = None
        self.metric = None
        self.dist = 0.0

    def __iter__(self):
        """Iterator for this group and all its descendents"""
        self.iter_stack = [self]
        return self
    
    def __next__(self):
        if self.iter_stack:
            res = self.iter_stack.pop()
            self.iter_stack = res.children + self.iter_stack
            return res
        else:
            raise StopIteration
    
    def __str__(self):
        return "Group {} ({} rows, {} children)".format(self.get_name(), self.num_rows, len(self.children))
    
    def compute_distance(self, 
                         combine_method="average", 
                         cluster_method="average", 
                         metric="euclidean",
                         **kwargs):
        """
        Compute distance between children clusters.
        """
        if (self.computed_distance and 
            (self.distance_method == combine_method) and 
            (self.cluster_method == cluster_method) and
            (self.metric == metric)):
            return self.dist
            
        self.computed_distance = True
        self.distance_method = combine_method
        self.cluster_method = cluster_method
        self.metric = metric
        
        if len(self.children) <= 1:
            self.dist = 0.0
            return self.dist

        dists = []
        for g1, g2 in combinations(self.children, 2):
            X1 = g1.get_data(False)
            X2 = g2.get_data(False)
            Y = cdist(X1, X2, metric=metric, **kwargs) # pairwise distances between groups 

            if cluster_method == 'average':
                dist = Y.mean()
            elif cluster_method in ['single', 'min']:
                dist = Y.min()
            elif cluster_method in ['complete', 'max']:
                dist = Y.max()
            # distance between just these two clusters
            dists.append(dist)
        
        # combine individual cluster distances into one
        if combine_method == 'average':
            self.dist = sum(dists) / len(dists)
        elif combine_method in ['single', 'min']:
            self.dist = min(dists)
        elif combine_method in ['complete', 'max']:
            self.dist = max(dists)
        
        return self.dist

    def _normalize_values(self):
        minval = 1.0
        maxval = 0.0
        for group in self:
            maxval = max(maxval, group.value)
            minval = min(minval, group.value)
        for group in self:
            group.value = (group.value - minval) / (maxval - minval)

    def get_group_by_name(self, name):
        for group in self:
            if group.get_name() == name:
                return group
        return None

    def get_name(self):
        return "{}.{}.{}".format(self.depth, self.id, self.parent_id)
    
    def get_data(self, full_data=True):
        if full_data:
            df = self.full_df
        else:
            df = self.dataset
        return df.iloc[self.rid_list, :]

    def get_stats(self):
        df = self.get_data()
        means = df.mean(axis=0)
        mins = df.min(axis=0)
        maxs = df.max(axis=0)
        return (means, mins, maxs)

    def add_child(self, child):
        child.depth = self.depth + 1
        child.parent = self
        child.parent_id = self.id
        child.dcs = self.dcs
        child.id = self.dcs.get(child.depth, 0)
        self.dcs[child.depth] = child.id + 1
        
        self.children.append(child)
    
    def as_igraph_graph(self, include_column_summary=False):
        import igraph
        pal = igraph.drawing.colors.AdvancedGradientPalette(["blue", "orange", "green", "red"], n=128)
        
        g = igraph.Graph()
        for group in self:
            cd = {}
            if include_column_summary:
                means, mins, maxs = group.get_stats()
                cd = {c: (mins.loc[c], means.loc[c], maxs.loc[c]) for c in means.index}

            name = group.get_name()
            g.add_vertex(name=name,
                         id=name,
                         num_rows=group.num_rows,
                         color=pal.get(int(round(group.value*127.0))),
                         **cd)

        for group in self:
            for child in group.children:
                g.add_edge(group.get_name(), child.get_name())
        return g

    def as_networkx_network(self):
        import networkx as nx

        g = nx.Graph()
        for group in self:
            name = group.get_name()
            g.add_node(name, id=name, name=name,
                       num_rows=self.num_rows)

        for group in self:
            for child in group.children:
                g.add_edge(group.get_name(), child.get_name())
        return g
    
    def color_density(self, normalize=False):
        for group in self:
            group.value = len(group.rids) / group.dataset.shape[0]
        if normalize:
            self._normalize_values()

    def color_by_rids(self, rids, normalize=False):
        rids = set(rids)
        for group in self:
            group.value = len(rids & group.rids) / len(group.rids)
        if normalize:
            self._normalize_values()
    
    def color_by_value(self, values, normalize=False):
        for group in self:
            gvs = [values[rid] for rid in group.rids]
            group.value = sum(gvs) / len(gvs)
        if normalize:
            self._normalize_values()

    def color_network_size(self):
        for group in self:
            group.value = group.network_size
        self._normalize_values()
    
    def get_dict(self, include_network=True, **kwargs):
        d = {
            "name": self.get_name(),
            "num_rows": len(self.rids),
            "rids": list(self.rids),
            "parent": self.parent.get_name() if self.parent else None,
            "children": [child.get_name() for child in self.children],
            "density": self.density,
            "num_nodes": self.network_size,
            "color": self.value,
            "data_shape": list(self.dataset.shape),
            "full_data_shape": list(self.full_df.shape)
        }
        
        if include_network:
            d["network"] = self.network.get_dict()
        
        return d
    
    def get_all_dicts(self, **kwargs):
        d = {}
        return {g.get_name(): g.get_dict(**kwargs) for g in self}

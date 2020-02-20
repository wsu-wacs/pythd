import copy
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
    """
    def __init__(self, dataset, filt, cover,
                 clustering=HierarchicalClustering(),
                 group_threshold=100):
        self.dataset = pd.DataFrame(dataset)
        self.filter = filt
        self.cover = cover
        self.clustering = clustering
        self.group_threshold = group_threshold
        self.reset()
        
    def reset(self):
        job = THDJob(self.dataset, self.filter, self.cover,
                     rids=list(self.dataset.index.values),
                     clustering=self.clustering,
                     group_threshold=self.group_threshold)
        self.jobs = [job]
        self.finished = []

    def run(self):
        while self.jobs:
            print(self.jobs)
            job = self.jobs.pop()
            job.run()
            self.jobs += job.child_jobs

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
    """
    def __init__(self, dataset, filt, cover, rids, 
                 clustering=HierarchicalClustering(),
                 group_threshold=100,
                 contract_amount=0.1):
        self.dataset = dataset
        self.filt = filt
        self.cover = copy.deepcopy(cover)
        self.rids = rids
        self.subset = self.dataset.loc[self.rids, :]
        self.clustering = clustering
        self.group_threshold = group_threshold
        self.contract_amount = contract_amount
        self.reset()
    
    def reset(self):
        self.mapper = MAPPER(filter=self.filt, cover=self.cover, clustering=self.clustering)
        self.child_jobs = []
        self.components = []
        self.groups = []
        self.job_groups = []
        self.is_run = False
    
    def run(self):
        if self.is_run:
            self.reset()
        # MAPPER -> get topological network and connected components
        self.result = self.mapper.run(self.subset.values, rids=list(self.subset.index.values))
        self.network = self.result.compute_k_skeleton(k=1)
        self.components = self.network.get_connected_components()
        
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
                             contract_amount=self.contract_amount)
                self.child_jobs.append(job)
        self.is_run = True
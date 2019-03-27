"""
Pure Python implementation of MAPPER

Original code by Xiu Huan Yap <yap.4@wright.edu>
Rewritten and modified by Kyle Brown <brown.718@wright.edu>
"""
class MAPPER:
    def __init__(self, filter=None, cover=None):
        self.set_filter(filter)
        self.set_cover(cover)
    
    def set_filter(self, filter):
        self.filter = filter
        return self
    
    def set_cover(self, cover):
        self.cover = cover
        return self
    
    def compute_1_skeleton(self, points):
        """
            Compute the MAPPER structure as a topological network
        """
        f_x = self.filter(points) # filter values
        d = self.cover.get_open_set_membership_dict(f_x)
        
        for set_id, members in d.items():
            if len(members) == 1:
                clusters = [[0]] # Just the one point 
            else:
                pass # TODO: clustering
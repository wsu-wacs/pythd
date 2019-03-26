#!/usr/bin/env python
#Segmenter classes: 1) Connected_Component_Segmenter()
from typing import Union, Dict, Type, List, Any
from scipy.sparse import coo_matrix
import numpy as np

class Connected_Component_Segmenter():
    """
    Function get_segments is called by class THD

    :param network_threshold: Minimum number of points for a valid network  
    """
    def __init__(self,network_threshold):
        # type: (int) -> None
        self.network_threshold=network_threshold
        
    def _get_connected_components(self,mapper_result):
        nodes=mapper_result['nodes']
        edges=mapper_result['edges']
        #Get edges in upper triangular matrix format
        edge_matrix=coo_matrix((len(nodes),len(nodes)),dtype=np.int8).toarray()
        if edges:
            for key in edges:
                edge_matrix[key]=1
        edge_matrix=edge_matrix+np.transpose(edge_matrix)
        connected_components=[]
        counted_nodes=[]
        idx=0
        while len(counted_nodes)<len(nodes):
            #Get all nodes that are in the same connected component as idx
            node_list=[idx]
            row=edge_matrix[idx]
            row[idx]=1
            connected_nodes=list(np.where(row>0)[0])
            while sorted(node_list)!=sorted(connected_nodes):
                new_nodes=[connected_nodes[i] for i in range(len(connected_nodes)) if (connected_nodes[i] not in node_list)]
                node_list.extend(new_nodes)
                for i in new_nodes:
                    row=row+edge_matrix[i]
                    connected_nodes=list(np.where(row!=0)[0])
            connected_component={}
            connected_component['nodes']=node_list
            counted_nodes.extend(node_list)
            connected_components.append(connected_component)
            #Get next idx
            idx=idx+1
            while (idx in counted_nodes):
                idx=idx+1
        #Get points for each connected component
        for connected_component in connected_components:
            point_list=[]
            for node in connected_component['nodes']:
		try:
                    point_list.extend(nodes[node]['points']) #node saved as dictionary to reduce dependency on mapper?
		except:
		    point_list.extend(nodes[node])
            point_list=np.unique(point_list)
            connected_component['points']=point_list 
        return connected_components #NB. point idx are local to mapper, not to global training data

    def _get_valid_connected_components(self,connected_components):
        network_count =0
        network_list=[]
#        group_count=0
#        group_list=[]
        for idx, connected_component in enumerate(connected_components):
            if len(connected_component['points'])>self.network_threshold:
                network_count=network_count+1
                network_list.append({'idx':idx,'points_list':connected_component['points']})
#            if len(connected_component['points'])>self.group_threshold:
#                group_count=group_count+1
#                group_list.append(idx)
#        for group_idx in group_list:
#            if group_idx in network_list:
#                group_list.remove(group_idx)
        return network_count, network_list#,group_count,group_list

    def get_segments(self,mapper_result):
        # type: (Dict) -> List
        connected_components=self._get_connected_components(mapper_result)
        network_count, network_list=self._get_valid_connected_components(connected_components)
        return network_list
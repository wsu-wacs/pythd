#!/usr/bin/env python
"""
Given filter values and gain, this package calculates the lowest resolution that ensures at least 2 connected components.
    -filt <list>: List of lists of filter values. Should have length of 2 (Currently only 2-D filters supported) 
    -gain <float>: Proportion of overlap [0,1] in the covering. Can also be defined Ayasdi-style (1.0,10]. 
    
Classes:
1) connectivity_check: Main class to instantiate. 
    -Call "get_n_gap(n)" to get resolution where criteria (at least 2 connected components) is met n times 
        (i.e. if n=3 and resolution is 5, then resolution=5,6 or 7 all meet the criteria)
        If desired: Set "points_threshold" to m for connected components to contain at least m points
        (Good if there are many outliers). Note that the connected components may separate into multiple clusters in mapper.   
2) Bins <Helper class>: Constructs 1D covering based on given parameters
3) Covering <Helper class>: Constructs covering based on given parameters. 
        Mainly converts information from 'bin_edges per filter interval' style to 'filter intervals per bin spacing' style
2) visualize_filt: Visualizes points in the covering space

General Workflow:
1) Resolution Scanning: Starting from resolution=3 and incrementing by 1, we build the covering and check for 'connectivity' in the filter space. 
2) Graph Embedding: Each filter interval represents a vertex while overlaps represent edges.
        If no points are present in the filter interval/overlap, then the vertex/edge is not represented.
        #Vertices indicate that at least 1 point is present in the interval.
        #Edges indicate that at least 1 point is present in the overlap. 
        #Higher-order overlaps give rise to higher-order simplicies
3) Graph Traversal: Breadth-first search to check for number of connected components.

Written by Xiu-Huan Yap, Wright State University
Contact:yapxiuhuan@gmail.com
Date: June 04, 2018.
"""
import copy
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import time

class Bins: #For 1-D
        def __init__(self,intervals,gain,filt_min,filt_max):
            assert filt_max>filt_min, "Value for filter max should be greater than filter min"
            self.intervals=intervals
            self.filt_min=filt_min
            self.filt_max=filt_max
            self.filt_length=float(self.filt_max)-self.filt_min
            if gain>1: #Ayasdi's gain formula: %Overlap=1-1/gain for 1<gain<10
                self.gain=1-(1./gain)
            else:
                self.gain=gain
            self.bin_edges=[] 
            self.overlap_edges=[]
    
        def _get_intervals(self):
            interval_length=self.filt_length/(self.intervals-(self.intervals-1)*self.gain)
            step_size=interval_length*(1-self.gain)
            #nogain_edges=[self.filt_min+(i*nogain_interval_length) for i in range(1,self.intervals)]
            return interval_length,step_size
    
        def get_bin_edges(self):
            interval_length,step_size=self._get_intervals()
            bin_edges=[]
            overlap_edges=[]
            for i in range(self.intervals):
                bin_edges.append(list(np.add([self.filt_min,self.filt_min+interval_length],i*step_size)))
                if i>0:
                    overlap_edges.append([bin_edges[i][0],bin_edges[i-1][1]])  
           #Update self
            self.bin_edges=bin_edges
            self.overlap_edges=overlap_edges
            return bin_edges,overlap_edges
class Covering: #TBD: 2D for now
    def __init__(self,all_bin_edges):
        self.all_bin_edges=all_bin_edges #3D list
        self.dim=len(all_bin_edges)
        self.intervals=[len(bin_edges) for bin_edges in all_bin_edges]
        #self.max_interval_list=[len(all_bin_edges[i]) for i in range(self.dim)] #Gets the number of intervals in each dimension
        self.all_spacings,self.all_intervals_in_spacings=self._get_all_spacings()
        
    def _get_spacings_in_1D(self,bin_edges_in_1D):
        # Divides 1D filter space into (unequal) bins and notes the component intervals that contribute to that bin 
        # Higher order overlaps will happen only if gain>50%
        # Max of res*2 (unequal) bins aka spacings (spacings defined by min value/left edge)
        intervals=len(bin_edges_in_1D)
        all_edges=sorted(set([i for sublist in bin_edges_in_1D for i in sublist]))[:-1]
        spacings=all_edges
        intervals_in_spacings=[]
        for edge in all_edges:
            intervals_in_spacing=[]
            for j in range(intervals):
                #If min<=edge<max of interval j, then that spacing is contained by interval j
                if bin_edges_in_1D[j][0]<=edge and bin_edges_in_1D[j][1]>edge:
                    intervals_in_spacing.append(j)
            intervals_in_spacings.append(intervals_in_spacing)
        return spacings,intervals_in_spacings
    
    def _get_all_spacings(self):
        each_spacings=[]
        each_intervals_in_spacings=[]
        for i in range(self.dim):
            bin_edges_in_1D=self.all_bin_edges[i]
            spacings,intervals_in_spacings=self._get_spacings_in_1D(bin_edges_in_1D)
            each_spacings.append(spacings)
            each_intervals_in_spacings.append(intervals_in_spacings)
        all_spacings=[]
        all_intervals_in_spacings=[]
        #Total number of spacings = multiply no. of spacings in each dimension
        #indexes=[0] *self.dim
        #max_indexes=[len(all_spacings)[i] for i in range(len(all_spacings))]
        #while indexes ~= max_indexes:
        
        ##Main loop; Gets all spacings and intervals in higher dimension
        ##TBD: Hard code to 2D for now.
        max_x=self.all_bin_edges[0][-1][-1]
        max_y=self.all_bin_edges[1][-1][-1]
        if self.dim==2:
            x_intervals=self.intervals[0]
            y_intervals=self.intervals[1]
            x_spacings=len(each_spacings[0])
            y_spacings=len(each_spacings[1])
            for i2 in range(y_spacings):
                for i1 in range(x_spacings):
                    #Get bin edges
                    if i1==x_spacings-1 and i2<y_spacings-1:
                        min_edge=[each_spacings[0][i1],each_spacings[1][i2]]
                        max_edge=[max_x,each_spacings[1][i2+1]]
                    elif i2==y_spacings-1 and i1<x_spacings-1:
                        min_edge=[each_spacings[0][i1],each_spacings[1][i2]]
                        max_edge=[each_spacings[0][i1+1],max_y]
                    elif i1==x_spacings-1 and i2==y_spacings-1:
                        min_edge=[each_spacings[0][i1],each_spacings[1][i2]]
                        max_edge=[max_x,max_y]
                    else:
                        min_edge=[each_spacings[0][i1],each_spacings[1][i2]]
                        max_edge=[each_spacings[0][i1+1],each_spacings[1][i2+1]]
                    all_spacings.append([min_edge,max_edge])
                    
                    #Get filter intervals (Numbering starts from zero)
                    x_filter_intervals=each_intervals_in_spacings[0][i1]
                    y_filter_intervals=each_intervals_in_spacings[1][i2]
                    intervals=list(itertools.product(x_filter_intervals,list(np.array(y_filter_intervals)*x_intervals)))
                    intervals=sorted([interval[0]+interval[1] for interval in intervals])
                    all_intervals_in_spacings.append(intervals)
        else: 
            assert self.dim<3, "All_spacings not yet configured for 3D and higher filters"
        return all_spacings,all_intervals_in_spacings       
                

class connectivity_check(object):
    def __init__(self,network=None,params=None,filt=None,gain=None,delete_interim_network_and_coloring=True): 
        #Either define (filt,gain), 
            #or define (Ayasdi) network, 
            #or define params, a dictionary with keys: {(Ayasdi) group,network} 
                    ##NB. Use params if the check is done on a subset of points specified by group
        #Filt is a list of list of filter values
        #Set delete_interim_network_and_coloring as False if you want to keep the group, network and colorings created on Ayasdi 
        if type(filt) != type(None):
            assert gain != None, "The parameter /'gain/' is required if /'filt/' is defined."
            filt=self._arrange_filt(filt) 
            self.gain=gain #From class bin_edges
            
        elif network !=None or params!=None:
            #Get group info from network
            if network!=None:
                node_dict=network.get_points(range(len(network.nodes)))
                points=[]
                for node in node_dict:
                    points_in_node=node_dict[node]
                    points.extend([point for point in points_in_node])
                points=list(np.unique(points))
                new_group=None
        
            if params!=None:
                network=params['network']
                new_group=params['group']
            #Get filt and gain from source and network information
            
            source=network.source
            new_lenses=copy.deepcopy(network.lenses)
            self.gain=network.lenses[0]['gain'] #Assume that gain is the same for all the lenses used                
            net_name=network.name   
            #Create new network that has points in its own nodes
            struct_time=time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime())  
            number=random.randint(0,99)
            new_net_name=net_name+"_"+str(struct_time+"_"+str(number))
            new_group_created=False
            if new_group==None:
                new_group=source.create_group(name=new_net_name,row_indices=points) #New group with the points from the network 
                new_group_created=True
            for lens in new_lenses:
                lens['gain']=1

            new_net=source.create_network(name=new_net_name,
                                          params_dict={'metric':{'id':network.metric['id']},
                                          'column_set_id':network.column_set_id,
                                          'lenses':new_lenses,
                                          'row_group_id':new_group['id']}
                                        )    
            filt=[]
            coloring_name_list=[]
            for lens in new_lenses:
                lens_values=[]
                lens_specification={'lens_id':lens['id'],
                                     'metric_id':network.metric['id'],
                                     'column_set_id':network.column_set_id
                                    }
                coloring_name=lens['id']+"_"+str(struct_time+"_"+str(number))
                coloring_name_list.append(coloring_name)
                coloring=source.create_coloring(name=coloring_name,lens_specification=lens_specification)
                node_values=new_net.get_coloring_values(name=coloring_name)
                for idx,node in enumerate(new_net.nodes):
                    row_count=node['row_count']
                    lens_values.extend([node_values[idx]]*row_count)
                filt.append(lens_values)
                source.sync()
            
            if delete_interim_network_and_coloring==True:
                for coloring_name in coloring_name_list:
                    source.delete_coloring(name=coloring_name)
                source.delete_network(id=new_net.id)
                if new_group_created==True:
                    source.delete_group(name=new_net_name)
                source.sync()

        self.dim=len(filt) #Number of dimensions of the filters
        self.filt=[np.array(filt[i]) for i in range(self.dim)] #Filter values
        for i in range(1,self.dim):
            assert len(filt[i-1])==len(filt[i]) #Must have same number of points in each dimension
        self.filt_min=[min(filt[i]) for i in range(self.dim)]
        self.filt_max=[max(filt[i]) for i in range(self.dim)]
        self.points_count=len(filt[0])
        
    def _arrange_filt(self,filt): #Change shape of the filter function from (d,N) to (N,d)
        shape=np.shape(filt)
        if shape[0]<shape[1]:  #Assume there are more data points than dimensions
            return filt
        else:
            filt1=[]
            dim=shape[1]
            for i in range(dim):
                filt1.append([row[i] for row in filt])
            return filt1
    
    def _get_common_points(self,list1,list2):
        common_points=[]
        for i in range(len(list2)):
            if list2[i] in list1:
                common_points.append(list2[i])
        return common_points
    
    def _get_instance(self,edge1,edge2): #TBD: Change this to run faster for large data sets
        #Get points that lie between two edges. 
        #If >1D, define edge1 as a list of min edge in each dimension
        points_list=np.array(range(len(self.filt[0]))) 
        for i in range(self.dim):
            list1=np.where(self.filt[i]>edge1[i])[0]
            list2=np.where(self.filt[i][list1]<edge2[i])[0]
            points_list=np.intersect1d(points_list,list2)
        return points_list
      

    def _graph_traversal(self,vertices,edges): #Traverses graphs to get number of connected components
        unseen_nodes=vertices
        conn_comp_count=0
        all_conn_comp=[]
        while len(unseen_nodes)>0:
            v=unseen_nodes[0]
            unseen_nodes.remove(v)
            conn_comp_count=conn_comp_count+1
            conn_comp=[v]
            #Get all connected edges
            connected_nodes=[]
            for edge in edges:
                if edge[0]==v and edge[1] in unseen_nodes:
                    connected_nodes.append(edge[1])
                    unseen_nodes.remove(edge[1])
                elif edge[1]==v and edge[0] in unseen_nodes:
                    connected_nodes.append(edge[0])
                    unseen_nodes.remove(edge[0])
            while len(connected_nodes)>0:
                conn_comp.extend(connected_nodes)
                new_connected_nodes=[]
                for node in connected_nodes:
                    for edge in edges:
                        if edge[0]==node and edge[1] in unseen_nodes:
                            new_connected_nodes.append(edge[1])
                            unseen_nodes.remove(edge[1])
                        if edge[1]==node and edge[0] in unseen_nodes:
                            new_connected_nodes.append(edge[0])
                            unseen_nodes.remove(edge[0])
                connected_nodes=new_connected_nodes
            all_conn_comp.append(conn_comp)
        return conn_comp_count, all_conn_comp
    
    def _get_all_bin_edges(self,interval):
        all_bin_edges=[]
        for i in range(self.dim):
            bins=Bins(interval,self.gain,self.filt_min[i],self.filt_max[i])
            bin_edges,overlap_edges=bins.get_bin_edges()
            all_bin_edges.append(bin_edges)
        return all_bin_edges
    
    def _get_points_in_conn_comp(self,conn_comp,all_bin_edges): #Hard-coded for 2D
        #Conn_comp <list>: List of filter intervals that are in the 'connected component'
        #all_bin_edges: As defined by self._get_all_bin_edges
        x_intervals=len(all_bin_edges[0])
        all_points_list=[]
        for v in conn_comp:
            v_x_interval=v%x_intervals
            v_y_interval=v/x_intervals
            min_edge=[all_bin_edges[0][v_x_interval][0],all_bin_edges[1][v_y_interval][0]]
            max_edge=[all_bin_edges[0][v_x_interval][1],all_bin_edges[1][v_y_interval][1]]
            points_list=self._get_instance(min_edge,max_edge)
            all_points_list.extend(points_list)
        all_points_list=sorted(list(set(all_points_list)))
        return all_points_list
        
    def _check_step(self,interval,points_threshold=1,get_graph=False,get_points=False): #Embed covering as a graph: Edges for points in overlap(s), vertex for point(s) in interval
        all_bin_edges=self._get_all_bin_edges(interval)
        covering=Covering(all_bin_edges)
        all_spacings=covering.all_spacings
        all_intervals_in_spacings=covering.all_intervals_in_spacings
        vertices=[]
        edges=[]
        for idx, spacing in enumerate(all_spacings):
            if len(self._get_instance(spacing[0],spacing[1])) >0: #i.e. there exists at least 1 point in this spacing
                filter_intervals=all_intervals_in_spacings[idx]
                if len(filter_intervals)==1: 
                    vertices.append(filter_intervals[0])
                else:
                    for edge in list(itertools.combinations(filter_intervals,2)):
                        edges.append(edge)
        for edge in edges:
            if edge[0] not in vertices:
                vertices.append(edge[0])
            if edge[1] not in vertices:
                vertices.append(edge[1])
        edges=sorted(list(set(edges)))
        conn_comp_count,all_conn_comp=self._graph_traversal(sorted(vertices),edges)

        if points_threshold>1 or get_points:
            conn_comp_count=0
            conn_comp_points=[]
            for conn_comp in all_conn_comp:
                all_points_list=self._get_points_in_conn_comp(conn_comp,all_bin_edges)
                if len(all_points_list)>points_threshold-1:
                    conn_comp_count=conn_comp_count+1
                    conn_comp_points.append(all_points_list)
                    
        if get_graph:
            import graphviz as gv
            h=gv.Digraph(name='filter_graph',engine='neato',format='png')
            h.attr(label=r'\n\nFilter Graph\ndrawn by NEATO')
            h.attr(fontsize='20')
            h.graph_attr['size']="10,10" #Hardcoded graph size
            h.edge_attr['dir']="none"
            for v in vertices:
                h.node(str(v),shape='circle')
            for edge in edges:
                h.edge(str(edge[0]),str(edge[1]))
            return conn_comp_count,h        
        elif get_points==True:
            return conn_comp_count, conn_comp_points
        else:
            return conn_comp_count
    
    def get_interval(self,n=3,output_all_steps=False,points_threshold=1,verbose=False): 
        #Returns the first interval that gives >1 connected component meeting points_threshold in the filter for n continuous resolution instances
        interval=2
        #conn_comp_count=self._check_step(interval)
        instance=0
        if output_all_steps==True:
            output=[['resolution','no_of_connected_components']]
        while instance<n:
            interval=interval+1
            if interval>self.points_count:
                if output_all_steps==True:
                    return None, "Resolution not found"
                else:
                    return None
            if verbose==True:
                print "Calculating resolution = %s" %(str(interval))
            conn_comp_count=self._check_step(interval,points_threshold=points_threshold)
            if conn_comp_count>1:
                instance=instance+1
            elif conn_comp_count==1:
                instance=0
            if output_all_steps==True:
                output.append([interval,conn_comp_count])
        if output_all_steps==True:
            return interval-n+1,output
        else:
            return interval-n+1


        

class visualize_filt: #For 2-D
    def __init__(self,filt,bin_edges): 
        #Filt is a list of lists of filter values
        #bin_edges is a list of list of bin_edges
        self.filt=filt
        self.dim=len(filt)
        self.filt_min=[min(filt[i]) for i in range(self.dim)]
        self.filt_max=[max(filt[i]) for i in range(self.dim)]
        self.bin_edges=bin_edges
        self.intervals=[len(bin_edges[i]) for i in range(self.dim)]
        self.graph_min=0
        self.graph_max=1
        self.scale=0 #To be updated by self._get_relative_bin_edges(...)
    
    def _get_relative_bin_edges(self,graph_min,graph_max):
        bin_min=[self.bin_edges[i][0][0] for i in range(self.dim)]
        bin_max=[self.bin_edges[i][-1][1] for i in range(self.dim)]
        bin_span=[bin_max[i]-bin_min[i] for i in range(self.dim)]
        scale=[(graph_max-graph_min)/bin_span[i] for i in range(self.dim)]
        self.scale=scale
        relative_bin_edges=[]
        for i in range(self.dim):
            relative_bin_edges_1D=[]
            for idx in range(len(self.bin_edges[0])):
                relative_bin_edge=[graph_min+(self.bin_edges[i][idx][0]-bin_min[i])*scale[i],
                                   graph_min+(self.bin_edges[i][idx][1]-bin_min[i])*scale[i]]
                relative_bin_edges_1D.append(relative_bin_edge)
            relative_bin_edges.append(relative_bin_edges_1D)
        return relative_bin_edges
    
    def get_plot(self):
        #Get covering
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='auto',facecolor='#FCE8C4')
        #ax.set_xticks([])
        #ax.set_yticks([])
        relative_bin_edges=self._get_relative_bin_edges(self.graph_min,self.graph_max)
        #Hard-code to 2-D for now
        for i2 in range(self.intervals[1]):
            for i1 in range(self.intervals[0]):
                ax.add_patch(
                     patches.Rectangle(
                        (relative_bin_edges[0][i1][0],relative_bin_edges[1][i2][0]),
                        relative_bin_edges[0][i1][1]-relative_bin_edges[0][i1][0],
                        relative_bin_edges[1][i2][1]-relative_bin_edges[1][i2][0],
                        fill=False      # remove background
                     ) ) 
        #Get points and plot (Hard-code to 2-D for now)
        relative_filt_points=[]
        for idx in range(len(self.filt[0])):
            relative_filt_point=[(self.filt[0][idx]-self.filt_min[0])*self.scale[0]+self.graph_min,
                                 (self.filt[1][idx]-self.filt_min[1])*self.scale[1]+self.graph_min
                                ]
            relative_filt_points.append(relative_filt_point)
        self.relative_filt_points=relative_filt_points    
        ax.scatter([point[0] for point in relative_filt_points],[point[1] for point in relative_filt_points]) 
        return fig,ax




# coding: utf-8

# In[ ]:


'''
This THD script uses mapper written by XH (from scratch). 
A re-written mapper is required in order to avoid stack overflow issues when calculating point-cloud distance in large datasets. 

Contact: Xiu Huan Yap, yap.4@wright.edu  
'''


# ### Helper functions

# In[ ]:


import itertools
import numpy as np
import graphviz as gv
from scipy.stats import mode

class Mapper:
    def __init__(self, points, sklearn_filter_function,gain,interval,training_labels=None):
        self.points=np.array(points)
        self.sklearn_filter_function=sklearn_filter_function 
        
        self.gain=gain
        self.interval=interval #Note that interval is the number of filter levels in 1D. i.e. resolution = interval^2
        
        self.training_labels=np.array(training_labels) #Needed if drawing mapper with training labels
        
        self.filter_values = self._get_filter_values()
        if type(self.training_labels) != None:
            self.label_names, self.label_colors=self._get_label_colors()  
            
    def _get_filter_values(self):
        filter_values=Filter(self.points,self.sklearn_filter_function).filter_values
        return filter_values

    def _get_label_colors(self):
        label_names=np.unique(self.training_labels)
        golden_ratio_conjugate=0.618033988749895
        hue=0
        label_colors=[]
        for i in range(len(label_names)):
            color=str(hue)+" 0.5"+ " 0.95"
            label_colors.append(color)
            hue=hue+golden_ratio_conjugate
            hue=hue%1
        return label_names,label_colors
    
    def get_mapper(self,store_filter_levels=False):
        #Get covering from self.filter_values
        cover=Covering(self.filter_values,self.gain,self.interval)
        all_bin_edges=cover.all_bin_edges
        intervals=cover.intervals
        
        #Get points into filter levels
        point_idx_per_level=self._get_point_idx_per_level(self.filter_values,intervals,all_bin_edges)
        nodes={}
        node_count=0
        node_to_filter_level=[]
        for filter_level in point_idx_per_level.keys():
            point_idx_in_level=point_idx_per_level[filter_level]
            if len(point_idx_in_level)==1:
                clusters=[[0]]
            elif len(point_idx_in_level)<1:
                continue
            else:
                point_values_in_level=self.points[point_idx_in_level]
                clusters=self._mapper_step(point_values_in_level) #List of list of points in same cluster
            if len(clusters)>0:
                for cluster_idx, cluster in enumerate(clusters):
                    node_name=node_count
                    node_count+=1
                    if (store_filter_levels):
                        node_filter_level=filter_level+"."+str(cluster_idx)
                        node_to_filter_level.append(node_filter_level)
                    nodes[node_name]=np.array(point_idx_in_level)[cluster]
        edges={}
        for pair in itertools.combinations(nodes.keys(),2): #Gets all possible combinations of 2 nodes
            common_points=list(set(nodes[pair[0]]).intersection(set(nodes[pair[1]]))) #Gets common points between 2 nodes
            if len(common_points)>0:
                edges[pair]=common_points
        #Save info
        self.mapper_result={'nodes':nodes,'edges':edges}
        return self.mapper_result
    
    def draw_mapper(self,model_entry=None,mapper_result=None,points_list=None,label_method='mode'):
        '''
        Either define model_entry or define mapper_result, points_list
        
        model_entry: <dict> contains keys mapper_result and points_list
        mapper_result: <dict> can be found under key 'mapper_result' for each output in model
        points_list: <list> Idx of points from the original training data. Need it to get the labels
        label_method: <str> 'mode', 'fraction'
                        In 'mode', node is colored by the most popular label
                        In 'fraction', node is colored by proportion of labels 
        '''
        if model_entry:
            mapper_result=model_entry['mapper_result']
            points_list=model_entry['points_list']
        elif type(mapper_result) ==None or type(points_list) ==None:
            raise ValueError('Parameters mapper_result and points_list not defined')
        
        h=gv.Digraph(name='mapper_graph',engine='neato',format='png')
        h.graph_attr['size']="10,10" #Hardcoded graph size
        #.graph_attr['pack']='false'
        h.edge_attr['dir']="none"
        nodes=mapper_result['nodes']
        edges=mapper_result['edges']
        for node in nodes:
            node_size=float(np.log(len(nodes[node])/200+1)) #Arbitrary hard-coded size constraints
            if label_method=='mode':
                node_label=mode(self.training_labels[points_list[nodes[node]]])[0][0]
                node_color=str(self.label_colors[np.where(self.label_names==node_label)[0][0]])
            elif label_method=='fraction':
                label_count=[]
                for node_label in self.label_names:
                    count=np.count_nonzero(self.training_labels[nodes[node]]==node_label)
                    label_count.append(count)
                label_count=np.divide(label_count,float(sum(label_count)))
                node_color=str()
                for idx, freq in enumerate(label_count):
                    if freq >0:
                        node_color=node_color+self.label_colors[idx]+';'+str(freq)+':'
            else:
                node_color='yellow'
            h.node(str(node),shape='circle',label=str(len(nodes[node])),height=str(node_size),style='wedged',fillcolor=node_color)
        for edge in edges:
            h.edge(str(edge[0]),str(edge[1]),penwidth=str(np.log(len(edges[edge])+1)))
        
        #Render mapper graph, legend and paste onto combined mapper_image
        #mapper_graph=Image.open(h.render())
        mapper_image=h
        if label_method!=None:
            legend=self._get_legend_as_graph()
            return mapper_image, legend
        #mapper_image_size=(mapper_graph.size[0]+legend.size[0],mapper_graph.size[1])
        #mapper_image=Image.new('RGB',mapper_image_size,color='white')
        #mapper_image.paste(mapper_graph,(0,0))
        #legend_pos=(mapper_graph.size[0],mapper_graph.size[1]-legend.size[1])
        #mapper_image.paste(legend,legend_pos)
        return mapper_image

    def _get_legend_as_graph(self):
        h=gv.Digraph(engine='dot',format='png')
        h.graph_attr['size']=str(2) #Hardcoded legend size
        h.graph_attr['rankdir']='LR'
        h.graph_attr['mindist']='0.0'
        h.graph_attr['ranksep']='0.0'
        h.graph_attr['nodesep']='0.0'
        h.node_attr['shape']='box'
        h.node_attr['margin']="0,0"
        h.node_attr['width']="1"
        h.node_attr['height']="0.5"
        h.edge_attr['style']='invis'
        #h.edge_attr['minlen']='0'
        for idx, label in enumerate(self.label_names):
            color=str(self.label_colors[idx])
            h.node(color[:3],style='filled',fillcolor=color,fontcolor=color)
            h.edge('label '+str(label),color[:3])
        return h    
    
    def _mapper_step(self, filter_values_in_level):
        #Need to convert filter_value_idx to filter_values_per_level
        point_idx_in_clusters=Clustering(filter_values_in_level).point_idx_in_clusters
        return point_idx_in_clusters
        
    
    def _get_point_idx_per_level(self, filter_values,intervals,all_bin_edges):
    #Returns a dictionary of key:value filter_level:list of indices of points in filter_level
    #Only works for 2D
        point_idx_per_level={}
        for y_interval in range(intervals[1]):
            for x_interval in range(intervals[0]):
                curr_interval=x_interval+y_interval*intervals[0]
                pt_idx_in_filter_level=[]
                for idx, filter_value in enumerate(filter_values):
                    if (self._check_filter_value(filter_value, all_bin_edges[0][x_interval], all_bin_edges[1][y_interval])):
                        pt_idx_in_filter_level.append(idx)
                point_idx_per_level[str(curr_interval)]=pt_idx_in_filter_level
        return point_idx_per_level

    def _check_filter_value(self, filter_value, x_boundaries, y_boundaries):
        if filter_value[0]>=x_boundaries[0] and filter_value[0]<=x_boundaries[1]:
            if filter_value[1]>=y_boundaries[0] and filter_value[1]<=y_boundaries[1]:
                return True
        return False
    
    
    
#These are mapper helper classes

class Filter:
#For sklearn filter functions
    def __init__(self, points, filter_fn):
        self.points=points
        self.filter_fn=filter_fn
        self.filter_values=self._get_filt_values()
        
    def _get_filt_values(self):
        filt=self.filter_fn()
        filter_values=filt.fit_transform(self.points)
        return filter_values

class Covering:
#TBD: 2D for now
#Adapted from getres (aka Adaptive Resolution) package
    def __init__(self, filter_values, gain, interval):
        try:
            filter_values=filter_values.tolist() #Change np array to python list. Prevent misplacing points in filter levels due to precision. 
        except:
            pass
        self.filter_values=filter_values
        self.gain=gain
        self.interval=interval
        
        self.dim=len(filter_values[0])
        for i in range(1,self.dim):
            assert len(filter_values[i-1])==len(filter_values[i]) #Must have same number of points in each dimension
        self.filt_min=[min([filter_values[i][j] for i in range(len(filter_values))]) for j in range(self.dim)]
        self.filt_max=[max([filter_values[i][j] for i in range(len(filter_values))]) for j in range(self.dim)]
        
        self.all_bin_edges=self._get_all_bin_edges(self.interval) #3D list
        self.intervals=[len(bin_edges) for bin_edges in self.all_bin_edges]
        
        self.dim=len(self.all_bin_edges)
        #self.max_interval_list=[len(all_bin_edges[i]) for i in range(self.dim)] #Gets the number of intervals in each dimension
        self.all_spacings,self.all_intervals_in_spacings=self._get_all_spacings()
    
    def _get_all_bin_edges(self,interval):
        all_bin_edges=[]
        for i in range(self.dim):
            bins=self.bins(interval,self.gain,self.filt_min[i],self.filt_max[i])
            bin_edges,overlap_edges=bins.get_bin_edges()
            all_bin_edges.append(bin_edges)
        return all_bin_edges
                
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
        
        

    class bins: #For 1-D
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
            bin_edges[-1][-1]=self.filt_max
           #Update self
            self.bin_edges=bin_edges
            self.overlap_edges=overlap_edges
            return bin_edges,overlap_edges    


from scipy.cluster.hierarchy import fcluster, linkage    
class Clustering:
#Single-linkage clustering, Histogram method, for 2D filters    
    def __init__(self, point_values_per_level):
        self.point_values=point_values_per_level
        self.point_idx_in_clusters=self._get_cluster_memberships()
        
    def _get_cluster_memberships(self):
        #Agglomerative clustering by single-linkage
        Z=linkage(self.point_values,'single') #Returns Z, linkage matrix as defined by scipy
        #Get first empty gap in histogram of edge distances (default=10 bins)
        edge_dist=[link[2] for link in Z]
        histo_freq=list(np.histogram(edge_dist)[0])
        try:
            first_gap=histo_freq.index(0) #Find first gap in histogram
        except:
            return [range(len(self.point_values))] #No gaps in histogram. All points belong to same cluster.
        
        cluster_memberships=list(fcluster(Z,first_gap,criterion='distance'))
        point_idx_in_clusters=[]
        for cluster in np.unique(cluster_memberships):
            point_idx_in_clusters.append(np.where([cluster_memberships[i]==cluster for i in range(len(cluster_memberships))])[0].tolist())
        return point_idx_in_clusters
        
    


# In[ ]:


import Queue
import random
import time

class THD:
    def __init__(self,load_file_name=None,
                 params_dict=None):
                 
#                 training_data,res_increase, 
#                 filter_fn,cov,cut,segmenter,metric='euclidean',training_labels=None,supervised_filter=False):
        '''
        Either enter load_file_name to load THD file (untested), or enter mapper parameters in params_dict
        params_dict <dict>: training_data
                            res_increase
                            res_increment
                            filter_fn
                            cov
                            cut
                            segmenter
                            metric: 'euclidean' <default>
                            training_labels: None <default> Required if supervised filter_fn is used.
                            supervised_filter: False <default> Set as True if supervised filter_fn is used.
                            points_threshold: None <default> parameter for res_increase if it is a getres_2D.connectivity_check
        Callable functions: get_model, draw_model, draw_mapper
        '''
            
        if type(params_dict)!=type(None):
            self.training_data=np.array(params_dict['training_data'])
            self.res_increase=params_dict['res_increase']
            self.res_increment=params_dict['res_increment'] #Alternative resolution increment if res_increase is a function        
            self.sklearn_filter_fn=params_dict['sklearn_filter_fn'] #either unsupervised (takes data parameter) or supervised (takes data, labels parameters)
            self.gain=params_dict['gain']
            self.interval=params_dict['interval']
            
            self.segmenter=params_dict['segmenter'] #This should be an instantiated segmenter class with associated function "get_segments"

            self.metric=params_dict['metric']
            self.training_labels=params_dict['training_labels']
            self.supervised_filter=params_dict['supervised_filter'] #Supervised filters=True: terminate THD when connected component has points of 1 unique label
            self.points_threshold=params_dict['points_threshold']


        self.model=[] #Hierarchy list of dictionaries with keywords /
                #{'idx','parent','children','mapper_result','connected_components'}
        self.next=0 #Next idx for model
        
        if type(load_file_name)!=type(None):
            self.load_model(load_file_name)
            self.training_data=self.training_data[0] #Strange problem where np.arrays are saved as (np.array)
            self.training_labels=self.training_labels[0] #Strange problem where np.arrays are saved as (np.array)
            
        self.queue=Queue.Queue()        
        
        if type(self.training_labels) != None:
            self.label_names, self.label_colors=self._get_label_colors()  
                 
            
    def _get_filt(self,points_list,get_pcd=False):
        data=self.training_data[points_list,:]  
        pcd=pdist(data,self.metric)
        try:
            filt=self.filter_fn(data)
        except: #Supervised filters require labels
            labels=self.training_labels[points_list]
            filt=self.filter_fn(data,labels)
        while len(filt)<len(points_list): #some filters take time! 
            time.sleep(0.1) 
        if get_pcd==True:
            return filt,pcd
        else:
            return filt
            
    def get_mapper_result(self,points_list,interval):
        points=self.training_data[points_list]
        new_mapper=Mapper(points,self.sklearn_filter_fn,self.gain,interval,training_labels=self.training_labels)
        mapper_result=new_mapper.get_mapper()
        
        return mapper_result   
    
    def _assign_res(self,interval, next_res):
        return next_res
    
    def _increase_res(self,interval,res_increment):
        return interval+res_increment
                           
                        
    def get_model(self,points_list=None): 
        if points_list==None:
            points_list=np.array(range(int(np.shape(self.training_data)[0])))
        if not type(self.res_increase)== int:
            filt=self._get_filt(points_list)
            gain=cov.fract_overlap[0]
            connectivity_check=self.res_increase(filt=filt,gain=gain)
            next_res=connectivity_check.get_n_gap(n=3,points_threshold=self.points_threshold)
            assert type(next_res)!=type(None), "First resolution not found"
            cov=self._assign_res(cov,next_res)
        #Queue Mappers  
        assert self.queue.empty()
        assert self.next==0
        item={'idx':self.next,'points_list':points_list,'interval':self.interval}
        self.queue.put(item)
        self.model.append({'idx':self.next,'parent':None,'points_list':np.array(points_list),'interval':self.interval})
        self.next=self.next+1

        #Main loop
        while self.queue.empty()==False:
            item=self.queue.get()
            parent_interval=item['interval']
            parent_points_list=item['points_list']
            mapper_result=self.get_mapper_result(parent_points_list,parent_interval)
            print "Resolution used: %s"%(parent_interval)
            segments=self.segmenter.get_segments(mapper_result)
            #Save mapper result to model
            self.model[item['idx']]['mapper_result']=mapper_result
            self.model[item['idx']]['segments']=segments
            #Get number of children, queue them, and update hierarchy list 
            if len(segments)>0:
                interval=parent_interval #copy parent interval
                parent_idx=item['idx']
                self.model[parent_idx]['children']=range(self.next,self.next+len(segments)) #update parent hierarchy entry
                for segment in segments:
                    points_list=parent_points_list[segment['points_list']] #Convert to point idx of global model
                    if len(segments)==1:
                        if type(self.res_increase)==int:
                            new_interval=self._increase_res(interval,self.res_increase)  #Res increase if only 1 segment
                        else: #If res_increase is a function...
                            new_interval=self._increase_res(interval,self.res_increment) ##TBD
                    elif len(segments)>1:
                        if type(self.res_increase)==int:
                            new_interval=parent_interval #Use parent_interval
                        if not type(self.res_increase)==int: 
                            filt=self._get_filt(points_list)
                            gain=self.gain
                            test=self.res_increase(filt=filt,gain=gain) ##TBD
                            next_res=test.get_n_gap(n=3,points_threshold=self.points_threshold)
                            if next_res==None:
                                new_interval=parent_interval #Use parent cover
                            else:
                                new_interval=self._assign_res(interval,next_res)                   

                    item={'idx':self.next, 'points_list':points_list, 'interval':new_interval}
                    self.model.append({'idx':self.next,'parent':parent_idx,'points_list':np.array(points_list),'interval':new_interval}) #update child hierarchy entry                
                    if self.supervised_filter==True:
                        if len(np.unique(self.training_labels[item['points_list']]))>1.1:
                            self.queue.put(item) #Queue for mapper only if there is more than 1 unique label
                    else:
                        self.queue.put(item)
                    self.next=self.next+1
        return self.model
    
    #Draw mapper from mapper_result
    def draw_model(self,label_method=None): #To do: Add labels
        h=gv.Digraph(engine='dot')
        h.graph_attr['size']="10,10" #Hardcoded graph size
        h.graph_attr['splines']='ortho'
        h.format='svg'
        for idx, mod in enumerate(self.model):
            node_name=str(idx)+": "+ str(len(mod['points_list']))+" points"
            if label_method=='mode':
                node_label=mode(self.training_labels[mod['points_list']])[0][0]
                node_color=str(self.label_colors[np.where(self.label_names==node_label)[0][0]])
            elif label_method=='fraction':
                label_count=[]
                for node_label in self.label_names:
                    count=np.count_nonzero(self.training_labels[mod['points_list']]==node_label)
                    label_count.append(count)
                label_count=np.divide(label_count,float(sum(label_count)))
                node_color=str()
                for i, freq in enumerate(label_count):
                    if freq >0:
                        node_color=node_color+self.label_colors[i]+';'+str(freq)+':'
            if label_method != None:
                h.node(str(idx),node_name+"\nInterval: "+str(mod['interval']),style='wedged',fillcolor=node_color)
            else: 
                h.node(str(idx),node_name+"\nInterval: "+str(mod['interval']))
                
            try:
                for child in mod['children']:
                    h.edge(str(idx),str(child))
            except:
                pass
        return h    
    
    def draw_mapper(self,model_entry=None,mapper_result=None,points_list=None,label_method='mode'):
        '''
        Either define model_entry or define mapper_result, points_list
        
        model_entry: <dict> contains keys mapper_result and points_list
        mapper_result: <dict> can be found under key 'mapper_result' for each output in model
        points_list: <list> Idx of points from the original training data. Need it to get the labels
        label_method: <str> 'mode', 'fraction'
                        In 'mode', node is colored by the most popular label
                        In 'fraction', node is colored by proportion of labels 
        '''
        if model_entry:
            mapper_result=model_entry['mapper_result']
            points_list=model_entry['points_list']
        elif mapper_result ==None or points_list ==None:
            raise ValueError('Parameters mapper_result and points_list not defined')
        
        h=gv.Digraph(name='mapper_graph',engine='neato',format='png')
        h.graph_attr['size']="10,10" #Hardcoded graph size
        #.graph_attr['pack']='false'
        h.edge_attr['dir']="none"
        nodes=mapper_result['nodes']
        edges=mapper_result['edges']
        for node in nodes:
            node_size=float(np.log(len(node['points'])/200+1)) #Arbitrary hard-coded size constraints
            if label_method=='mode':
                node_label=mode(self.training_labels[points_list[node['points']]])[0][0]
                node_color=str(self.label_colors[np.where(self.label_names==node_label)[0][0]])
            elif label_method=='fraction':
                label_count=[]
                for node_label in self.label_names:
                    count=np.count_nonzero(self.training_labels[points_list[node['points']]]==node_label)
                    label_count.append(count)
                label_count=np.divide(label_count,float(sum(label_count)))
                node_color=str()
                for idx, freq in enumerate(label_count):
                    if freq >0:
                        node_color=node_color+self.label_colors[idx]+';'+str(freq)+':'
            h.node(str(node['idx']),shape='circle',label=str(len(node['points'])),height=str(node_size),style='wedged',fillcolor=node_color)
        for edge in edges:
            h.edge(str(edge[0]),str(edge[1]),penwidth=str(np.log(edges[edge]+1)))
        
        #Render mapper graph, legend and paste onto combined mapper_image
        #mapper_graph=Image.open(h.render())
        mapper_image=h
        legend=self._get_legend_as_graph()
        #mapper_image_size=(mapper_graph.size[0]+legend.size[0],mapper_graph.size[1])
        #mapper_image=Image.new('RGB',mapper_image_size,color='white')
        #mapper_image.paste(mapper_graph,(0,0))
        #legend_pos=(mapper_graph.size[0],mapper_graph.size[1]-legend.size[1])
        #mapper_image.paste(legend,legend_pos)
        return mapper_image,legend
    
    def _get_label_colors(self):
        label_names=np.unique(self.training_labels)
        golden_ratio_conjugate=0.618033988749895
        hue=0
        label_colors=[]
        for i in range(len(label_names)):
            color=str(hue)+" 0.5"+ " 0.95"
            label_colors.append(color)
            hue=hue+golden_ratio_conjugate
            hue=hue%1
        return label_names,label_colors
    
    def _get_legend_as_graph(self):
        h=gv.Digraph(engine='dot',format='png')
        h.graph_attr['size']=str(2) #Hardcoded legend size
        h.graph_attr['rankdir']='LR'
        h.graph_attr['mindist']='0.0'
        h.graph_attr['ranksep']='0.0'
        h.graph_attr['nodesep']='0.0'
        h.node_attr['shape']='box'
        h.node_attr['margin']="0,0"
        h.node_attr['width']="1"
        h.node_attr['height']="0.5"
        h.edge_attr['style']='invis'
        #h.edge_attr['minlen']='0'
        for idx, label in enumerate(self.label_names):
            color=str(self.label_colors[idx])
            h.node(color[:3],style='filled',fillcolor=color,fontcolor=color)
            h.edge('label '+str(label),color[:3])
        return h

    
    def save_model(self,base_name):
        assert self.queue.empty(), "Queue is not empty, please wait for get_model() to be completed. "
        import pickle
        struct_time=time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime())  
        number=random.randint(0,99)
        file_name=base_name+"_"+str(struct_time+"_"+str(number))
        self.file_name=file_name
        params_dict={
            'training_data':self.training_data,
            'file_name':self.file_name,
            'res_increase':self.res_increase,
            'res_increment':self.res_increment,
            'sklearn_filter_fn':self.sklearn_filter_fn,
            'gain':self.gain,
            'interval':self.interval,
            'segmenter':self.segmenter,
            'metric':self.metric,
            'training_labels':self.training_labels,
            'supervised_filter':self.supervised_filter,
            'points_threshold':self.points_threshold,
        }
        
        try:
            params_dict['model']=self.model
            params_dict['next']=self.next
        except:
            pass

        with open(file_name,'w') as fp:
            pickle.dump(params_dict, fp)
            print "Model successfully saved as %s." %(self.file_name)
        return True
        
    def load_model(self,file_name):
        import pickle
        with open(file_name,'r') as fp:
            params_dict=pickle.load(fp)        
            
        self.training_data=params_dict['training_data'], 
        self.file_name=params_dict['file_name'],
        self.res_increase=params_dict['res_increase'],
        self.res_increment=params_dict['res_increment'],
        self.sklearn_filter_fn=params_dict['sklearn_filter_fn'],
        self.gain=params_dict['gain'],
        self.interval=params_dict['interval'],
        self.segmenter=params_dict['segmenter'],
        self.metric=params_dict['metric'],
        self.training_labels=params_dict['training_labels'], 
        self.supervised_filter=params_dict['supervised_filter'],
        self.points_threshold=params_dict['points_threshold']

        try:
            self.model=params_dict['model']
            self.next=params_dict['next']
            print "Model successfully loaded."

        except:
            pass

        return True


# In[ ]:


import numpy as np
from scipy.spatial.distance import cdist

class topo_predict:
    def __init__(self, mapper_result, training_data, training_labels, testing_data, k, prediction_method='mode', metric='euclidean'):

        self.mapper_result=mapper_result
        self.nodes=mapper_result['nodes']
        self.training_data=training_data
        self.testing_data=testing_data
        self.training_labels=training_labels
        self.k=k
        self.prediction_method=prediction_method
        self.metric=metric
        self.predictions=[]
        self.pt_membership=self._get_pt_membership()
        
        self.predict_all_testing()
    def __repr__(self):
        '''
        Represent the predictions of topo_predict by a string.
        '''
        return repr({
            'mapper_result': self.mapper_result,
            'testing_data': self.testing_data,
            'testing_labels': self.testing_labels,
            'k': self.k,
            'prediction_method': self.prediction_method,
            'metric': self.metric
        })
    def _get_pt_membership(self):
        #Get the nodes each point belongs to
        pt_membership=[[] for i in range(len(self.training_data))]
        for node_idx in self.nodes.keys():
            for pt_idx in self.nodes[node_idx]:
                pt_membership[pt_idx].append(node_idx)
        self.pt_membership=pt_membership
        return pt_membership
    def predict(self,testing_point): #Predict one testing point at a time
        #Find nearest training point to testing point
        dist=cdist(self.training_data,[testing_point],metric=self.metric)
        dist=np.reshape(dist,(len(dist))) #Convert to 1-D array
        closest_idx=dist.argsort()[0]
        #Get nodes that nearest training point belong to
        closest_nodes=self.pt_membership[closest_idx]
        #Find k-NN in closest_nodes
        points=[]
        for node_idx in closest_nodes:
            points.extend(self.nodes[node_idx])
        points=sorted(np.unique(points))
        k_points=list(dist[points].argsort()[:self.k])
        neighbors_idx=np.array(points)[k_points]
        if self.prediction_method=='mode':
            from scipy.stats import mode
            neighbors_labels=self.training_labels[neighbors_idx]
            [[predicted_label],[freq]]=mode(neighbors_labels)
            freq=float(freq)/self.k
        prediction={'predicted_label':predicted_label,'closest_idx':closest_idx,'neighbors_idx':neighbors_idx,
                   'neighbors_labels':neighbors_labels}
        return prediction
    def predict_all_testing(self): #Predict all testing points
        assert self.predictions==[] #Raise error message 'predictions have already been made'
        for test in self.testing_data:
            prediction=self.predict(test)
            self.predictions.append(prediction)
            


# In[ ]:


from threading import Thread
class THD_predict(THD):
    def __init__(self, load_predict_file=None,
                 load_file_name=None,
                 testing_data=None,
                 testing_labels=None,
                 neighbors_count=1,
                 interpolation_source='containing-nodes',
                 num_threads=8
                ):
        '''
        Either call load_predict_file to load a THD_predict save file, or 
                call load_file_name to load THD_Ayasdi save file, with accompanying parameters.
        '''        
        if load_predict_file!=None:
            self.load_predict_model(load_predict_file)
        else:
            THD.__init__(self,load_file_name=load_file_name)
            
            self.testing_data=testing_data
            self.testing_labels=testing_labels
            self.neighbors_count=neighbors_count
            self.interpolation_source=interpolation_source
            self.num_threads=num_threads
        try:
            self.activations
        except:
            self.activations=self._activations() #Dict of (test_idx: List of networks in THD hierarchy that the closest neighbor belongs) key:value pair
        
        self.queue=Queue.Queue()
        self.queue_complete=False
        
    def _activations(self):
        activations={}
        for test_idx in range(len(self.testing_data)):
            activations[str(test_idx)]=[0]
        return activations
    
    def get_THD_predictions(self):
        #Step1of3: Get activations
        print "Getting activations based on nearest neighbor."
        activations=self._get_activations()
        
        #Step2of3: Get leaf activations
        print "Finding deepest (leaf) activations for each test point."
        leaf_activations=[]
        for test_idx in range(len(self.testing_data)):
            leaf_activation=[]
            activation=self.activations[str(test_idx)]
            for network_idx in activation:
                try:
                    children_idx=self.model[network_idx]['children']
                    if len(children_idx)<1: #Case3: Network has no children - IS a leaf activation
                        leaf_activation.append(network_idx)
                    elif any(np.isin(children_idx,activation)): #Case1: Child network is in activation - NOT a leaf activation
                        pass
                    else:
                        leaf_activation.append(network_idx) #Case2: No child network in activation - IS a leaf activation
                except: #Case3: Network has no children - IS a leaf activation
                    leaf_activation.append(network_idx)
            leaf_activations.append(leaf_activation)
        self.leaf_activations=leaf_activations
        
        #Step3of3: Get predictions using leaf activation networks
        print "Getting predictions based on leaf activations."
        self._get_leaf_predictions()
        
        return self.predictions
    
    def _get_activations(self):
        test_idx = range(len(self.testing_data))
        #Queue activations
        assert self.queue.empty(), "Queue is not empty, please re-initialize THD_predict."
        self.queue_complete=False
        item={'idx':0,'test_idx':test_idx}
        self.queue.put(item)
        
        #Main loop
        threads=[]
        for i in range(self.num_threads):
            t=Thread(target=self._get_activations_worker, name=i)
            t.daemon=True
            t.start()
            threads.append(t)
        self.queue.join()
        self.queue_complete=True
        return self.activations
    
    def _get_activations_worker(self):
        while self.queue_complete==False:
            try:
                item=self.queue.get()
                parent_idx=item['idx']
                parent_test_idx=item['test_idx']
                mapper_result=self.model[parent_idx]['mapper_result']
                training_data=self.training_data[self.model[parent_idx]['points_list']]
                training_labels=self.training_labels[self.model[parent_idx]['points_list']]
                testing_data=self.testing_data[parent_test_idx]
                #Get predictions
                predictions=topo_predict(mapper_result, 
                                         training_data, 
                                         training_labels, 
                                         testing_data,
                                         self.neighbors_count,
                                         prediction_method='mode',
                                         metric='euclidean'
                                        ).predictions  
                self.model[parent_idx]['topo_predict_result']=predictions
                #Get children activations
                try: 
                    self.model[parent_idx]['children']
                    child_test_idx=self._get_next_test_group_and_activation(parent_idx,parent_test_idx,predictions)
                    for child_network in child_test_idx:
                        if len(child_network['test_idx'])>0:
                            self.queue.put(child_network)
                except: #Parent network has not children
                    pass
                print "Finished getting activations for parent network %s." %(parent_idx)
                print "Items left in queue: %s" %(self.queue.qsize())
                self.queue.task_done()
            except:
                pass
#                try:
#                    self.queue.task_done()
#                    print "Having trouble with parent network %s. Re-submitting to queue." %(parent_idx)
#                    self.queue.put(item)
#                except:
#                    print "Unable to resubmit item to queue."
                
    def _get_next_test_group_and_activation(self,parent_idx,parent_test_idx, prediction):
        activation=[[]*len(testing_data)]
        children_idx=self.model[parent_idx]['children']
        parent_training_point_idx=self.model[parent_idx]['points_list']
        nn_array=np.array([parent_training_point_idx[pred['closest_idx']] for pred in prediction]) #Note that closest_idx is relative to parent's training_idx
        child_test_idx=[]
        for child_idx in children_idx:
            #Array of binary values: if nearest-neighbor of test-row is found in child network, then True. Else, False.
            match_array=np.isin(nn_array,self.model[child_idx]['points_list'])
            activated_idx=np.where(match_array)[0]
            activated_test_idx=list(np.array(parent_test_idx)[activated_idx]) #Convert activated_idx to test_idx
            for test_idx in activated_test_idx:
                self.activations[str(test_idx)].append(child_idx) #Save activation
            child_test_idx.append({'idx':child_idx,'test_idx':activated_test_idx})
        return child_test_idx
        
    def _get_leaf_predictions(self):
        self.predictions=[[] for i in range(len(self.leaf_activations))]
        assert self.queue.empty(), "Queue is not empty, please re-initialize THD_predict."
        self.queue_complete=False
        self.queue=Queue.Queue()
        for network_idx in range(len(self.model)):
            self.queue.put(network_idx)
        #Main loop
        threads=[]
        for i in range(self.num_threads):
            t=Thread(target=self._get_leaf_prediction_worker,name=i)
            t.daemon=True
            t.start()
            threads.append(t)
            
        self.queue.join()
        self.queue_complete=True
        return self.predictions
    
    def _get_leaf_prediction_worker(self):
        while self.queue_complete==False:
            print self.queue.qsize()
            network_idx=self.queue.get()
            test_idx=np.where([np.isin(network_idx,leaf_activation) for leaf_activation in self.leaf_activations])[0]
            mapper_result=self.model[network_idx]['mapper_result']
            training_data=self.training_data[self.model[network_idx]['points_list']]
            training_labels=self.training_labels[self.model[network_idx]['points_list']]
            testing_data=self.testing_data[test_idx]            
            
            #Get predictions
            predictions=topo_predict(mapper_result, 
                                         training_data, 
                                         training_labels, 
                                         testing_data,
                                         self.neighbors_count,
                                         prediction_method='mode',
                                         metric='euclidean'
                                        ).predictions 
            self.model[network_idx]['leaf_prediction_result']=predictions
            for idx,test_point_idx in enumerate(test_idx):
                self.predictions[test_point_idx].extend([predictions[idx]['predicted_label']])
            print "Finished getting predictions for Network %s" %(str(network_idx))
            print "Current queue size: %s" %(self.queue.qsize())
            self.queue.task_done()
            


# ### Example usage

# In[ ]:


import numpy as np
from ayasdi.core.api import Api
#Log in to Ayasdi
connection=Api(user_name,user_password)
src=connection.get_source(name='XY_MNIST_ConvolutionalSet.csv')
training_labels=np.array(src.export(column_indices=[2])['data'])[0]

#Get first 5k MNIST258
points_list=[]
for idx,label in enumerate(training_labels):
    if label%3==2:
        points_list.append(int(idx))
    if len(points_list)>4999:
        break
training_labels=training_labels[points_list]

training_data=np.array(src.export(column_indices=range(4,1546),row_indices=points_list)['data'])
training_data=np.transpose(training_data)

#Get 258's in testing set
testing_labels=np.array(src.export(column_indices=[3],row_indices=range(60000,70000))['data'])[0]
testing_points_list=[]
for idx,label in enumerate(testing_labels):
    if label%3==2:
        testing_points_list.append(int(idx))
testing_labels=testing_labels[testing_points_list]
testing_points_list=[testing_points_list[i]+60000 for i in range(len(testing_points_list))]
testing_data=np.array(src.export(column_indices=range(4,1546),row_indices=testing_points_list)['data'])
testing_data=np.transpose(testing_data)


# In[ ]:


from sklearn.manifold import TSNE

gain=0.2
interval=5

new_mapper=Mapper(training_data,TSNE,gain,interval,training_labels=training_labels)
mapper_result=new_mapper.get_mapper()


# In[ ]:


new_mapper.draw_mapper(mapper_result=mapper_result,points_list=new_mapper.points, label_method='fraction')[0]


# In[ ]:


from sklearn.manifold import TSNE
gain =0.2
interval=5
from mapper_extensions.Segmenters.Connected_Component_Segmenter import Connected_Component_Segmenter
connected_component_segmenter=Connected_Component_Segmenter(network_threshold=5)
params_dict={
    'training_data':training_data,
    'res_increase':1,
    'res_increment':None,
    'sklearn_filter_fn':TSNE,
    'gain':gain,
    'interval':interval,
    'segmenter':connected_component_segmenter,
    'metric':'euclidean',
    'training_labels':training_labels,
    'supervised_filter':False,
    'points_threshold':5
}

new_THD2=THD(params_dict=params_dict)
new_THD2.get_model()


# In[ ]:


new_THD2.draw_model(label_method='fraction')


# ### Comparing time taken between python mapper and Xiu's mapper

# In[ ]:


import time
start=time.time()
gain=0.1
interval=5
new_mapper=Mapper(training_data,TSNE,gain,interval,training_labels=training_labels)
mapper_result=new_mapper.get_mapper()
end=time.time()
print("Time taken for Xiu's mapper: %.2f s." %(end-start))


# In[ ]:


from mapper.cover import cube_cover_primitive
from mapper.cutoff import histogram
from mapper import mapper
from scipy.spatial.distance import pdist
start=time.time()
cov=cube_cover_primitive(intervals=5, overlap=10)
cut=histogram(10) #Overlap default=50
pcd=pdist(training_data,'euclidean')
filt=TSNE().fit_transform(training_data).astype('float')
mapper_result2=mapper(pcd, filt, cov, cut)
end=time.time()
print("Time taken for python mapper: %.2f s." %(end-start))


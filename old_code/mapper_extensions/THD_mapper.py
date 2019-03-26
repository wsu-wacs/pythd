#!/usr/bin/env python
import os
import sys
import copy
import Queue
import random
import time
from typing import Union, Dict, Type, List, Any

import numpy as np
import graphviz as gv
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mode

import mapper
from mapper.cover import cube_cover_primitive

class HiddenPrints: 
	"""Enables suppression of mapper outputs. Used by _get_mapper_result
	"""
	def __enter__(self):
		self._original_stdout = sys.stdout
		sys.stdout = None
	def __exit__(self, exc_type,exc_val,exc_tb):
		sys.stdout=self._original_stdout

class THD:
	"""
	THD class

	:param load_file_name: file name to load THD file
	:param params_dict: parameters dictionary with keys: 
						training_data
						res_increase
						res_increment
						filter_cls
						cov
						cut
						segmenter
						metric: 'euclidean' <default>
						training_labels: None <default> Required if supervised filter_cls is used.
						supervised_filter: False <default> Set as True if supervised filter_cls is used.
						points_threshold: None <default> parameter for res_increase if it is a getres_2D.connectivity_check
							   
	Callable functions: get_model, draw_model, draw_mapper
	"""
	def __init__(self,load_file_name=None,
				 params_dict=None):
		# type: (str, Dict) -> None
		options = {'get_pcd':True, #Set as False if training_data is a squareform distance matrix
					'get_test_transform':{} #Set as dict with keys 'test_points' and 'clf' if test transforms required
				  }
		if type(params_dict)!=type(None):
			self.training_data=params_dict['training_data']
			self.res_increase=params_dict['res_increase']
			self.res_increment=params_dict['res_increment'] #Alternative resolution increment if res_increase is a function        
			self.filter_cls=params_dict['filter_cls'] #sklearn class. Will call filter_cls.fit_transform: unsupervised (takes data parameter) or supervised (takes data, labels parameters)
			self.cov=params_dict['cov']
			self.cut=params_dict['cut']
			self.segmenter=params_dict['segmenter'] #This should be an instantiated segmenter class with associated function "get_segments"

			self.metric=params_dict['metric']
			self.training_labels=params_dict['training_labels']
			self.supervised_filter=params_dict['supervised_filter'] #Supervised filters=True: terminate THD when connected component has points of 1 unique label
			self.points_threshold=params_dict['points_threshold']
			
			#Update options dictionary
			for key in params_dict.keys():
				options[key] = params_dict[key]
		
		self.get_pcd = options['get_pcd']
		self.get_test_transform = options['get_test_transform']

		self.queue=Queue.Queue()
		self.model=[] #Hierarchy list of dictionaries with keywords /
				#{'idx','parent','children','mapper_result','connected_components'}
		self.next=0 #Next idx for model
		
		if type(load_file_name)!=type(None):
			self.load_model(load_file_name)
		
		self.label_names, self.label_colors=self._get_label_colors()  
				 
			
	def _get_filt(self,points_list, return_pcd=False):
		# type: (List, bool) -> Tuple(np.ndarray)
		filter_cls=copy.deepcopy(self.filter_cls)
		if self.get_pcd == True:
			data=self.training_data[points_list,:]  
			if return_pcd ==True:
				pcd=pdist(data,self.metric)
		else: #Training_data is a distance matrix. Get points associated with training_data
			pcd=copy.deepcopy(np.array([self.training_data[i][points_list] for i in points_list]))
			data = np.array(copy.deepcopy(pcd))
		if self.supervised_filter: #Supervised filters
			labels=self.training_labels[points_list]
			filt=filter_cls.fit_transform(data,labels).astype('float64')
		else:
			filt=filter_cls.fit_transform(data).astype('float64') #Specify dtype to ensure assert conditions by Mapper module are met
		while len(filt)<len(points_list): #some filters take time! 
			time.sleep(0.1) 

		if return_pcd == False:
			return filt
		else:
			if len(self.get_test_transform.keys())>0:
				test_transform=dict()
				test_transform['test_points']=filter_cls.transform(self.get_test_transform['test_points'])
				return filt, pcd, test_transform
			return filt,pcd

			
	def get_mapper_result(self,points_list,cov):
		# type: (List, Cover) -> Dict
		if len(self.get_test_transform.keys())>0:
			filt,pcd, test_transform = self._get_filt(points_list, return_pcd=True)
		else:
			filt,pcd=self._get_filt(points_list, return_pcd=True)
		#with HiddenPrints():
		mapper_output=mapper.mapper(pcd, filt, cov, self.cut,
			   mask=None,
			   #cluster=cluster_default,
			   point_labels=None,
			   metricpar={},
			   simple=False,
			   filter_info=None,
			   verbose=False)
		mapper_result={}
		mapper_result['nodes']=[]
		for idx,node in enumerate(mapper_output.nodes): 
			mapper_result['nodes'].append({'idx':idx,'points':node.points})
		try:
			mapper_result['edges']=mapper_output.simplices.simplices[1] 
		except:
			mapper_result['edges']=None #No edges
		if len(self.get_test_transform.keys())>0:
			return mapper_result, filt, test_transform
		return mapper_result   
	
	def _assign_res(self,cov, next_res):
		new_cov=copy.deepcopy(cov)
		if type(new_cov.intervals)==int: #1-D filter
			new_cov.intervals=next_res
		else: #Filter is 2-D or higher-D
			new_cov.intervals=np.array([next_res for i in range(len(cov.intervals))])
		return new_cov
	
	def _increase_res(self,cov,res_increment):
		new_cov=copy.deepcopy(cov)
		if type(new_cov.intervals)==int:
			new_cov.intervals=cov.intervals+res_increment
		else:
			new_cov.intervals=np.array([cov.intervals[i]+res_increment for i in range(len(cov.intervals))])
		return new_cov
						   
	def _check_unique_points(self,points_list):
		if self.get_pcd==False:
			data = np.array([self.training_data[i][points_list] for i in points_list])
			if len(np.unique(data))==1:
				return False
			else:
				return True
		else:
			print "Unable to check of uniqueness of points for data matrix, function currently only available for distance matrix"
			
	def get_model(self,points_list=None): 
		# type: (List) -> None
		if points_list==None:
			points_list=range(int(np.shape(self.training_data)[0]))
		cov=copy.deepcopy(self.cov)
		if not type(self.res_increase)== int:
			filt=self._get_filt(points_list)
			gain=cov.fract_overlap[0]
			connectivity_check=self.res_increase(filt=filt,gain=gain)
			try:
				next_res=connectivity_check.get_interval(n=3,points_threshold=self.points_threshold)
			except TypeError: #Supervised get_res
				next_res=connectivity_check.get_interval(y=self.training_labels[points_list],
					points_threshold=self.points_threshold)
			assert type(next_res)!=type(None), "First resolution not found"
			cov=self._assign_res(cov,next_res)
		#Queue Mappers  
		assert self.queue.empty()
		assert self.next==0
		item={'idx':self.next,'points_list':points_list,'cov':cov}
		self.queue.put(item)
		self.model.append({'idx':self.next,'parent':None,'points_list':np.array(points_list),'cov':cov})
		self.next=self.next+1

		#Main loop
		while self.queue.empty()==False:
			item=self.queue.get()
			parent_cov=copy.deepcopy(item['cov'])
			parent_points_list=np.array(item['points_list'])
			if len(self.get_test_transform.keys())>0:
				mapper_result, filt, test_transform=self.get_mapper_result(parent_points_list, copy.deepcopy(parent_cov))
			else:
				mapper_result=self.get_mapper_result(parent_points_list,copy.deepcopy(parent_cov))
			print "Resolution used: %s"%(parent_cov.intervals[0])
			segments=self.segmenter.get_segments(mapper_result)
			#Save mapper result to model
			self.model[item['idx']]['mapper_result']=mapper_result
			self.model[item['idx']]['segments']=segments
			if len(self.get_test_transform.keys())>0:
				self.model[item['idx']]['test_transform']=test_transform
				self.model[item['idx']]['filt']=filt
			#Get number of children, queue them, and update hierarchy list 
			if len(segments)>0:
				cov=parent_cov #copy parent cover
				parent_idx=item['idx']
				self.model[parent_idx]['children']=range(self.next,self.next+len(segments)) #update parent hierarchy entry
				for segment in segments:
					points_list=parent_points_list[segment['points_list']] #Convert to point idx of global model
					if len(segments)==1:
						if type(self.res_increase)==int:
							new_cov=self._increase_res(cov,self.res_increase)  #Res increase if only 1 segment
						else: #If res_increase is a function...
							new_cov=self._increase_res(cov,self.res_increment) ##TBD
					elif len(segments)>1:
						if type(self.res_increase)==int:
							new_cov=copy.deepcopy(cov) #Use parent cover
						if not type(self.res_increase)==int: 
							if self._check_unique_points(points_list)==True:
								filt=self._get_filt(points_list)                                
								gain=cov.fract_overlap[0]
								test=self.res_increase(filt=filt,gain=gain) ##TBD
								try:
									next_res=test.get_interval(n=3,points_threshold=self.points_threshold)
								except TypeError:
									next_res=test.get_interval(self.training_labels[points_list],points_threshold=self.points_threshold)
							else: #No unique points in segments
								next_res=None
							if next_res==None:
								new_cov=copy.deepcopy(cov) #Use parent cover
							else:
								new_cov=self._assign_res(cov,next_res)                   

					item={'idx':self.next, 'points_list':points_list, 'cov':new_cov}
					self.model.append({'idx':self.next,'parent':parent_idx,'points_list':np.array(points_list),'cov':new_cov}) #update child hierarchy entry                
					if self.supervised_filter==True:
						if len(np.unique(self.training_labels[item['points_list']]))>1.1:
							self.queue.put(item) #Queue for mapper only if there is more than 1 unique label
					elif self._check_unique_points(item['points_list'])==False:
						pass #Do not queue if points are not unique
					else:             
						self.queue.put(item)
					self.next=self.next+1
		#Get filter values and test_transform for leaf nodes
		if len(self.get_test_transform.keys())>0:
			self._get_leaf_test_transform()
		#return self.model
	
	def _get_leaf_test_transform(self):
		for idx, thd_node in enumerate(self.model):
			thd_node['test_transform']=thd_node.get('test_transform',{})
			if len(thd_node['test_transform'].keys())<1:
				filt, pcd, test_transform=self._get_filt(thd_node['points_list'], return_pcd=True)
				self.model[idx]['filt']=filt
				self.model[idx]['test_transform']=test_transform

	#Draw mapper from mapper_result
	def draw_model(self,label_method=None): 
		# type: (str) -> gv.dot.Digraph
		h=gv.Digraph(engine='dot')
		h.graph_attr['size']="10,10" #Hardcoded graph size
		h.graph_attr['splines']='ortho'
		h.format='svg'
		for idx, mod in enumerate(self.model):
			node_name=str(idx)+": "+ str(len(mod['points_list']))+" points"
			if label_method=='mode':
				if len(self.label_colors)<2:
					self.label_names, self.label_colors=self._get_label_colors() 
				node_label=mode(self.training_labels[mod['points_list']])[0][0]
				node_color=str(self.label_colors[np.where(self.label_names==node_label)[0][0]])+';1:'
			elif label_method=='fraction':
				if len(self.label_colors)<2:
					self.label_names, self.label_colors=self._get_label_colors()  
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
				h.node(str(idx),node_name+"\nRes: "+str(mod['cov'].intervals[0]),style='wedged',fillcolor=node_color)
			else: 
				h.node(str(idx),node_name+"\nRes: "+str(mod['cov'].intervals[0]))
				
			try:
				for child in mod['children']:
					h.edge(str(idx),str(child))
			except:
				pass
		return h    
	
	def draw_mapper(self,model_entry=None,mapper_result=None,points_list=None,label_method='mode'):
		# type: (Dict, Dict, List, str) -> (gv.dot.Digraph, gv.dot.Digraph)
		"""
		Either define model_entry or define mapper_result, points_list

		:param model_entry: contains keys: mapper_result and points_list
		:param mapper_result: can be found under key 'mapper_result' for each output in model
		:param points_list: Idx of points from the original training data. Need it to get the labels.
		:param label_method: 'mode', 'fraction'. 
							In 'mode', a node is colored by its most popular label. 
							In 'fraction', a node is colored by the proportion of its labels
		"""
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
				node_color=str(self.label_colors[np.where(self.label_names==node_label)[0][0]])+';1:'
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
		# type: (str) -> bool
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
			'cov':self.cov,
			'cut':self.cut,
			'segmenter':self.segmenter,
			'metric': self.metric,
			'training_labels':self.training_labels,
			'points_threshold':self.points_threshold,
			'supervised_filter':self.supervised_filter,

		}
		try: 
			with open(file_name,'w') as fp:
				pickle.dump(self.filter_cls, fp)
			params_dict['filter_cls']=self.filter_cls
		except:
			print "filter_cls cannot be pickled. This parameter is not saved into file."
		try:
			params_dict['model']=self.model
			params_dict['next']=self.next
		except:
			pass

		with open(file_name,'w') as fp:
			pickle.dump(params_dict, fp)
		return True
		
	def load_model(self,file_name):
		# type: (str) -> bool
		import pickle
		with open(file_name,'r') as fp:
			params_dict=pickle.load(fp)        
			
		self.training_data=params_dict['training_data'],
		self.file_name=params_dict['file_name'],
		self.res_increase=params_dict['res_increase'],
		self.res_increment=params_dict['res_increment'],

		self.cov=params_dict['cov'],
		self.cut=params_dict['cut'],
		self.segmenter=params_dict['segmenter'],
		self.metric=params_dict['metric'],
		self.training_labels=np.array(params_dict['training_labels'])
		self.points_threshold=params_dict['points_threshold'],
		self.supervised_filter=params_dict['supervised_filter'],
		
		try:
			self.model=params_dict['model']
			self.next=params_dict['next']
			print "Model successfully loaded."

		except:
			pass
		try:
			self.filter_cls=params_dict['filter_cls']
		except:
			pass
		return True
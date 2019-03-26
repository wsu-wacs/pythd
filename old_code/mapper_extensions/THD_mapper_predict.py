#!/usr/bin/env python
import copy
import Queue
import time
import warnings
from typing import Union, Dict, Type, List, Any
from threading import Thread

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import mode
import graphviz as gv

from .THD_mapper import THD

class THD_mapper_predict(THD):
	"""
	This class implements predictions on THD class. 
	Either call load_predict_file to load a THD_mapper_predict save-file, or 
		call load_THD_file to load THD_mapper save file, with accompanying parameters. 

	:param load_predict_file: file name to load THD_mapper_predict instance
	:param load_THD_file: file name to load THD instance
	:param test_data: 
	:param neighbors_count: Number of neighbors used for prediction in 'leaf activation' node
	:param neighbors_method: "dataspace_constant" assumes point cloud distances remain constant for all networks
									 "dataspace_network" re-calculates point cloud distances for each network
									 "xspace_network" calculates point cloud distances in transformed space for each network. Requires get_test_tranform in THD class.
	:param num_threads: Number of threads used in finding activations and generating predictions								 
	"""

	def __init__(self,
		load_predict_file=None,

		load_THD_file=None,
		test_data=None,
		neighbors_count=1,
		neighbors_method="dataspace_constant",

		num_threads=8
		):
		# type: (str, str, Union[np.ndarray, pd.core.frame.DataFrame], int, str, int) -> None
		self.activations={}

		if load_predict_file!=None:
			self.load_predict_model(load_predict_file)
	
		else:
			THD.__init__(self, load_file_name=load_THD_file)
			self.test_data=test_data
			self.neighbors_count=neighbors_count
			self.neighbors_method=neighbors_method
			if neighbors_method=="dataspace_constant":
				self.test_train_dist=self._get_test_train_dist()

		if len(self.activations.keys())<1:
			self.activations=self._activations() #Holds nearest-neighbor path along THD for each testing point
	
		self.num_threads=num_threads
		self.queue=Queue.Queue()
		self.queue_complete=False

	def _activations(self):
		activations={}
		for test_id in range(len(self.test_data)):
			activations[str(test_id)]=[0] #All points belong to root node
		return activations

	def _get_test_train_dist(self):
		return cdist(self.test_data, self.training_data[0], metric=self.metric[0]) #Training_data and metric somehow saved as a tuple


	def get_THD_predictions(self):
		# type: () -> List
		#Step1of3: Get activations
		print "Getting activations based on nearest neighbor."
		activations=self._get_activations()

		#Step2of3: Get leaf activations
		print "Finding deepest (leaf) activations for each test point."
		self.leaf_activations=self._get_leaf_activations(activations)

		#Step3of3: Get predictions using leaf activation networks
		print "Getting predictions based on leaf activations."
		predictions=self._get_leaf_predictions()

		return self.predictions

	def _get_activations(self):
		#Queue activations
		assert self.queue.empty(), "Queue is not empty. Re-initialize THD_mapper_predict."
		item={'idx':0, 'test_row_ids':range(len(self.test_data))}
		self.queue_complete=False
		self.queue.put(item)
		#Multi-threading
		threads=[]
		for i in range(self.num_threads):
			t=Thread(target=self._get_activations_worker,name=i)
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
				parent_test_row_ids=item['test_row_ids']
				#Get topo predict on mapper
				parent_train_row_ids=self.model[parent_idx]['points_list']
				topo_predict_result=self._get_topo_predict(parent_train_row_ids, parent_test_row_ids, network_idx=parent_idx)
				self.model[parent_idx]['topo_predict_result']=topo_predict_result
				try:
					self.model[parent_idx]['children']
					child_test_row_ids=self._get_next_test_group_and_activation(parent_idx, 
						parent_test_row_ids)
					for child_network in child_test_row_ids:
						if len(child_network['test_row_ids'])>0:
							self.queue.put(child_network)
				except:
					pass #Network parent_idx does not have any children in THD
				#print "Finished getting activations for network %s." %(parent_idx)
				self.queue.task_done()
			except:
				self.queue.task_done()
				print "Having trouble with network %s. Re-submitting to queue." %(parent_idx)
				#self.queue.put(item)


	def _get_topo_predict(self, 
		train_row_ids,
		test_row_ids, 
		network_idx=None, 
		k=1):
		
		if self.neighbors_method=="dataspace_constant":
			tt_dist=copy.deepcopy(self.test_train_dist[np.ix_(test_row_ids, train_row_ids)])

		else:
			assert type(network_idx)!=type(None)
			#Get point cloud distance
			if self.neighbors_method=="dataspace_network":
				test_data=self.test_data[test_row_ids]
				training_data=self.training_data[0][self.model[network_idx]['points_list']]
			elif self.neighbors_method=="xspace_network":
				test_data=self.model[network_idx]['test_transform']['test_points'][test_row_ids]
				training_data=self.model[network_idx]['filt']
			tt_dist=cdist(test_data, training_data, metric=self.metric[0])
		if k==1:
			topo_predict_result=np.reshape(np.array(train_row_ids)[np.argmin(tt_dist,axis=1)],(len(test_row_ids),1))
		elif k<len(train_row_ids):
			topo_predict_result=np.array(map(lambda x: np.array(train_row_ids)[np.argpartition(x,k)[:k]], tt_dist))
		else:
			topo_predict_result=np.array(map(lambda x: np.array(train_row_ids), tt_dist))
		return topo_predict_result

	def _get_next_test_group_and_activation(self, 
		parent_THDnode_idx,
		test_row_ids):
		#Returns the next test_row_ids in the THD hierarchy, and saves the activations
		assert len(test_row_ids)==len(self.model[parent_THDnode_idx]['topo_predict_result']), "Topo predict result length mismatch"
		#activation=[[]*len(test_row_ids)] Seems out of place
		children_network_idx=self.model[parent_THDnode_idx]['children']
		nn_array=copy.deepcopy(self.model[parent_THDnode_idx]['topo_predict_result'])
		child_test_row_ids=[]
		for child_idx in children_network_idx:
			#Array of binary values: if nearest-neighbor of test-row is found in child network, then True. Else False. 
			matched_array=np.isin(nn_array, self.model[child_idx]['points_list'])
			activated_idx=list(np.array(test_row_ids)[(np.where(matched_array)[0])])
			for test_row_id in activated_idx:
				self.activations[str(test_row_id)].append(child_idx) #Save activation for child_node_id
			child_test_row_ids.append({'idx':child_idx, 'test_row_ids':activated_idx})
		return child_test_row_ids

	def _get_leaf_activations(self, activations):
		leaf_activations=[]
		for test_row_id in range(len(self.test_data)):
			leaf_activation=[]
			activation=activations[str(test_row_id)]
			for THD_node_id in activation:
				try:
					children_idx=self.model[THD_node_id]['children']
					if any(np.isin(children_idx,activation)): #Case1: Child network is in activation - NOT a leaf activation
						pass
					else:
						leaf_activation.append(THD_node_id) #Case2: No child network in activation - IS a leaf activation
				except: #Case3: Network has no children - IS a leaf activation
					leaf_activation.append(THD_node_id)
			leaf_activations.append(leaf_activation)
		return leaf_activations

	def _get_leaf_predictions(self):
		self.predictions=[{} for i in range(len(self.leaf_activations))]
		assert self.queue.empty(), "Queue is not empty. Re-initialize THD_predict."
		self.queue=Queue.Queue()
		self.queue_complete=False
		for network_idx in range(len(self.model)):
			self.queue.put(network_idx)
		#Multithreading
		threads=[]
		for i in range(self.num_threads):
			t=Thread(target=self._get_leaf_prediction_worker, name=i)
			t.daemon=True
			t.start()
			threads.append(t)

		self.queue.join()
		self.queue_complete=True
		for i in range(len(self.predictions)):
			self.predictions[i]['prediction']=mode(self.predictions[i]['closest_row_ids_labels'])[0][0]
		return self.predictions

	def _get_leaf_prediction_worker(self):
		while self.queue_complete==False:
			try:
				network_idx=self.queue.get()
				test_row_ids=np.where([np.isin(network_idx,leaf_activation) for leaf_activation in self.leaf_activations])[0]
				train_row_ids=self.model[network_idx]['points_list']
				if self.neighbors_count>len(train_row_ids):
					warnings.warn('There are fewer training points in network %s than neighbors_count.' %(network_idx))
				topo_predict_neighbors=self._get_topo_predict(train_row_ids, test_row_ids,network_idx=network_idx,k=self.neighbors_count)
				result=self._get_label_from_neighbors(topo_predict_neighbors)
				self.model[network_idx]['leaf_prediction_result']=copy.deepcopy(result)

				for idx, test_row_id in enumerate(test_row_ids):
					#Get empty list or extend current list
					self.predictions[test_row_id]['closest_row_ids_labels']=self.predictions[test_row_id].get('predicted_values',[])
					self.predictions[test_row_id]['closest_row_ids_labels'].extend(result[idx])
					self.predictions[test_row_id]['closest_row_ids']=self.predictions[test_row_id].get('closest_row_ids',[])
					self.predictions[test_row_id]['closest_row_ids'].extend([topo_predict_neighbors[idx]])
				#print "Finished getting predictions for network %s" %(str(network_idx))
				self.queue.task_done()
			except:
				self.queue.task_done()
				print "Having trouble with network %s. Re-submitting to queue." %(str(network_idx))
				#self.queue.put(network_idx)


	def _get_label_from_neighbors(self, topo_predict_neighbors):
		if len(topo_predict_neighbors.shape)==1:
			result=map(lambda x: self.training_labels[x], topo_predict_neighbors)
			return result
		else:
			result=map(lambda x: [self.training_labels[i] for i in x],topo_predict_neighbors)
			return result

	def get_predict_proba(self):
		# type: () -> List
		#Probability of prediction class. Assumes no missing values. 
		predict_probas=[]
		for idx, test_predict in enumerate(self.predictions):
			predict_proba=np.divide(mode(test_predict['closest_row_ids_labels'])[1][0],
				float(len(test_predict['closest_row_ids_labels'])))
			self.predictions[idx]['predict_proba']=predict_proba
			predict_probas.append(predict_proba)
		return predict_probas

	def load_predict_model(self):
		#To be written
		pass
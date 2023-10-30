import numpy as np
import pandas as pd
import math
from rdflib import Graph, URIRef, Literal, BNode, Namespace
from rdflib.namespace import FOAF, RDF, RDFS, OWL
import ontospy
import networkx as nx
import networkx.algorithms.community as nx_comm
import pickle
from sklearn.cluster import DBSCAN, Birch, AffinityPropagation
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
import hdbscan
from collections import defaultdict
from itertools import combinations
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import json 
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules



def complete_parse(graph):
  # define sets and dicts for simplicity 
  # they store key information for the functions
  
  property_set = set()
  property_count_dict = dict()
  entity_set = set()
  entity_type_dict = dict()
  entity_prop_dict = dict()
  all_types = set()
  all_types_dict = dict()

  for (s, p, o) in graph: 
    entity_set.add(s)
    #property_set.add(p)
    property_set.add(str(p+'.out'))
    if p in property_count_dict:
      property_count_dict[p] = property_count_dict[p] + 1
    else: 
      property_count_dict[p] = 1

    if s in entity_prop_dict:
      entity_prop_dict[s].add(str(p+'.out'))
    else:
      entity_prop_dict[s] = set()
      entity_prop_dict[s].add(str(p+'.out'))
    if p == RDF.type:
      all_types.add(o)
      if o in all_types_dict.keys():
        all_types_dict[o] = all_types_dict[o] + 1
      else: 
        all_types_dict[o] = 1
      if s in entity_type_dict:
        entity_type_dict[s].add(o)
      else: 
        entity_type_dict[s] = set()
        entity_type_dict[s].add(o)
    
    if not isinstance(o, Literal):
      property_set.add(str(p+'.in'))
      entity_set.add(o)
      if o in entity_prop_dict:
        entity_prop_dict[o].add(str(p+'.in'))
      else:
        entity_prop_dict[o] = set()
        entity_prop_dict[o].add(str(p+'.in'))
    
  
  return (property_set, property_count_dict, entity_set, entity_type_dict, 
          entity_prop_dict, all_types, all_types_dict)

def extract_property_types(entity_prop_dict):
  # first order all sets to make sure that the produced strings match
  sorted_dict = {key: sorted(value) for key, value in entity_prop_dict.items()}

  # merge the set of properties into a string to be used as key
  merged_dict = {key:','.join(value) for key, value in sorted_dict.items()}

  # reverse the dictionary to have property type sets as keys and 
  # entities as values 
  reversed_dict = defaultdict(list)
  for key, value in merged_dict.items():
    reversed_dict[value].append(key)
  
  # split into two different dictionaries 
  # one with the patterns and one with their allocation to entities
  entity_dict = {index: value for index, value in enumerate(reversed_dict.values(), 0)}
  property_dict = {index: key for index, key in enumerate(reversed_dict.keys(), 0)}

  property_dict_analytical = dict()
  for key, value in property_dict.items():
    property_dict_analytical[key] = value.split(',')

  
  return entity_dict, property_dict_analytical

def generate_one_hot_matrix(property_dict_analytical, property_set):
  # create an one hot encoded numpy matrix with 
  # columns: properties in the order contained in the dict 
  # rows: samples-property sets with 1 where property exists

  total_columns = len(list(property_set))
  total_rows = len(property_dict_analytical.keys())

  one_hot_property_matrix = np.zeros((total_rows, total_columns))

  properties_general_map = {str(item): index for index, item in enumerate(list(property_set), start=0)}

  for key, value_list in property_dict_analytical.items():
    for value in value_list:
      one_hot_property_matrix[key][properties_general_map[value]] = one_hot_property_matrix[key][properties_general_map[value]] + 1
  
  return one_hot_property_matrix


def find_subset_property_sets(one_hot_matrix):
  num_rows, num_cols = one_hot_matrix.shape

  # Compute the subset matrix
  subset_matrix = np.matmul(one_hot_matrix, one_hot_matrix.T)
  subset_matrix = subset_matrix == np.sum(one_hot_matrix, axis=1, keepdims=True)

  # Find the subset property sets
  subsets = {}
  for i in range(num_rows):
    subset_indices = np.where(subset_matrix[i] & (i != np.arange(num_rows)))[0].tolist()
    if subset_indices:
      subsets[i] = subset_indices

  return subsets

def generate_plain_subset_hierarchy(num_clusters):
  hierarchy_graph = nx.DiGraph()

  nodes = [index for index in range(num_clusters + 1)]
  for i in nodes:
    hierarchy_graph.add_edge(-1, i)
  return hierarchy_graph

def generate_simple_subset_hierarchy(subset_dict, num_clusters):
  hierarchy_graph = nx.DiGraph()

  nodes = [index for index in range(num_clusters)]
  for key in subset_dict.keys():
    for val in subset_dict[key]:
      hierarchy_graph.add_edge(val, key)


  return hierarchy_graph

def cluster_with_DBSCAN(one_hot_property_matrix, eps = 0.8, min_samples = 1, metric='jaccard'):
  # Calculate matrix of distances
  distance_matrix = pdist(one_hot_property_matrix, metric=metric)
  distance_matrix = squareform(distance_matrix)

  clustering = DBSCAN(metric='precomputed',eps=eps, min_samples=min_samples)
  clustering.fit(distance_matrix)

  num_clusters = max(clustering.labels_)

  return clustering, num_clusters


def cluster_with_HDBSCAN(one_hot_property_matrix, metric='jaccard'):

  distance_matrix = pdist(one_hot_property_matrix, metric=metric)
  distance_matrix = squareform(distance_matrix)
  clusterer = hdbscan.HDBSCAN(metric='precomputed')
  labels = clusterer.fit_predict(distance_matrix)

  hierarchy = clusterer.condensed_tree_

  hierarchy_network = hierarchy.to_networkx()


  return hierarchy_network, len(labels)

def extract_cluster_graphs_hierarchical(graph_list, hierarchy_network):
  
  max_first_element, _ = max(list(hierarchy_network.edges), key=lambda x: x[0])

  # FIX REMAINING CLUSTERS
  for i in range(max_first_element - len(graph_list) + 1):
    graph_list.append(Graph())
  
  for (i,j) in hierarchy_network.edges:
    graph_list[i] = graph_list[i] = graph_list[j]

  return graph_list


def extract_cluster_dictionaries(entity_dict, clustering):
  # Create a list of graphs where each graph is a cluster
  cluster_entity_dict = dict()
  for key, value_list in entity_dict.items():
    for value in value_list:
      cluster_entity_dict[value] = clustering.labels_[key]
  
  return cluster_entity_dict

def extract_cluster_dictionaries_mod(entity_dict, cluster_list):
  # Create a list of graphs where each graph is a cluster
  cluster_entity_dict = dict()
  for key, value_list in entity_dict.items():
    for value in value_list:
      cluster_entity_dict[value] = cluster_list[key]
  
  return cluster_entity_dict

def extract_cluster_graphs(cluster_entity_dict, graph, num_clusters):
  graph_list = [] 

  for i in range(num_clusters+1):
    graph_list.append(Graph())
  
  for (s, p, o) in graph: 
    tmp_index = cluster_entity_dict[s]
    graph_list[tmp_index].add((s, p, o))
    if not isinstance(o, Literal):
      tmp_index = cluster_entity_dict[o]
      graph_list[tmp_index].add((s, p, o))
  
  return graph_list 


def generate_hierarchy(graph_list, hierarchy_network, one_hot_matrix, weight_matrix, cluster_cs_dict):
  
  cluster_list = [Clustering(obj) for obj in graph_list]
  final_hierarchy = Hierarchy(hierarchy_network, cluster_list, one_hot_matrix, weight_matrix, cluster_cs_dict)

  return final_hierarchy


def compare_clusters(cluster_1, cluster_2, comparison='property'):

  cluster_1_one_hot = cluster_1.one_hot_property_representation()
  cluster_2_one_hot = cluster_2.one_hot_property_representation()

  similarity_scores_1 = cosine_similarity(cluster_1_one_hot, cluster_1_one_hot)
  most_similar_indices_1 = np.argsort(similarity_scores_1, axis=0)[-1]


  return most_similar_indices_1


## GMM Preprocessing 
  
def gmm_preprocessing(entity_prop_dict, entity_type_dict, all_types):
  # first order all sets to make sure that the produced strings match
  sorted_dict_prop = {key: sorted(value) for key, value in entity_prop_dict.items()}

  # merge the set of properties into a string to be used as key
  merged_dict_prop = {key:' '.join(value) for key, value in sorted_dict_prop.items()}

  # Do the same for types
  sorted_dict_type = {key: sorted(value) for key, value in entity_type_dict.items()}
  merged_dict_type = {key:' '.join(value) for key, value in sorted_dict_type.items()}

  labs_sets_string = set(merged_dict_type.values())
  labs_sets = [string.split(' ') for string in labs_sets_string]

  distinct_labels = set(all_types)

  merged_dict = {}

  # Iterate over keys in dict1
  for key in merged_dict_type:
    if key in merged_dict_prop:
      merged_dict[key] = merged_dict_type[key] + merged_dict_prop[key]
    else:
      merged_dict[key] = merged_dict_type[key]

  # Iterate over keys in dict2 that are not in dict1
  for key in merged_dict_prop:
    if key not in merged_dict_type:
      merged_dict[key] = merged_dict_prop[key]

  # reverse the dictionary to have property type sets as keys and 
  # entities as values 
  reversed_dict = defaultdict(list)
  for key, value in merged_dict.items():
    reversed_dict[value].append(key)
  
  amount_dict = {key: len(value) for key, value in reversed_dict.items()}
  list_of_distinct_nodes = amount_dict.keys()

  
  # split into two different dictionaries 
  # one with the patterns and one with their allocation to entities
  entity_dict = {index: value for index, value in enumerate(reversed_dict.values(), 0)}
  property_dict = {index: key for index, key in enumerate(reversed_dict.keys(), 0)}

  property_dict_analytical = dict()
  for key, value in property_dict.items():
    property_dict_analytical[key] = value.split(',')

  
  return amount_dict, list_of_distinct_nodes, distinct_labels, labs_sets


def remove_namespace(url_string):
  
  return url_string.rsplit('/', 1)[-1]


def identify_most_distinct_labels(root_node, leaf_nodes, property_count_matrix, total_count_matrix):

  # Select only the rows of interest for the property count 
  selected_rows_property_c =  property_count_matrix[leaf_nodes]
  selected_rows_total = total_count_matrix[leaf_nodes]

  #normalize each row based on the count of each row
  norm_rows_property_c = selected_rows_property_c / selected_rows_total

  # Initialize an empty result array
  result_array = np.empty_like(norm_rows_property_c)

  # Iterate over each row
  for i in range(norm_rows_property_c.shape[0]):
    # Get the maximum values in each column for the remaining rows
    max_values = np.max(np.delete(norm_rows_property_c, i, axis=0), axis=0)
    
    # Calculate the difference between the current row and the max values
    diff_array = norm_rows_property_c[i] - max_values
    
    # Add the difference array to the result array
    result_array[i] = diff_array
  
  max_indices = np.argmax(result_array, axis=1)

  return max_indices

def mask_one_hot_matrix(one_hot_matrix, columns, negative=1, strength='full'):
  # This function is used to apply feedback in the clustering process
  # The one hot matrix is altered to encorporate the feedback of the user
  # Features either get deleted or augmented according to the feedback 

  if (negative==1):
    result = one_hot_matrix
    result[:][columns] = 0
  else:
    shape = one_hot_matrix.shape
    result = np.zeros(shape)
    
    for col in columns:
      result[:, col] = one_hot_matrix[:, col]
  
  return result 

#####################################################################
## Start of functions to support frequent-itemsets implementation ###
#####################################################################

# convert an array to a list of indexes where the value is 1
def get_non_zero_indices(arr):
    # convert array to int 
    arr = arr.astype(int)
    result_list = []
    for i in range(len(arr)):
        if arr[i] != 0:
            result_list.append(i)
    return result_list

def weighted_matrix_to_list_of_lists(weighted_one_hot_matrix):
    result_list = []
    for i in range(weighted_one_hot_matrix.shape[0]):
        result_list.append(get_non_zero_indices(weighted_one_hot_matrix[i]))
    return result_list

def multiply_list_of_lists(list_of_lists, weight_matrix):
    # for each list in the list of lists
    # insert it as many times as the weight of the cluster
    # in a new list
    result_list = []
    for i in range(len(list_of_lists)):
        for j in range(weight_matrix[i]):
            result_list.append(list_of_lists[i])
    
    return result_list

def generate_frequent_itemset(list_of_lists):

    encoder = TransactionEncoder()
    data_encoded = encoder.fit(list_of_lists).transform(list_of_lists)
    data_df = pd.DataFrame(data_encoded, columns=encoder.columns_)
    
    return data_df

def generate_cs_from_itemsets(frequent_itemsets, feature_len):
    shape = (len(frequent_itemsets), feature_len)
    cs_frequent_matrix = np.zeros(shape)
    for i in range(len(frequent_itemsets)):
        for j in frequent_itemsets['itemsets'][i]:
            cs_frequent_matrix[i][j] = 1
    return cs_frequent_matrix, list(frequent_itemsets['support'])

# generate a function that takes two arrays 
# and returns true if the first array has 1 in every position that the second array has 1
def compare_arrays_for_subset(arr1, arr2):
    # convert to int
    arr1 = arr1.astype(int)
    arr2 = arr2.astype(int)
    # compare arrays
    common = np.logical_and(arr1, arr2)
    # get the negative of the first array
    neg_arr1 = np.logical_not(arr1)
    # get the or of both arrays
    result = np.logical_or(common, neg_arr1) 
    # check if the result array has only ones
    if np.all(result == 1):
        return True
    else:
        return False

def generate_initial_global_clusters(cs_frequent_matrix, weighted_one_hot_matrix):
    cs_global_freq_dict = {}
    # for every row of the weighted one hot matrix 
    for i in range(weighted_one_hot_matrix.shape[0]):
        # for every row of the cs frequent matrix
        for j in range(cs_frequent_matrix.shape[0]):
            # check if the cs frequent matrix row is a subset of the weighted one hot matrix row
            if compare_arrays_for_subset(cs_frequent_matrix[j], weighted_one_hot_matrix[i]):
                # if it is add it to the dict
                if i in cs_global_freq_dict.keys():
                    cs_global_freq_dict[i].append(j)
                else:
                    cs_global_freq_dict[i] = [j]
    
    return cs_global_freq_dict

def generate_exclusive_clusters_dummy(cs_global_freq_dict, total_cs):
    exclusive_dict = {}
    for (key,val) in cs_global_freq_dict.items():
        exclusive_dict[key] = max(val)
    # generate a set with all values from zero to total_cs
    total_cs_set = set(range(total_cs))
    # get the difference between the two sets
    exclusive_set = total_cs_set - set(exclusive_dict.keys())
    excluded = list(exclusive_set)
    return exclusive_dict, excluded

def reverse_and_normalize_dict(exclusive_dict, excluded):
    reverse_dict = {}
    for (key, val) in exclusive_dict.items():
        if val in reverse_dict.keys():
            reverse_dict[val].append(key)
        else:
            reverse_dict[val] = [key]
    normalized_dict = {}
    counter = 0 
    # Iterate through the original dictionary
    for key in reverse_dict.keys():
        normalized_dict[counter] = reverse_dict[key]
        counter += 1
    if len(excluded) != 0:
        normalized_dict[counter] = excluded
    return normalized_dict


#####################################################################
#### End of functions to support frequent-itemsets implementation ###
#####################################################################

class Clustering:
  def __init__(self, graph):
    self.graph = graph

    tmp_analysis = complete_parse(graph)

    self.property_set = tmp_analysis[0]
    self.property_count_dict = tmp_analysis[1] 
    self.entity_set = tmp_analysis[2]
    self.entity_type_dict = tmp_analysis[3]
    self.entity_prop_dict = tmp_analysis[4]
    self.all_types = tmp_analysis[5]
    self.all_types_dict = tmp_analysis[6]

  def get_total_instances(self):
    total_instances = len(self.entity_set)
    return total_instances

  def print_length(self):
    print(len(self.graph))
  
  def missingness_ratio(self):
    total_entities = len(self.entity_set)
    total_properties = len(self.property_set)
    total_triples = 0 
    
    for key, value in self.property_count_dict.items():
      if value > total_entities:
        total_triples = total_triples + total_entities
      else: 
        total_triples = total_triples + value
    
    missingness = 1 - ((total_triples)/(total_entities * total_properties))

    return missingness
  
  def best_label(self):
    best_label = 'None' 
    if bool(self.all_types_dict):
      best_label = max(self.all_types_dict, key=lambda k: self.all_types_dict[k])

    return best_label 
  
  def most_common_properties(self, n=1):
    keys_with_top_n_values = 'None' 
    if bool(self.property_count_dict):
      keys_with_top_n_values = sorted(self.property_count_dict, 
                                    key=lambda k: self.property_count_dict[k], 
                                    reverse=True)[:n]
    
    return keys_with_top_n_values


  # function to get simple property subset hierarchy 
  def simple_subset_hierarchy(self):
    property_types = extract_property_types(self.entity_prop_dict)
    one_hot_matrix = generate_one_hot_matrix(property_types[1], self.property_set)
    subset_dict = find_subset_property_sets(one_hot_matrix)
    num_clusters = len(property_types[1].keys())
    dummy_label_list = [index for index in range(num_clusters)]
    graph_dict = extract_cluster_dictionaries_mod(property_types[0], dummy_label_list)
    graph_list = extract_cluster_graphs(graph_dict, self.graph, num_clusters)
    hierarchy_network = generate_simple_subset_hierarchy(subset_dict, num_clusters)
    final_hierarchy = generate_hierarchy(graph_list, hierarchy_network)

    return final_hierarchy
    
  def DBSCAN(self, eps=0.8, min_samples=1, metric='jaccard'):
    property_types = extract_property_types(self.entity_prop_dict)
    one_hot_matrix = generate_one_hot_matrix(property_types[1], self.property_set)
    weight_matrix = np.array([len(lst) for lst in list(property_types[0].values())])
    hierarchy_network, leaf_clusters_len = cluster_with_DBSCAN(one_hot_matrix, eps = eps, min_samples = min_samples, metric=metric)
    num_clusters = max(list(hierarchy_network.labels_))
    # for every label in the hierarchy network we need assign its index to the key of the corresponding cluster
    cluster_cs_dict = dict()
    for i in range(len(hierarchy_network.labels_)):
      if hierarchy_network.labels_[i] in cluster_cs_dict:
        cluster_cs_dict[hierarchy_network.labels_[i]].append(i)
      else:
        cluster_cs_dict[hierarchy_network.labels_[i]] = [i]
    graph_dict = extract_cluster_dictionaries_mod(property_types[0], hierarchy_network.labels_)
    graph_list = extract_cluster_graphs(graph_dict, self.graph, num_clusters)
    hierarchy_network = generate_plain_subset_hierarchy(num_clusters)
    final_hierarchy = generate_hierarchy(graph_list, hierarchy_network, one_hot_matrix, weight_matrix, cluster_cs_dict)

    return final_hierarchy

  def HDBSCAN(self):
    property_types = extract_property_types(self.entity_prop_dict)
    one_hot_matrix = generate_one_hot_matrix(property_types[1], self.property_set)
    hierarchy_network, leaf_clusters_len = cluster_with_HDBSCAN(one_hot_matrix)
    dummy_label_list = [index for index in range(leaf_clusters_len)]
    cluster_dicts = extract_cluster_dictionaries_mod(property_types[0], dummy_label_list)
    cluster_graphs = extract_cluster_graphs(cluster_dicts, nobel_prize_graph, leaf_clusters_len)
    cluster_hierarchy = extract_cluster_graphs_hierarchical(cluster_graphs, hierarchy_network)
    final_hierarchy_obj = generate_hierarchy(cluster_hierarchy, hierarchy_network)

    return final_hierarchy_obj
  
  def frequent_itemset_dummy_total(self, min_support=0.2):
    property_types = extract_property_types(self.entity_prop_dict)
    one_hot_matrix = generate_one_hot_matrix(property_types[1], self.property_set)
    # weight this based on the instances of each class
    weight_matrix = np.array([len(lst) for lst in list(property_types[0].values())])
    weighted_one_hot_matrix = one_hot_matrix * weight_matrix[:, np.newaxis]
    
    weighted_list_of_lists = weighted_matrix_to_list_of_lists(weighted_one_hot_matrix)        
    multiplied_list_of_lists = multiply_list_of_lists(weighted_list_of_lists, weight_matrix)

    # generate frequent itemsets 
    data_df = generate_frequent_itemset(multiplied_list_of_lists)
    frequent_itemsets = apriori(data_df, min_support, use_colnames=True)
    # generate cs from frequent itemsets
    cs_frequent_matrix, support_list = generate_cs_from_itemsets(frequent_itemsets, one_hot_matrix.shape[1])
    # generate initial global clusters
    cs_global_freq_dict = generate_initial_global_clusters(cs_frequent_matrix, weighted_one_hot_matrix)
    # generate exclusive clusters
    exclusive_dict, excluded = generate_exclusive_clusters_dummy(cs_global_freq_dict, weighted_one_hot_matrix.shape[0])
    # reverse and normalize the dict
    normalized_dict = reverse_and_normalize_dict(exclusive_dict, excluded)
    num_clusters = max(list(normalized_dict.keys()))

    # get labels 
    label_list = get_labels_from_dict(normalized_dict)

    graph_dict = extract_cluster_dictionaries_mod(property_types[0], label_list)
    graph_list = extract_cluster_graphs(graph_dict, self.graph, num_clusters)
    hierarchy_network = generate_plain_subset_hierarchy(num_clusters)
    final_hierarchy = generate_hierarchy(graph_list, hierarchy_network, one_hot_matrix, weight_matrix, normalized_dict)


    return final_hierarchy


class Hierarchy: 
  def __init__(self, structure_network, graph_list, one_hot_matrix, weight_matrix, cluster_cs_dict):
    self.structure_network = structure_network
    self.graph_list = graph_list
    self.one_hot_matrix = one_hot_matrix 
    self.weight_matrix = weight_matrix 
    self.cluster_cs_dict = cluster_cs_dict

  def get_depth(self):
    # Get the heights of the nodes in the hierarchy and the total height
    # As root we take node -1 
    # find the shortest path length from the root node to all other nodes
    path_lengths = nx.single_source_shortest_path_length(self.structure_network, -1)

    # find the maximum path length, which is the height of the tree
    depth = max(path_lengths.values())

    return depth, path_lengths
  
  def get_classes(self):
    # classes are equal to the length of the graph list
    classes_num = len(self.graph_list)
    return classes_num


  def metadata_graph(self):
    metadata_graph = Graph()
    namespace = URIRef("http://explainer.ontology/")
    metadata_graph.bind("ex", namespace)
    map_dict = dict()
    map_dict[-1] = URIRef(namespace + "ROOT")
    for i in range(len(self.graph_list)):
      tmp_type = self.graph_list[i].best_label()
      tmp_properties = self.graph_list[i].most_common_properties(n=3)

      if not isinstance(tmp_type, list):
        tmp_type = [tmp_type]
      if not isinstance(tmp_properties, list):
        tmp_properties = [tmp_properties]
      
      tmp_type = [remove_namespace(string) for string in tmp_type]
      tmp_properties = [remove_namespace(string) for string in tmp_properties]

      combined_list = tmp_type + tmp_properties
      tmp_name = "-GAP-".join(combined_list)
      tmp_class_uri = URIRef(namespace + tmp_name)
      map_dict[i] = tmp_class_uri
      metadata_graph.add((tmp_class_uri, RDF.type, RDFS.Class))
    
    for (i,j) in self.structure_network.edges:
      subclass_uri = map_dict[j]
      superclass_uri = map_dict[i]
      metadata_graph.add((subclass_uri, RDFS.subClassOf, superclass_uri))
    
    return metadata_graph
  
  def get_description_dict(self):
    description_dict = {}

    # get the one hot property representation of all clusters
    one_hot_matrix = self.one_hot_property_representation()
    #get the coordinates on 2d space of the clusters based on the one hot matrix
    xs, ys, mock_labels = plot_2D_with_MDS(one_hot_matrix)

    #store the coordinates in a seperate entry in the description dict
    description_dict["coordinates"] = {"x": xs.tolist(), "y": ys.tolist(), "labels": mock_labels}
    
    clustering_metrics = get_clustering_metrics(self.one_hot_matrix, self.cluster_cs_dict)

    #store general info regarding this clustering 
    description_dict["general_info"] = {
        'total_classes': self.get_classes(),
        'depth': self.get_depth()[0],
        'clustering_method': 'DBSCAN',
        'similarity_method': 'jaccard',
        'silhouette_score': clustering_metrics['silhouette_score'],
        'calinski_harabasz_score': clustering_metrics['calinski_harabasz_score'],
        'davies_bouldin_score': clustering_metrics['davies_bouldin_score'],

        # 'missingness_ratio': int(missingness_ratio_retrieve(self.one_hot_matrix, self.cluster_cs_dict ,self.weight_matrix)),
    }

    tmp_missing_dict = missingness_ratio_retrieve(self.one_hot_matrix, self.cluster_cs_dict ,self.weight_matrix)
    
    instance_dict = {}

    for key in self.cluster_cs_dict.keys():
        instance_dict[key] = 0
        for val in self.cluster_cs_dict[key]:
            instance_dict[key] = instance_dict[key] + self.weight_matrix[val] 

    for i in range(len(self.graph_list)):
      class_name = f"Cluster_{i}"
      total_instances = int(instance_dict[i])
      dominant_type = self.graph_list[i].best_label()
      most_common_properties = self.graph_list[i].most_common_properties(n=3)
      depth, path_lengths = self.get_depth()

      # Convert the type and property URIs to strings without the namespace
      if isinstance(dominant_type, list):
        dominant_type = [remove_namespace(string) for string in dominant_type]
      else:
        dominant_type = remove_namespace(dominant_type)

      if isinstance(most_common_properties, list):
        most_common_properties = [remove_namespace(string) for string in most_common_properties]
      else:
        most_common_properties = remove_namespace(most_common_properties)

      description_dict[class_name] = {
          "total_instances": total_instances,
          "dominant_type": dominant_type,
          "most_common_properties": most_common_properties,
          "depth": path_lengths[i],
          "missingness_ratio_retrieve": tmp_missing_dict[i]
      }

    return description_dict
    
  def get_description_json(self, title="data_desc.json"):
    description_dict = self.get_description_dict()
    with open(title, 'w') as fp:
      json.dump(description_dict, fp, indent=4)
    return 1

  
  def one_hot_property_representation(self):
    total_property_set = set()
    for i in self.graph_list:
      total_property_set |= i.property_set
    
    total_property_dict = {key: [] for key in total_property_set}
    
    for i in range(len(self.graph_list)):
      for val in list(self.graph_list[i].property_set):
        total_property_dict[val].append(i)
    
    total_property_dict_ind = {index: value for index, value in enumerate(total_property_dict.values())}
    
    # THIS CAN DEFINETLY BECOME FASTER 

    max_row = len(self.graph_list)
    max_col = len(total_property_set)

    one_hot_array = np.zeros((max_row, max_col))

    for col_id, row_ids in total_property_dict_ind.items():
      one_hot_array[row_ids, col_id] = 1
    
    return one_hot_array

# Functions to use for the API

def cluster_api_DBSCAN(filename, eps, min_samples, metric):
    # load graph from rdf file
    graph = Graph()
    graph.parse(filename)
    # cluster with DBSCAN
    clustering = Clustering(graph)
    final_hierarchy = clustering.DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    # return the final hierarchy
    return final_hierarchy

def cluster_api_frequent_itemsets(filename, min_support, min_similarity):
    # load graph from rdf file
    graph = Graph()
    graph.parse(filename)
    # cluster with DBSCAN
    clustering = Clustering(graph)
    final_hierarchy = clustering.frequent_itemset_dummy_total(min_support=min_support)
    # return the final hierarchy
    return final_hierarchy

def generate_html_hierarchy(input_file, output_dir):
    # Run the 'ontospy scan' command with the argument
    result = subprocess.run(['ontospy', 'scan', file_path], stdout=subprocess.PIPE)

def plot_2D_with_MDS(one_hot_matrix):
    # convert the one-hot-matrix of properties to dissimilarity matrix
    dissimilarity_matrix = 1 - cosine_similarity(one_hot_matrix)
    # apply MDS to the dissimilarity matrix
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(dissimilarity_matrix)  # shape (n_components, n_samples)
    xs, ys = pos[:, 0], pos[:, 1]
    # plot the 2D coordinates
    mock_labels = list(range(len(xs)))
    # plot the 2D coordinates
    '''
    plt.scatter(xs, ys)
    for x, y, label in zip(xs, ys, mock_labels):
        plt.annotate(label, (x, y))
    plt.show()
    '''
    return xs, ys, mock_labels

def sankey_generator(dict_1, arr_1, weights_1, dict_2, arr_2, weights_2):
    # check if the property arrays have the same length
    label, source, target, value = [], [], [], [] 
    # fill the label with string equal to "Cluster_1_" + str(i) for i in range dict_1.keys() 
    label = ["Cluster_1_" + str(i) for i in range(len(dict_1.keys()))]
    # fill the label with string equal to "Cluster_2_" + str(i) for i in range dict_2.keys()
    label += ["Cluster_2_" + str(i) for i in range(len(dict_2.keys()))]

    new_dict_2 = {key + len(dict_1.keys()): value for key, value in dict_2.items()}

    if (arr_1.all() != arr_2.all()):
        return None

    # if they do add the weights to the arrays 
    common_pairs = []
    for key1, values1 in dict_1.items():
        for key2, values2 in new_dict_2.items():
            common_values = set(values1) & set(values2)
            if common_values:
                common_pairs.append((key1, key2, common_values))
    
    for pair in common_pairs:
        source.append(pair[0])
        target.append(pair[1])
        tmp_val = 0 
        for i in pair[2]:
            tmp_val += weights_1[i]
        value.append(tmp_val)
    
    return label, source, target, value

def sankey_generator(dict_1, arr_1, weights_1, dict_2, arr_2, weights_2):
    # check if the property arrays have the same length
    label, source, target, value = [], [], [], [] 
    # fill the label with string equal to "Cluster_1_" + str(i) for i in range dict_1.keys() 
    label = ["Cluster_1_" + str(i) for i in range(len(dict_1.keys()))]
    # fill the label with string equal to "Cluster_2_" + str(i) for i in range dict_2.keys()
    label += ["Cluster_2_" + str(i) for i in range(len(dict_2.keys()))]

    new_dict_2 = {key + len(dict_1.keys()): value for key, value in dict_2.items()}

    if (arr_1.all() != arr_2.all()):
        return None
    
    # Initialize a dictionary to store common values
    common_values = {}

    # Iterate through the keys of the first dictionary
    for key1, values1 in dict_1.items():
        # Iterate through the keys of the second dictionary
        for key2, values2 in new_dict_2.items():
            # Find the common values by using set intersection
            common = list(set(values1) & set(values2))
            if common:
                # Store the common values in the common_values dictionary
                common_values[(key1, key2)] = common
    
    # Iterate through the common values dictionary
    for (key1, key2), values in common_values.items():
        # Iterate through the common values
        source.append(key1)
        target.append(key2)
        tmp_val = 0
        for v in values:
            # Append the source, target and value lists
            tmp_val += weights_1[v]
        value.append(tmp_val)
    
    return label, source, target, value

def missingness_ratio_retrieve(one_hot_matrix, cluster_cs_dict, weight_matrix):
    missingness_dict = {}
    for key, value in cluster_cs_dict.items():
        selected_rows = one_hot_matrix[value]
        new_row = np.logical_or.reduce(selected_rows, axis=0)
        tmp_missing = 0
        tmp_total = np.sum(new_row)*np.sum(weight_matrix[value])
        for i in value:
            tmp_missing += np.sum(new_row-one_hot_matrix[i])*weight_matrix[i]
        missingness_dict[key] = tmp_missing/tmp_total

    return missingness_dict

def get_labels_from_dict(cluster_cs_dict):
    # transform cluster_cs_dict to a list where 
    # in the value index of the list you find the key 
    # of the dictionary
    cs_cluster_dict = {}
    for key, value in cluster_cs_dict.items():
        for i in value:
            cs_cluster_dict[i] = key
    #order dict based on keys 
    ordered_cs_cluster_dict = {key: cs_cluster_dict[key] for key in sorted(cs_cluster_dict.keys())}
    # get a list of the values 
    label_list = list(ordered_cs_cluster_dict.values())

    return label_list

def get_comparison_metrics(one_hot_matrix_1, cluster_cs_dict_1, one_hot_matrix_2, cluster_cs_dict_2):
    # get labels from dict 
    label_list_1 = get_labels_from_dict(cluster_cs_dict_1)
    label_list_2 = get_labels_from_dict(cluster_cs_dict_2)

    # get ami score
    ami = adjusted_mutual_info_score(label_list_1, label_list_2)
    # get ari score
    ari = adjusted_rand_score(label_list_1, label_list_2)

    return ami, ari


def get_clustering_metrics(one_hot_matrix, cluster_cs_dict):
    # get labels from dict 
    label_list = get_labels_from_dict(cluster_cs_dict)
    result_dict = {}

    # get silhouette score
    silhouette_sc = silhouette_score(one_hot_matrix, label_list, metric='jaccard')
    result_dict['silhouette_score'] = silhouette_sc
    # get calinski harabasz score
    calinski_harabasz_sc = calinski_harabasz_score(one_hot_matrix, label_list)
    result_dict['calinski_harabasz_score'] = calinski_harabasz_sc
    # get davies bouldin score
    davies_bouldin_sc = davies_bouldin_score(one_hot_matrix, label_list)
    result_dict['davies_bouldin_score'] = davies_bouldin_sc

    return result_dict

def get_comparison_dict(one_hot_matrix_1, cluster_cs_dict_1, weight_matrix_1,  one_hot_matrix_2, cluster_cs_dict_2, weight_matrix_2):
    result_dict = {}
    # get comparison metrics
    ami, ari = get_comparison_metrics(one_hot_matrix_1, cluster_cs_dict_1, one_hot_matrix_2, cluster_cs_dict_2)
    result_dict['ami'] = ami
    result_dict['ari'] = ari

    # get sankey diagram
    label, source, target, value = sankey_generator(cluster_cs_dict_1, one_hot_matrix_1, weight_matrix_1, cluster_cs_dict_2, one_hot_matrix_2, weight_matrix_2)
    source = [int(i) for i in source]
    target = [int(i) for i in target]
    value = [int(i) for i in value]
    result_dict['sankey'] = {'label': label, 'source': source, 'target': target, 'value': value}

    return result_dict

def get_comparison_json(one_hot_matrix_1, cluster_cs_dict_1, weight_matrix_1,  one_hot_matrix_2, cluster_cs_dict_2, weight_matrix_2, title="comparison.json"):
    comparison_dict = get_comparison_dict(one_hot_matrix_1, cluster_cs_dict_1, weight_matrix_1,  one_hot_matrix_2, cluster_cs_dict_2, weight_matrix_2)
    with open(title, 'w') as fp:
      json.dump(comparison_dict, fp, indent=4)
    return 1


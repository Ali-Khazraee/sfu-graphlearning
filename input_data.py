import torch
import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import zipfile
import math
from  Synthatic_graph_generator import Synthetic_data
from scipy.sparse import csr_matrix
from torch_geometric.datasets import IMDB
from torch_geometric.datasets import Amazon, Planetoid
import torch.nn.functional as F
from utils import * 


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(args):
    """the method will return adjacency matrix, node features, nodes label, edges label, and circles.
      None in case the dataset does not come with the information"""

    # Define the list of heterogeneous datasets


    if args.graph_type== "heterogeneous":
        if args.dataset == "acm-multi":
            adj, features, labels, edge_labels, circles, mapping_details, important_feat_ids, feats_for_reconstruction, one_hot_labels = acm_multi()
        elif args.dataset == "imdb-multi":
            adj, features, labels, edge_labels, circles, mapping_details, important_feat_ids, feats_for_reconstruction, one_hot_labels = imdb_multi()

        # Process features for reconstruction
        feats_for_reconstruction_count = {}
        for node_type, (start_idx, end_idx) in mapping_details['node_type_to_index_map'].items():
            tensor_slice = torch.tensor(feats_for_reconstruction[start_idx:end_idx], dtype=torch.float32).to(args.device)
            feats_for_reconstruction_count[node_type] = tensor_slice
    else:
        if args.dataset == "cora":
            adj, features, labels, edge_labels, circles, mapping_details, important_feat_ids, feats_for_reconstruction, one_hot_labels = cora()
        elif args.dataset == "citeseer":
            adj, features, labels, edge_labels, circles, mapping_details, important_feat_ids, feats_for_reconstruction, one_hot_labels = citeseer()
        elif args.dataset == "acm":
            adj, features, labels, edge_labels, circles, mapping_details, important_feat_ids, feats_for_reconstruction, one_hot_labels = acm_homogenized()

        feats_for_reconstruction_count = torch.tensor(feats_for_reconstruction).to(args.device)

    if (type(features) == np.ndarray):
        features = torch.tensor(features, dtype=torch.float32)
    else:
        features = torch.tensor(features.todense(), dtype=torch.float32)

    return adj, features, labels, edge_labels, circles, mapping_details, important_feat_ids, feats_for_reconstruction ,feats_for_reconstruction_count, one_hot_labels



def imdb_multi():

    dataset = IMDB("data")
    heterodata = dataset[0]
    labels = heterodata['movie']['y']

    num_nodes = sum(heterodata[node_type].num_nodes for node_type in heterodata.node_types)
    adj = csr_matrix((num_nodes, num_nodes), dtype=int)
    edge_labels = csr_matrix((num_nodes, num_nodes), dtype=int)

    node_type_to_index_map = {}
    current_index = 0
    for node_type in heterodata.node_types:
        node_count = heterodata[node_type].num_nodes
        node_type_to_index_map[node_type] = (current_index, current_index + node_count)
        current_index += node_count

    node_labels = np.zeros(num_nodes, dtype=int)
    for node_type, (start, end) in node_type_to_index_map.items():
        node_labels[start:end] = list(heterodata.node_types).index(node_type)

    edge_type_encoding = {}
    counter = 1
    for edge_type in heterodata.edge_types:
        simplified_edge_type = tuple(sorted([edge_type[0], edge_type[2]]))
        if simplified_edge_type not in edge_type_encoding:
            edge_type_encoding[simplified_edge_type] = counter
            counter += 1

    for edge_type in heterodata.edge_types:
        edge_index = heterodata[edge_type].edge_index.numpy()
        src_indices_global = edge_index[0] + node_type_to_index_map[edge_type[0]][0]
        dst_indices_global = edge_index[1] + node_type_to_index_map[edge_type[2]][0]
        simplified_edge_type = tuple(sorted([edge_type[0], edge_type[2]]))
        edge_code = edge_type_encoding[simplified_edge_type]
        adj[src_indices_global, dst_indices_global] = 1
        edge_labels[src_indices_global, dst_indices_global] = edge_code

    features = np.vstack([heterodata[node_type].x.numpy() for node_type in heterodata.node_types])
   
    
    circles = None

    mapping_details = {
        'node_type_to_index_map': node_type_to_index_map,
        'edge_type_encoding': edge_type_encoding,
    }



    features_with_labels = np.array(features[:mapping_details['node_type_to_index_map']['movie'][1]])
    features_binary = np.where(features_with_labels >= 1, 1, 0)

    _, important_feat_ids = reduce_node_features(features_binary, labels, random_seed = 0)
    important_feats = features[:, important_feat_ids]
    feats_for_reconstruction = np.where(important_feats >= 1, 1, 0)
    features = csr_matrix(features)
    num_classes = labels.unique().size(0)
    one_hot_labels = F.one_hot(labels, num_classes)
    return adj, features, labels, edge_labels, circles, mapping_details, important_feat_ids, feats_for_reconstruction , one_hot_labels

def acm_multi():
    
    heterodata = torch.load('/home/majid/sfu-graphlearning-master/acm_multi/multi_acm.pt')
    labels = heterodata['paper']['y']
    
    num_nodes = sum(heterodata[node_type].num_nodes for node_type in heterodata.node_types)
    adj = csr_matrix((num_nodes, num_nodes), dtype=int)
    edge_labels = csr_matrix((num_nodes, num_nodes), dtype=int)

    node_type_to_index_map = {}
    current_index = 0
    for node_type in heterodata.node_types:
        node_count = heterodata[node_type].num_nodes
        node_type_to_index_map[node_type] = (current_index, current_index + node_count)
        current_index += node_count

    node_labels = np.zeros(num_nodes, dtype=int)
    for node_type, (start, end) in node_type_to_index_map.items():
        node_labels[start:end] = list(heterodata.node_types).index(node_type)

    edge_type_encoding = {}
    counter = 1
    for edge_type in heterodata.edge_types:
        simplified_edge_type = tuple(sorted([edge_type[0], edge_type[2]]))
        if simplified_edge_type not in edge_type_encoding:
            edge_type_encoding[simplified_edge_type] = counter
            counter += 1

    for edge_type in heterodata.edge_types:
        edge_index = heterodata[edge_type].edge_index.numpy()
        src_indices_global = edge_index[0] + node_type_to_index_map[edge_type[0]][0]
        dst_indices_global = edge_index[1] + node_type_to_index_map[edge_type[2]][0]
        simplified_edge_type = tuple(sorted([edge_type[0], edge_type[2]]))
        edge_code = edge_type_encoding[simplified_edge_type]
        adj[src_indices_global, dst_indices_global] = 1
        edge_labels[src_indices_global, dst_indices_global] = edge_code

    features = np.vstack([heterodata[node_type].x.numpy() for node_type in heterodata.node_types])
   
    
    circles = None

    mapping_details = {
        'node_type_to_index_map': node_type_to_index_map,
        'edge_type_encoding': edge_type_encoding,
    }



    features_with_labels = np.array(features[:mapping_details['node_type_to_index_map']['paper'][1]])
    features_binary = np.where(features_with_labels >= 1, 1, 0)
    _, important_feat_ids = reduce_node_features(features_binary, labels, random_seed = 0)
    important_feats = features[:, important_feat_ids]
    feats_for_reconstruction = np.where(important_feats >= 1, 1, 0)
    features = csr_matrix(features)
    num_classes = labels.unique().size(0)
    one_hot_labels = F.one_hot(labels, num_classes)
    return adj, features, labels, edge_labels, circles, mapping_details, important_feat_ids, feats_for_reconstruction , one_hot_labels




def acm_homogenized():
    # Load the dataset from the specified directory
    ds = torch.load("D:/acm/acm.pt")
    ds['y'] = torch.tensor(ds['y'])  # Ensure labels are tensors
    
    num_nodes = ds['y'].shape[0]
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    # Iterate over the edge index and fill in the adjacency matrix
    edge_index = ds['edge_index']
    for i in range(edge_index.shape[1]):
        start_node = edge_index[0, i].item()
        end_node = edge_index[1, i].item()
        adjacency_matrix[start_node, end_node] = 1
        adjacency_matrix[end_node, start_node] = 1  # For undirected graph
    
    features = ds['x'].numpy()
    label = ds['y']
    _, important_feat_ids = reduce_node_features(np.array(features), label, 0)
    important_feats = features[:, important_feat_ids]
    feats_for_reconstruction = np.where(important_feats >= 1, 1, 0)
    features = csr_matrix(features)

    # Calculate number of classes and generate one-hot labels
    num_classes = label.unique().size(0)
    one_hot_labels = F.one_hot(label, num_classes)
    
    return csr_matrix(adjacency_matrix), features, label, csr_matrix(adjacency_matrix), None, None, important_feat_ids, feats_for_reconstruction, one_hot_labels

    
        
def citeseer():
    ds = Planetoid("data", "citeseer")[0]
    num_nodes = ds['y'].shape[0]
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    # Iterate over the edge index and fill in the adjacency matrix
    edge_index = ds['edge_index']
    for i in range(edge_index.shape[1]):
        start_node = edge_index[0, i].item()
        end_node = edge_index[1, i].item()
        adjacency_matrix[start_node, end_node] = 1
        adjacency_matrix[end_node, start_node] = 1  # For und
    
    features = ds['x'].numpy()
    label = ds['y']
    _, important_feat_ids = reduce_node_features(np.array(features), label, 0)
    important_feats = features[:, important_feat_ids]
    feats_for_reconstruction = np.where(important_feats >= 1, 1, 0)
    features = csr_matrix(features)
    num_classes = label.unique().size(0)
    one_hot_labels = F.one_hot(label, num_classes)
    return csr_matrix(adjacency_matrix),features, label, csr_matrix(adjacency_matrix), None, None, important_feat_ids, feats_for_reconstruction, one_hot_labels


def cora():
    ds = Planetoid("\data", "cora")[0]
    num_nodes = ds['y'].shape[0]
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    # Iterate over the edge index and fill in the adjacency matrix
    edge_index = ds['edge_index']
    for i in range(edge_index.shape[1]):
        start_node = edge_index[0, i].item()
        end_node = edge_index[1, i].item()
        adjacency_matrix[start_node, end_node] = 1
        adjacency_matrix[end_node, start_node] = 1  # For und
    
    features = ds['x'].numpy()
    label = ds['y']
    _, important_feat_ids = reduce_node_features(np.array(features), label, 0)
    important_feats = features[:, important_feat_ids]
    feats_for_reconstruction = np.where(important_feats >= 1, 1, 0)
    features = csr_matrix(features)
    num_classes = label.unique().size(0)
    one_hot_labels = F.one_hot(label, num_classes)
    return csr_matrix(adjacency_matrix),features, label, csr_matrix(adjacency_matrix), None, None, important_feat_ids, feats_for_reconstruction, one_hot_labels

    



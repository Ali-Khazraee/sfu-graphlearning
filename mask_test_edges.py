import numpy as np
import dgl
import pylab as p
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix,average_precision_score
from scipy.sparse import csr_matrix
import torch
from utils import *

def roc_auc_estimator_onGraphList(pos_edges, negative_edges, reconstructed_adj, origianl_agjacency):
    prediction = []
    true_label = []
    for i,_ in enumerate(reconstructed_adj):
        for edge in pos_edges[i]:
            prediction.append(reconstructed_adj[i][edge[0],edge[1]])
            true_label.append(origianl_agjacency[i][edge[0], edge[1]])

        for edge in negative_edges[i]:
            prediction.append(reconstructed_adj[i][edge[0], edge[1]])
            true_label.append(origianl_agjacency[i][edge[0], edge[1]])

    pred = [1 if x>.5 else 0 for x in prediction]
    auc = roc_auc_score(y_score= prediction, y_true= true_label)
    acc = accuracy_score(y_pred= pred, y_true= true_label, normalize= True)
    ap=average_precision_score(y_score= prediction, y_true= true_label)
    cof_mtx = confusion_matrix(y_true=true_label, y_pred=pred)
    return auc , acc,ap, cof_mtx

def roc_auc_estimator(pos_edges, negative_edges, reconstructed_adj, origianl_agjacency):
    prediction = []
    true_label = []
    if type(pos_edges) == list or type(pos_edges) ==np.ndarray:
        for edge in pos_edges:
            prediction.append(reconstructed_adj[edge[0],edge[1]])
            true_label.append(origianl_agjacency[edge[0], edge[1]])

        for edge in negative_edges:
            prediction.append(reconstructed_adj[edge[0], edge[1]])
            true_label.append(origianl_agjacency[edge[0], edge[1]])
    else:
        prediction = list(reconstructed_adj.reshape(-1))
        true_label = list(np.array(origianl_agjacency.todense()).reshape(-1))
    pred = np.array(prediction)
    pred[pred>.5] = 1
    pred[pred < .5] = 0
    pred = pred.astype(int)
    # pred = [1 if x>.5 else 0 for x in prediction]

    auc = roc_auc_score(y_score= prediction, y_true= true_label)
    acc = accuracy_score(y_pred= pred, y_true= true_label, normalize= True)
    ap=average_precision_score(y_score= prediction, y_true= true_label)
    cof_mtx = confusion_matrix(y_true=true_label, y_pred=pred)
    return auc , acc,ap, cof_mtx

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape



def mask_test_edges_new(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    # assert np.diag(adj.todense()).sum() == 0
    assert adj.diagonal().sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    ######
    sorted_arr = np.sort(edges_all, axis=1)
    sorted_tuples = [tuple(row) for row in sorted_arr]
    unique_tuples = np.unique(sorted_tuples, axis=0)
    edges = np.array(unique_tuples)
    edges_all = edges
    ######
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    
    # add reverse of every edge as well
    train_edges = np.vstack((train_edges, train_edges[:, ::-1]))
    val_edges = np.vstack((val_edges, val_edges[:, ::-1]))
    test_edges = np.vstack((test_edges, test_edges[:, ::-1]))
    

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)
    
    
    assert ~ismember(test_edges.tolist(), train_edges)
    assert ~ismember(val_edges.tolist(), train_edges)
    assert ~ismember(val_edges.tolist(), test_edges)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue

        test_edges_false.append([idx_i, idx_j])
        test_edges_false.append([idx_j, idx_i])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_i, idx_j], np.array(test_edges_false)):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])
        val_edges_false.append([idx_j, idx_i])

    train_edges_false = []
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_i, idx_j], np.array(val_edges_false)):
            continue
        if ismember([idx_i, idx_j], np.array(test_edges_false)):
            continue
        if train_edges_false:
            if ismember([idx_j, idx_i], np.array(train_edges_false)):
                continue
        train_edges_false.append([idx_i, idx_j])
        train_edges_false.append([idx_j, idx_i])
    # print(test_edges_false)
    # print(val_edges_false)
    # print(test_edges)
    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # -----------------------------------------------------------------
    ## the test and val edges wont effect reconstruction loss
    # ignore_edges_inx = [list(np.array(val_edges_false)[:,0]),list(np.array(val_edges_false)[:,1])]
    # ignore_edges_inx[0].extend(val_edges[:,0])
    # ignore_edges_inx[1].extend(val_edges[:,1])
    # import copy
    #
    # val_edge_idx = copy.deepcopy(ignore_edges_inx)
    # ignore_edges_inx[0].extend(test_edges[:, 0])
    # ignore_edges_inx[1].extend(test_edges[:, 1])
    # ignore_edges_inx[0].extend(np.array(test_edges_false)[:, 0])
    # ignore_edges_inx[1].extend(np.array(test_edges_false)[:, 1])

    #-----------------------------------------------------------------
    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, train_edges, train_edges_false,ignore_edges_inx, val_edge_idx


def mask_test_edges(adj, ignore_val_test_edges=False, ignore_self_loop=True):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    # assert np.diag(adj.todense()).sum() == 0
    assert adj.diagonal().sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    index = list(range(train_edges.shape[0]))
    np.random.shuffle(index)
    train_edges_true = train_edges[index[0:num_val]]

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue

        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_i, idx_j], np.array(test_edges_false)):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    train_edges_false = []
    while len(train_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_i, idx_j], np.array(val_edges_false)):
            continue
        if ismember([idx_i, idx_j], np.array(test_edges_false)):
            continue
        if train_edges_false:
            if ismember([idx_j, idx_i], np.array(train_edges_false)):
                continue
        train_edges_false.append([idx_i, idx_j])
    # print(test_edges_false)
    # print(val_edges_false)
    # print(test_edges)
    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T



    val_edge_idx = [list(np.array(val_edges_false)[:,0]),list(np.array(val_edges_false)[:,1])]
    val_edge_idx[0].extend(val_edges[:,0])
    val_edge_idx[1].extend(val_edges[:,1])
    import copy

    # -----------------------------------------------------------------
    ## the test and val edges wont effect reconstruction loss
    ignore_edges_inx= None
    if ignore_val_test_edges != False:
        ignore_edges_inx = copy.deepcopy(val_edge_idx)
        ignore_edges_inx[0].extend(test_edges[:, 0])
        ignore_edges_inx[1].extend(test_edges[:, 1])
        ignore_edges_inx[0].extend(np.array(test_edges_false)[:, 0])
        ignore_edges_inx[1].extend(np.array(test_edges_false)[:, 1])
    # -----------------------------------------------------------------

    if ignore_self_loop==True:
        if not ignore_edges_inx:
            ignore_edges_inx=[[],[]]
        ignore_edges_inx[0].extend([i for i in range(adj.shape[0])])
        ignore_edges_inx[1].extend([i for i in range(adj.shape[0])])
    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, list(train_edges_true), train_edges_false,ignore_edges_inx, val_edge_idx






def train_test_split(args, original_adj, node_label):

    #add 0.2 to function mask_test_edges 
    

    if args.split_the_data_to_train_test and args.task == "link_prediction":
        adj_train, _, val_edges_poitive, val_edges_negative, test_edges_positive, test_edges_negative, train_edges_positive, train_edges_negative, ignore_edges_inx, val_edge_idx = mask_test_edges(original_adj)
        ignored_edges = []
        gt_labels, masked_indexes = mask_labels(node_label, 0)
    if args.split_the_data_to_train_test and args.task == "node_classification" :
        adj_train, _, val_edges_poitive, val_edges_negative, test_edges_positive, test_edges_negative, train_edges_positive, train_edges_negative, ignore_edges_inx, val_edge_idx = mask_test_edges(original_adj)
        adj_train = original_adj
        gt_labels, masked_indexes = mask_labels(node_label, 0.2)
        ignored_edges = None

    num_nodes = adj_train.shape[-1]

    return adj_train, val_edges_poitive, val_edges_negative, test_edges_positive, test_edges_negative, train_edges_positive, train_edges_negative, ignore_edges_inx, val_edge_idx, gt_labels, ignored_edges, num_nodes, masked_indexes



import copy
import numpy as np

def mask_labels(node_label, mask_ratio):
    """
    Masks a given percentage of labels by setting them to -1.
    
    :param node_label: A numpy array of node labels to be masked
    :param mask_ratio: The ratio of labels to mask (default is 20%)
    :return: A copy of the original labels with some labels masked
    """
    gt_labels = copy.deepcopy(node_label)
    num_to_mask = round(gt_labels.shape[0] * mask_ratio)
    
    masked_indexes = np.random.choice(gt_labels.shape[0], num_to_mask, replace=False)
    
    gt_labels[masked_indexes] = -1
    
    return gt_labels , masked_indexes





def create_dgl_graph(hemogenized, edge_labels, adj_train, val_edges_poitive, val_edges_negative, test_edges_positive, test_edges_negative):
    num_obs = 1  # number of relateion; default is hemogenous dataset with one type of edge
    if hemogenized != True:
        edge_relType_train = edge_labels.multiply(adj_train)
        rel_type = np.unique(edge_labels.data)
        num_obs = len(rel_type)  # number of relateion; heterougenous setting
    # edge_relType = edge_relType + sp.eye(adj_train.shape[-1]) * (len(np.unique(edge_relType.data)) + 1)
        graph_dgl = []

        train_matrix = []
        pre_self_loop_train_adj = []
        for rel_num in rel_type:
            tm_mtrix = csr_matrix(edge_relType_train.shape)
            tm_mtrix[edge_relType_train == (rel_num)] = 1
            pre_self_loop_train_adj.append(tm_mtrix.todense())
            tr_matrix = tm_mtrix + sp.eye(adj_train.shape[-1])
            train_matrix.append(tr_matrix.todense())

            graph_dgl.append(
            dgl.graph((list(tr_matrix.nonzero()[0]), list(tr_matrix.nonzero()[1])), num_nodes=adj_train.shape[0]))

        train_matrix = [torch.tensor(mtrix) for mtrix in train_matrix]
        adj_train = torch.stack(train_matrix)

    # add the self loop; ToDO: Not the best approach
        graph_dgl.append(
        dgl.graph((list(range(adj_train.shape[-1])), list(range(adj_train.shape[-1]))), num_nodes=adj_train.shape[-1]))

        categorized_val_edges_pos, categorized_val_edges_neg = categorize(val_edges_poitive, val_edges_negative,
                                                                      edge_labels)

        categorized_Test_edges_pos, categorized_Test_edges_neg = categorize(test_edges_positive, test_edges_negative,
                                                                      edge_labels)
    # graph_dgl.append(dgl.from_scipy(adj_train))
    else:
        adj_train = adj_train + sp.eye(adj_train.shape[0])  # the library does not add self-loops
        graph_dgl = [dgl.from_scipy(adj_train)]
        adj_train = torch.tensor(adj_train.todense())  # Todo: use sparse matix
        adj_train = torch.unsqueeze(adj_train, 0)
        categorized_val_edges_pos = {1: val_edges_poitive}
        categorized_val_edges_neg = {1: val_edges_negative}
        categorized_Test_edges_pos = {1: test_edges_positive}
        categorized_Test_edges_neg = {1: test_edges_negative}
    return adj_train,graph_dgl,pre_self_loop_train_adj,categorized_val_edges_pos,categorized_val_edges_neg,categorized_Test_edges_pos,categorized_Test_edges_neg





def process_data(args, hemogenized, original_adj, node_label, edge_labels):
    adj_train, val_edges_poitive, val_edges_negative, test_edges_positive, test_edges_negative, train_edges_positive, train_edges_negative, ignore_edges_inx, val_edge_idx, gt_labels, ignored_edges, num_nodes, masked_indexes  = train_test_split(args, original_adj, node_label)
    adj_train, graph_dgl, pre_self_loop_train_adj, categorized_val_edges_pos, categorized_val_edges_neg, categorized_Test_edges_pos, categorized_Test_edges_neg = create_dgl_graph(hemogenized, edge_labels, adj_train, val_edges_poitive, val_edges_negative, test_edges_positive, test_edges_negative)
    return adj_train,ignore_edges_inx,val_edge_idx,graph_dgl,pre_self_loop_train_adj,categorized_val_edges_pos,categorized_val_edges_neg,categorized_Test_edges_pos,categorized_Test_edges_neg , num_nodes, gt_labels, ignored_edges, masked_indexes
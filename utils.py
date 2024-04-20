import numpy as np
# from mask_test_edges import *
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix,average_precision_score
import torch
import torch.nn as nn
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix, f1_score, classification_report
def edgeList_to_listsOfEdges(edges, ht_graph):
    """
    This function take a list of edges and seperate them based on the edges label,
    in the way the categorized_list[i] contail all the edges with type i

    :param edges_list: dic lists, where edges_list[i] = [[m, n]] shows an ege from node m to node n
    :param ht_graph: a csr matrix where the ht_graph[m,n] indicatte the type of edge between nodes m,n; for none edges ht_graph[m,n]=0
    :return: categorized_list
    """
    rel_label = np.unique(ht_graph.data)

    # categorized_list = {0:[]} # none-edges are assgined to label 0
    categorized_list = {}
    # initialize the dic
    for rel in rel_label:
        categorized_list[rel] = []


    for edge in edges:

        categorized_list[ht_graph[edge[0],edge[1]]].append(edge)

    return categorized_list

def Hemogenizer(adj_matrix):
    """

    :param adj_matrix: given the numpy tesnsor, homegenize it into matix
    :return:
    """
    return adj_matrix.sum(0)

def descrizer(graph, threshold=.5):
    """

    :param graph: numpy array
    :return: discretize numpy array using the threshold
    """
    graph[graph >= 0.5] = 1
    graph[graph < 0.5] = 0
    return graph

def categorize(pos_edges, neg_edges, labels):
    # none-edges are assgined to label 0
    pos_dic = edgeList_to_listsOfEdges(pos_edges, labels)
    # neg_dic = edgeList_to_listsOfEdges(neg_edges, labels) # note len(neg_dic)==1
    neg_dic={}
    count=0
    for key, edg_lists in  pos_dic.items():
        neg_dic[key] = neg_edges[count:count+len(edg_lists)]
        count += len(edg_lists)

    return pos_dic, neg_dic

def Link_prection_eval(pos_edges, negative_edges, reconstructed_adj, original_adj_label):
    """
    This function evaluate the model performance on link prediction for all relation-types
    :param pos_edges:
    :param negative_edges:
    :param adj_mtrix:
    :return:
    """

    def roc_auc_estimator(pos_edges, negative_edges, reconstructed_adj, origianl_agjacency):
        prediction = []
        true_label = []
        #todo: replace the for loop with indexing; its too slow
        if type(pos_edges) == list or type(pos_edges) == np.ndarray:
            for edge in pos_edges:
                prediction.append(reconstructed_adj[edge[0], edge[1]])
                true_label.append(origianl_agjacency[edge[0], edge[1]])

            for edge in negative_edges:
                prediction.append(reconstructed_adj[edge[0], edge[1]])
                true_label.append(origianl_agjacency[edge[0], edge[1]])
        else:
            prediction = list(reconstructed_adj.reshape(-1))
            true_label = list(np.array(origianl_agjacency.todense()).reshape(-1))
        true_label = [1 if x>0 else 0 for x in true_label]# node classification(prediction) for each class
        pred = np.array(prediction)
        true_label = np.array(true_label)
        true_label[true_label>1]=1
        pred[pred > .5] = 1
        pred[pred < .5] = 0
        pred = pred.astype(int)
        # pred = [1 if x>.5 else 0 for x in prediction]

        auc = roc_auc_score(y_score=prediction, y_true=true_label)
        acc = accuracy_score(y_pred=pred, y_true=true_label, normalize=True)
        ap = average_precision_score(y_score=prediction, y_true=true_label)
        cof_mtx = confusion_matrix(y_true=true_label, y_pred=pred)
        return auc, acc, ap, cof_mtx

    for rel_label, egdes in pos_edges.items():
        if rel_label==0:
            pass# the type zero is used for negative edges and will be ignores

        auc , acc,ap, cof_mtx = roc_auc_estimator(egdes, negative_edges[int(rel_label)],
                          reconstructed_adj[int(rel_label)-1], original_adj_label)
        print("The result for rel: {:01d}".format(int(rel_label), ), "with Number of positive edges: {:01d}".format(len(egdes)))
        print("AUC: {:05f}".format(auc) ,"AP: {:05f}".format(ap),"Acuuracy: {:05f}".format(acc))

        print("ConfMtirx:")
        print(cof_mtx)
        

def reduce_node_features(x, y , random_seed,  n_components=5):
    model = ExtraTreesClassifier()
    model.fit(x,y)
    feat_importances = pd.Series(model.feature_importances_)
    important_feats = np.array(feat_importances.nlargest(n_components).index)
    x_reduced = x[:, important_feats] 
    return x_reduced, important_feats
    

def get_metrices(labels_test, labels_pred):
    accuracy = accuracy_score(labels_test, labels_pred)
    
    recall = recall_score(labels_test, labels_pred, average = 'weighted')
    precision = precision_score(labels_test, labels_pred,  average = 'weighted')
    
    results = "Accuracy: {:.4f}, Precision:{:.4f}, recall:{:.4f}".format(accuracy, precision, recall)
    print(results) 



def test_label_decoder(node_labels_pred, gt_labels,masked_indexes):
    predicted_labels = torch.argmax(node_labels_pred, dim=1)
    get_metrices(gt_labels[masked_indexes], predicted_labels[masked_indexes].numpy())


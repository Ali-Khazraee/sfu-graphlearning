#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 12:18:26 2024

@author: pnaddaf
"""


from typing import Optional, Tuple, List
import torch
from torch import Tensor
from torch.nn import Module
from torch_geometric.nn.inits import reset
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn.init as init
import random
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Amazon, Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.utils import train_test_split_edges
import copy
import numpy as np
from sklearn.decomposition import PCA
from pymysql import connect
from pandas import DataFrame
from numpy import zeros, int64, int32, float64, float32, multiply, dot, identity, sum
from itertools import permutations
from math import log
import torch
import gc
import matplotlib.pyplot as plt

from utils import * 


def reduce_node_features(data, random_seed,  n_components=5):
    np.random.seed(random_seed)
    model = ExtraTreesClassifier()
    model.fit(data.x,data.y)
    feat_importances = pd.Series(model.feature_importances_)
    important_feats = np.array(feat_importances.nlargest(n_components).index)
    data.x = data.x[:, important_feats] 
    return data, important_feats
    


random_seed = 0
random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)


# 
# dataset = Planetoid("\..", "citeseer")
# dataset = Amazon("\..", "photo")
# dataa = dataset[0]
dataa = torch.load("../VGAE/db/acm.pt")


data_bi = dataa
data_re, important_feats = reduce_node_features(data_bi, random_seed)
data_re.x = data_re.x.type(torch.float)
data1 = copy.deepcopy(data_re)
data = train_test_split_edges(data_re)




print("Dumping data to database")
db_params = {
    'host': 'database-1.cxcqxpvbnnwo.us-east-2.rds.amazonaws.com',
    'user': 'admin', 
    'password': 'newPassword', 
    'db':'acm' 
}

connection = connect(**db_params)
cursor = connection.cursor()




cursor.execute("DROP DATABASE IF EXISTS %s" %(db_params['db']))
cursor.execute("CREATE DATABASE IF NOT EXISTS %s" %(db_params['db']))
cursor.execute("USE %s" %(db_params['db']))

cursor.execute("""
CREATE TABLE IF NOT EXISTS nodes_table (
    node_id INT PRIMARY KEY,
    feature_1 FLOAT,
    feature_2 FLOAT,
    feature_3 FLOAT,
    feature_4 FLOAT,
    feature_5 FLOAT,
    label FLOAT
)
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS edges_table (
        source_node_id INT,
        target_node_id INT,
        PRIMARY KEY (source_node_id, target_node_id),
        FOREIGN KEY (source_node_id) REFERENCES nodes_table(node_id),
        FOREIGN KEY (target_node_id) REFERENCES nodes_table(node_id)
    )
""")


# Insert nodes into nodes_table
for i, features in enumerate(data1.x):
    # Convert tensor values to Python floats
    feature_values = [float(val) for val in features]
    
    cursor.execute("INSERT INTO nodes_table (node_id, feature_1, feature_2, feature_3, feature_4, feature_5, label) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (i, *feature_values,float(data1.y[i])))

print("Done adding to nodes_table")
# Insert edges into edges_table
edge_index = data1.edge_index.t().numpy()
for edge in edge_index:
    cursor.execute("INSERT INTO edges_table (source_node_id, target_node_id) VALUES (%s, %s)", (edge[0], edge[1]))

connection.commit()
cursor.close()
connection.close()




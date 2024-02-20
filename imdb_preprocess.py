#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:09:24 2024

@author: pnaddaf
"""

from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
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
from torch_geometric.datasets import Amazon, Planetoid, IMDB
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





def  reduce_node_features(x, y , random_seed,  n_components=5):
    model = ExtraTreesClassifier()
    model.fit(x,y)
    feat_importances = pd.Series(model.feature_importances_)
    important_feats = np.array(feat_importances.nlargest(n_components).index)
    x_reduced = x[:, important_feats] 
    return x_reduced, important_feats
    




random_seed = 0
random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)



dataa = IMDB("\..")[0]


data_bi = dataa
x_reduced, important_feats = reduce_node_features(data_bi['movie']['x'], data_bi['movie']['y'], random_seed, )
x_reduced = x_reduced.type(torch.float)





print("Dumping data to database")
db_params = {
    'host': 'database-1.cxcqxpvbnnwo.us-east-2.rds.amazonaws.com',
    'user': 'admin', 
    'password': 'newPassword', 
    'db':'imdb' 
}

connection = connect(**db_params)
cursor = connection.cursor()




cursor.execute("DROP DATABASE IF EXISTS %s" %(db_params['db']))
cursor.execute("CREATE DATABASE IF NOT EXISTS %s" %(db_params['db']))
cursor.execute("USE %s" %(db_params['db']))

cursor.execute("""
CREATE TABLE IF NOT EXISTS movies_table (
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
CREATE TABLE IF NOT EXISTS actors_table (
    node_id INT PRIMARY KEY
)
""")


cursor.execute("""
CREATE TABLE IF NOT EXISTS directors_table (
    node_id INT PRIMARY KEY
)
""")


cursor.execute("""
    CREATE TABLE IF NOT EXISTS movies_directors_table (
        source_node_id INT,
        target_node_id INT,
        PRIMARY KEY (source_node_id, target_node_id),
        FOREIGN KEY (source_node_id) REFERENCES movies_table(node_id),
        FOREIGN KEY (target_node_id) REFERENCES directors_table(node_id)
    )
""")



cursor.execute("""
    CREATE TABLE IF NOT EXISTS movies_actors_table (
        source_node_id INT,
        target_node_id INT,
        PRIMARY KEY (source_node_id, target_node_id),
        FOREIGN KEY (source_node_id) REFERENCES movies_table(node_id),
        FOREIGN KEY (target_node_id) REFERENCES actors_table(node_id)
    )
""")


# Insert nodes into movies_table
for i, features in enumerate(x_reduced):
    # Convert tensor values to Python floats
    feature_values = [float(val) for val in features]
    
    cursor.execute("INSERT INTO movies_table (node_id, feature_1, feature_2, feature_3, feature_4, feature_5, label) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (i, *feature_values,float(dataa['movie']['y'][i])))

print("Done adding to movies_table")



# Insert nodes into actors_table
for i in range(dataa['actor']['x'].shape[0]):

    cursor.execute("INSERT INTO actors_table (node_id) VALUES (%s)",(i))

print("Done adding to actors_table")



# Insert nodes into directors_table
for i in range(dataa['director']['x'].shape[0]):

    cursor.execute("INSERT INTO directors_table (node_id) VALUES (%s)",(i))

print("Done adding to directors_table")



# Insert edges into movies_actors_table
edge_index = dataa.edge_index_dict[('movie', 'to', 'actor')]
for edge in edge_index:
    cursor.execute("INSERT INTO movies_actors_table (source_node_id, target_node_id) VALUES (%s, %s)", (edge[0], edge[1]))

print("Done adding to movies_actors_table")

# Insert edges into movies_directors_table
edge_index = dataa.edge_index_dict[('movie', 'to', 'director')]
for edge in edge_index:
    cursor.execute("INSERT INTO movies_directors_table (source_node_id, target_node_id) VALUES (%s, %s)", (edge[0], edge[1]))


print("Done adding to movie_directors_table")
connection.commit()
cursor.close()
connection.close()
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


np.random.seed(0)
random.seed(0)
torch.seed()
torch.manual_seed(0)
torch.cuda.manual_seed(0)



def reduce_node_features(x, y , random_seed,  n_components=5):
    np.random.seed(random_seed)
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



dataa = IMDB("data")[0]


data_bi = dataa
data_bi['movie']['x'] = torch.where(data_bi['movie']['x'] >= 1, 1, 0)
data_bi['actor']['x'] = torch.where(data_bi['actor']['x'] >= 1, 1, 0)
data_bi['director']['x'] = torch.where(data_bi['director']['x'] >= 1, 1, 0)
x_reduced, important_feats = reduce_node_features(data_bi['movie']['x'], data_bi['movie']['y'], random_seed)
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
CREATE TABLE IF NOT EXISTS movies (
    movie_id INT PRIMARY KEY,
    movie_feature_1 FLOAT,
    movie_feature_2 FLOAT,
    movie_feature_3 FLOAT,
    movie_feature_4 FLOAT,
    movie_feature_5 FLOAT,
    label FLOAT
)
""")



cursor.execute("""
CREATE TABLE IF NOT EXISTS actors (
    actor_id INT PRIMARY KEY,
    actor_feature_1 FLOAT,
    actor_feature_2 FLOAT,
    actor_feature_3 FLOAT,
    actor_feature_4 FLOAT,
    actor_feature_5 FLOAT
)
""")


cursor.execute("""
CREATE TABLE IF NOT EXISTS directors (
    director_id INT PRIMARY KEY,
    director_feature_1 FLOAT,
    director_feature_2 FLOAT,
    director_feature_3 FLOAT,
    director_feature_4 FLOAT,
    director_feature_5 FLOAT
)
""")


cursor.execute("""
    CREATE TABLE IF NOT EXISTS movies_directors (
        movie_id INT,
        director_id INT,
        PRIMARY KEY (movie_id, director_id),
        FOREIGN KEY (movie_id) REFERENCES movies(movie_id),
        FOREIGN KEY (director_id) REFERENCES directors(director_id)
    )
""")




cursor.execute("""
    CREATE TABLE IF NOT EXISTS movies_actors (
        movie_id INT,
        actor_id INT,
        PRIMARY KEY (movie_id, actor_id),
        FOREIGN KEY (movie_id) REFERENCES movies(movie_id),
        FOREIGN KEY (actor_id) REFERENCES actors(actor_id)
    )
""")


# Insert nodes into movies_table
for i, features in enumerate(x_reduced):
    # Convert tensor values to Python floats
    feature_values = [float(val) for val in features]
    
    cursor.execute("INSERT INTO movies (movie_id, movie_feature_1, movie_feature_2, movie_feature_3, movie_feature_4, movie_feature_5, label) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (i, *feature_values,float(dataa['movie']['y'][i])))

print("Done adding to movies_table")


# Insert nodes into actors
for i, features in enumerate(dataa['actor']['x'][:,important_feats ]):
    # Convert tensor values to Python floats
    feature_values = [float(val) for val in features]
    
    cursor.execute("INSERT INTO actors (actor_id, actor_feature_1, actor_feature_2, actor_feature_3, actor_feature_4, actor_feature_5) VALUES (%s, %s, %s, %s, %s, %s)",
                    (i, *feature_values))



# for i in range(dataa['actor']['x'].shape[0]):

#     cursor.execute("INSERT INTO actors (actor_id) VALUES (%s)",(i))

print("Done adding to actors_table")


# Insert nodes into actors
for i, features in enumerate(dataa['director']['x'][:,important_feats ]):
    # Convert tensor values to Python floats
    feature_values = [float(val) for val in features]
    
    cursor.execute("INSERT INTO directors (director_id, director_feature_1, director_feature_2, director_feature_3, director_feature_4, director_feature_5) VALUES (%s, %s, %s, %s, %s, %s)",
                    (i, *feature_values))




# # Insert nodes into directors_table
# for i in range(dataa['director']['x'].shape[0]):

#     cursor.execute("INSERT INTO directors (director_id) VALUES (%s)",(i))

print("Done adding to directors_table")



# Insert edges into movies_actors_table
edge_index = dataa.edge_index_dict[('movie', 'to', 'actor')]
for i in range(edge_index[0].shape[0]):
    cursor.execute("INSERT INTO movies_actors (movie_id, actor_id) VALUES (%s, %s)", (edge_index[0][i].item(),edge_index[1][i].item()))

print("Done adding to movies_actors_table")

# Insert edges into movies_directors_table
edge_index = dataa.edge_index_dict[('movie', 'to', 'director')]
for i in range(edge_index[0].shape[0]):
    cursor.execute("INSERT INTO movies_directors (movie_id, director_id) VALUES (%s, %s)",( edge_index[0][i].item(), edge_index[1][i].item()))


print("Done adding to movie_directors_table")
connection.commit()
cursor.close()
connection.close()
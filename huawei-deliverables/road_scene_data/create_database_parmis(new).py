#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 09:49:54 2024

@author: pnaddaf
"""

import pickle
import pandas as pd
import mysql.connector
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np
import json
import math
import ast


Directory = ""

df_car_edge =pd.read_csv(Directory+'df_edge_car.csv')
df_ego_car_edge =pd.read_csv(Directory+'df_edge_ego_car.csv')
df_car =  pd.read_csv(Directory+ 'car_df.csv')
df_ego_car = pd.read_csv(Directory+ 'ego_car_df.csv')


bin_size = 3
df_car['equal_frequency_velocity_diff'] = pd.qcut(df_car['velocity_diff'], q=bin_size, labels=["low", "medium", "high"])  # why +1 ?  set labels to low, medium, high march 11th 
df_car['equal_frequency_distance_difff'] = pd.qcut(df_car['distance_diff_level'], q=bin_size, labels=["low", "medium", "high"])  # should this this be q=3 instead march 11th
df_car['equal_frequency_velocity_level'] = pd.qcut(df_car['velocity_abs'], q=bin_size, labels=["low", "medium", "high"])

df_ego_car['equal_frequency_velocity_level'] = pd.qcut(df_ego_car['velocity_abs'], q=bin_size, labels=["low", "medium", "high"])


df_car_edge['equal_frequency_velocity_diff'] = pd.qcut(df_car_edge['velocity_diff'], q=bin_size, labels=["low", "medium", "high"])




df_ego_car[['scene_id', 'frame_id']] = df_ego_car['id'].str.strip('()').str.split(',', expand=True)
df_car[['scene_id', 'frame_id']] = df_car['id'].str.strip('()').str.split(',', expand=True)




df_car = df_car[['Name', 'velocity_abs', 'lane_id', 'road_id',
       'left_blinker_on', 'right_blinker_on', 'brake_light_on', 'lane_idx',
       'orig_lane_idx', 'invading_lane', 'id',
       'velocity_abs_ego', 'velocity_ego', 'location_ego', 'rotation_ego',
       'abs_distance', 'velocity_diff', 'velocity_level',
       'velocity_diff_level', 'distance_diff_level',
       'equal_frequency_velocity_diff', 'equal_frequency_distance_difff',
       'equal_frequency_velocity_level', 'isInLane', 'isInego', 'index',
       'scene_id', 'frame_id']]

df_ego_car=df_ego_car[['velocity_abs', 'velocity', 'location',
       'rotation', 'ang_velocity', 'lane_id', 'road_id',
       'left_blinker_on', 'right_blinker_on', 'brake_light_on', 'lane_idx',
       'orig_lane_idx', 'invading_lane', 'id', 'velocity_level',
       'isInLane', 'scene_id','equal_frequency_velocity_level',
       'frame_id']]

df_ego_car['scene_id'] = df_ego_car['scene_id'].astype(int)
df_ego_car['frame_id'] = df_ego_car['frame_id'].astype(int)

df_car['scene_id'] = df_car['scene_id'].astype(int)
df_car['frame_id'] = df_car['frame_id'].astype(int)




#create connection to database
connection = mysql.connector.connect(
    host="database-1.cxcqxpvbnnwo.us-east-2.rds.amazonaws.com",
    user="admin",
    password="newPassword",
    database="road_scene_2_new"
)
cursor = connection.cursor()




#table for cars
create_table_query = """
CREATE TABLE cars (
    car_id INT,
    PRIMARY KEY (car_id)
);
"""
cursor.execute(create_table_query)



create_table_query = """
CREATE TABLE ego_cars (
    ego_id INT,
    PRIMARY KEY (ego_id)
);
"""
cursor.execute(create_table_query)



#table for frames
create_table_query = """
CREATE TABLE frames (
    f_id VARCHAR(100) NOT NULL,
    PRIMARY KEY (f_id)
);
"""
cursor.execute(create_table_query)


#table for successive frames
create_table_query = """
CREATE TABLE succ_frame (
    f_id1 VARCHAR(100) NOT NULL,
    f_id2 VARCHAR(100) NOT NULL,
    PRIMARY KEY (f_id1,f_id2),
    FOREIGN KEY(f_id1)  REFERENCES frames(f_id),
    FOREIGN KEY(f_id2)  REFERENCES frames(f_id)
);
"""
cursor.execute(create_table_query)



#table for cars
create_table_query = """
CREATE TABLE ego_frame (
    f_id VARCHAR(100) NOT NULL,
    ego_id INT NOT NULL,
    Speed VARCHAR(11) NOT NULL,
    PRIMARY KEY (f_id,ego_id),
    FOREIGN KEY(f_id)  REFERENCES frames(f_id),
    FOREIGN KEY(ego_id)  REFERENCES ego_cars(ego_id)
);
"""
cursor.execute(create_table_query)







# we can call it car_in_frame march 11th
create_table_query = f"""
CREATE TABLE car_in_frame (
    f_id VARCHAR(100) NOT NULL,
    car_id INT NOT NULL,
    speed_diff VARCHAR(11),
    near_level VARCHAR(11),
    Lane VARCHAR(100) NOT NULL,
    Speed VARCHAR(11) NOT NULL,
    PRIMARY KEY (f_id, car_id),
    FOREIGN KEY(f_id)  REFERENCES frames(f_id),
    FOREIGN KEY(car_id)  REFERENCES cars(car_id)
)
"""
cursor.execute(create_table_query)
print(f"Table car_in_frame created successfully.")







scene_list = df_car['scene_id'].unique()


scene_list = scene_list

scene_set= []
car_dict = dict()
for x in scene_list:
    t_car_df = df_car[df_car['scene_id']==x]
    t_ego_df = df_ego_car[df_ego_car['scene_id']==x]
    frame_list =t_ego_df['frame_id'].tolist()
    car_dict[x] = (t_car_df['Name'].unique())
    frame_list.sort()
    succ = []
    pre= frame_list[0]
    for i in range(1,len(frame_list)):
        cur = frame_list[i]
        succ.append((x,pre,cur))
        pre = cur
    scene_set.append(succ)
    
    
df_car = df_car[df_car['scene_id'].isin(scene_list)]
df_ego_car = df_ego_car[df_ego_car['scene_id'].isin(scene_list)]
#df_car_edge= df_car_edge[df_car_edge['scene_id'].isin(scene_list)]


df_car['car_id'] = df_car.groupby(['scene_id', 'Name']).ngroup()




#load data into the cars table
table_name = 'cars' 
insert_list = 'car_name,car_id'
d_car = df_car
d_car= d_car.drop_duplicates(subset=['Name','scene_id'])

for index, row in d_car.iterrows():
    query = "INSERT INTO " + table_name + " (car_id) VALUES (%s)"    
    values= (str(row['car_id']),)
    cursor.execute(query,values)
connection.commit()


for x in df_ego_car['scene_id'].unique():
    query = "INSERT INTO ego_cars (ego_id) VALUES (" +str(x)+ ")"    
    cursor.execute(query)
connection.commit()



#load data into the frames table
table_name = 'frames'
insert_list = 'f_id'
for index,row in df_ego_car.iterrows():
    query = "INSERT INTO " + table_name + " ("+ insert_list+ ") VALUES ( %s)"
    values =((str(row['frame_id']))+'|'+str(row['scene_id']),)
    cursor.execute(query,values)  
connection.commit()


# what is the difference between scene and frame march 11th
# each scene has many frames and exactly one ego_car march 11th
grouped = df_ego_car.groupby('scene_id')
grouped_dataframes = {}
for group_name, group_data in grouped:
    grouped_dataframes[group_name] = group_data
    
    

table_name = 'succ_frame'
insert_list = 'f_id1,f_id2'
frame_id_dict= dict()
for x in grouped_dataframes.keys():
    l= grouped_dataframes[x]['frame_id'].tolist()
    frame_id_dict[x]=sorted(l)

for x in frame_id_dict.keys(): 
    l =frame_id_dict[x]
    for i in range(1,len(l)):
        query = "INSERT INTO " + table_name + " ("+ insert_list+ ") VALUES ( %s,%s)"
        values = (str(l[i-1])+'|'+str(x),str(l[i])+'|'+str(x))
        cursor.execute(query,values)  
connection.commit()


table_name = 'ego_frame'
insert_list = 'f_id,ego_id,Speed'
for index, row in df_ego_car.iterrows():
    query = "INSERT INTO " + table_name + " ("+ insert_list+ ") VALUES (%s,%s,%s)"
    values = (str(row['frame_id'])+'|'+str(row['scene_id']),str(row['scene_id']),row['equal_frequency_velocity_level'])
    cursor.execute(query,values)  
connection.commit()


c_df = df_car_edge
merged_df = pd.merge(c_df, df_car[['id','Name','isInLane','frame_id','scene_id','velocity_level','car_id','equal_frequency_velocity_level']], left_on=['id', 'Node_1'],right_on=['id', 'Name'],suffixes=('','_car'))
merged_df = pd.merge(merged_df, df_ego_car[['id']], left_on=['id'],right_on=['id'],suffixes=('','_ego'))


# f_id = frame_id + scence_i march 11th
for index, row in merged_df.iterrows():
    insert_query = "INSERT INTO car_in_frame (f_id, car_id,speed_diff,near_level,Lane,Speed) VALUES (%s,%s,%s,%s,%s,%s)"
    values = (str(row['frame_id'])+'|'+str(row['scene_id']), str(row['car_id']),
              str(row['equal_frequency_velocity_diff']),str(row['near_level']),row['isInLane'],row['equal_frequency_velocity_level']) # changed equal_frequency_velocity_diff to equal_frequency_velocity_level march 11th
    cursor.execute(insert_query, values)

connection.commit()

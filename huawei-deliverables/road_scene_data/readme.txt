
*csv files are raw data

* the notebook and python scripts translate csv data to pandas and then to SQL

Here are the databases for the road data that Mingyi sent: 
Timeslice_1: contain all scenes
Timeslice_2:  10 scenes
Timeslice_3:  3 scenes,  251 frames
Timeslice_4:  1 scene, containing all relation
for now we have only tested on timeslice_2



Difference between Parmis and Mingyi's create_database.py
1) set all the bin_sizes to 3
2) categorized the bins from integer values to strings of low/medium/high
3) changed the name of "relation" table to "car_in_frame"
4) # changed equal_frequency_velocity_diff to equal_frequency_velocity_level (this was a bug before)

* output of Factorbase rule learner is stored in sql files
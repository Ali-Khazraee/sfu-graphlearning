# Graph Learning With Rules



## Overview

This project introduces a novel approach for enhancing graph generative models—specifically a Variational Graph Autoencoder (VGAE)—by integrating a first-order semantic loss function. This loss function enforces a rule moment matching constraint, ensuring that the expected instance count of each first-order logic rule matches its observed count in the data. The method leverages matrix multiplication for efficient computation and demonstrates significant improvements in both graph quality and downstream node classification performance.



## Code overview

The script RuleLearning.py includes the training pipeline.
AEmodels.py contains Encoders and Decoders architecture implementations. 
All the Python packages used in our experiments are provided
in environment.yml. 
nn_model include any other NN modules.
datasets contains all datasets
# File Structure

## motif_count.py
**Description:**  
This file contains the `motif_count` class, which is responsible for counting motifs in the graph.

**Main Methods:**
- **setup_function()**  
  Initializes all the variables needed for the subsequent matrix multiplication process.
- **iteration_function()**  
  Performs the matrix multiplication and counts the motifs based on the variables initialized in `setup_function()`.
- **process_reconstructed_data()**  
  Prepares the graph data for processing by the `motif_count` class.

**Note:**  
Motifs on the training data are counted between lines 123 to 137, and the results are stored in an `observed` variable for each rule.

## helper.py
**Description:**  
This file mostly handles the training part of the model, including various operations on the predicted graph data. Among its functionalities, it counts motifs for the predicted data.  
- The motif counting occurs between lines 142 to 150, with the results saved in the `predicted` variable.

## loss.py
**Description:**  
This file contains the `OptimizerVAE` class, which computes the loss by comparing the observed and predicted motif counts.

**Integration:**  
This module is central to incorporating the rule moment matching constraint into the Variational Graph Autoencoder (VGAE) training process.

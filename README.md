# LLGF

This is the official PyTorch implementation of Latent Layer Generative Framework. 
Workshop Vaersion is accepted at Deep Learning on Graphs: Methods and Applications (DLG-AAAI’22) and the long version is submitted at kdd.

## Code overview

The script main.py includes the training
pipeline. models.py contains the graph VAE with different
Encoder and Decoder architecture implementations. All the
Python packages used in our experiments are provided
in environment.yml. Logs, Hyper-parameters details and
results for each datasets and models are provided in the
result/ directory. datasets contains all datasets

## Run Demos

The framework includes LL version of some of the recent VGAE models.  To run each model you only need to set the encoder and decoder type. Below are some of the main implemented VGAE models and corresponding commands.

### DGLFRM
python VGAE_FrameWork.2.1.py  -decoder_type "MultiLatetnt_SBM_decoder " -encoder_type "mixture_of_GCNs" 

### GRAPHITE
python VGAE_FrameWork.2.1.py   -decoder_type "graphitDecoder" -encoder_type "mixture_of_GCNs"

### VGAE
python VGAE_FrameWork.2.1.py -decoder_type "InnerDot" -encoder_type "mixture_of_GCNs" 

### VGAE*
python VGAE_FrameWork.2.1.py  -decoder_type "multi_inner_product " -encoder_type "mixture_of_GCNs" 

### S-VGAE
python VGAE_FrameWork.2.1.py   -decoder_type "InnerDot" -encoder_type "mixture_of_sGCNs" 

**VGNAE**

python VGAE_FrameWork.2.1.py  -decoder_type "InnerDot" -encoder_type "mixture_of_NGCNs" 


## Sampled Dyad representation from GRAN

## Cite
Please cite our paper if you use this code in your research work.


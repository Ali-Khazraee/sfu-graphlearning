# LLGF

This is the official PyTorch implementation of Latent Layer Generative Framework. 

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

**DGLFRM**

python VGAE_FrameWork.2.1.py  -decoder_type "MultiLatetnt_SBM_decoder " -encoder_type "mixture_of_GCNs" 

**GRAPHITE**

python VGAE_FrameWork.2.1.py   -decoder_type "graphitDecoder" -encoder_type "mixture_of_GCNs"

**VGAE**

python VGAE_FrameWork.2.1.py -decoder_type "InnerDot" -encoder_type "mixture_of_GCNs" 

**VGAE***

python VGAE_FrameWork.2.1.py  -decoder_type "multi_inner_product " -encoder_type "mixture_of_GCNs" 

**S-VGAE**

python VGAE_FrameWork.2.1.py   -decoder_type "InnerDot" -encoder_type "mixture_of_sGCNs" 

**VGNAE**

python VGAE_FrameWork.2.1.py  -decoder_type "InnerDot" -encoder_type "mixture_of_NGCNs" 


## Sampled Dyad representation
We use method-LL to denote the baseline method extended by adding latent layers.  In figures, colors show the ground truth dyad type; blue
= Non-adjacent node pairs, green and red = Adjacent node pairs
with one of 2 edge types. We map learned dyad representations to
a 2-D plane with t-SNE projection. In each figure, Right: Baseline uses 2𝑑′ node
concatenation representation (here 2𝑑′ =128), Left: 𝐿𝐿𝐺𝐹 uses the
novel L dimensional dyad representation, 𝐿 =6 for ACM.1 and
DBLP.1, and 8 for IMDB.1. For the 𝐿𝐿𝐺𝐹 low dimensional dyad
representation, we observe a clear correlation with the ground truth
edge types.
![This is dyad representation](https://github.com/kiarashza/LLGF/blob/master/result/DyadRepresentation/DyadVis.png)

## Cite
Please cite our paper if you use this code in your research work.


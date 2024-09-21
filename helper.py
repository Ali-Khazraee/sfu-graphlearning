import numpy as np

import torch
import torch.optim as optim
from torch import nn
from AEmodels import *
from setup import *
from loss import OptimizerVAE

def create_model_and_optimizer(encoder, decoder, features, num_of_communities, encoder_layers,
                               DropOut_rate, graph_dgl, adj_train, batch_norm,
                               feats_for_reconstruction, node_label, lr, num_nodes):

    # Select and create the encoder model
    if encoder == "GCN_Encoder":
        encoder_model = GCN_Encoder(
            in_feature=features.shape[1],
            latent_dim=num_of_communities,
            layers=encoder_layers,
            DropOut_rate=DropOut_rate
        )
    elif encoder == "RGCN_Encoder":
        encoder_model = RGCN_Encoder(
            in_feature=features.shape[1],
            num_relation=len(graph_dgl),
            latent_dim=num_of_communities,
            layers=encoder_layers,
            DropOut_rate=DropOut_rate
        )
    else:
        raise Exception("Sorry, this Encoder is not implemented; check the input arguments.")

    # Select and create the decoder model
    if decoder == "MultiRelational_SBM":
        decoder_model = MultiRelational_SBM_decoder(
            number_of_rel=adj_train.shape[0],
            Lambda_dim=num_of_communities,
            in_dim=num_of_communities,
            normalize=batch_norm,
            DropOut_rate=DropOut_rate
        )
    elif decoder == "InnerProductDecoder":
        decoder_model = InnerProductDecoder()
    else:
        raise Exception("Sorry, this Decoder is not implemented; check the input arguments.")

    # Create feature and label decoder models
    feature_decoder_model = MLPDecoder(num_of_communities, feats_for_reconstruction.shape[1])
    label_decoder_model = NodeClassifier(num_of_communities, np.unique(node_label).shape[0])

    # Assemble the GVAE framework model
    model = GVAE_FrameWork(
        encoder=encoder_model,
        decoder=decoder_model,
        node_feat_decoder=feature_decoder_model,
        label_decoder=label_decoder_model
    )

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Compute pos_weight for handling class imbalance
    total_edges = adj_train.shape[0] * adj_train.shape[1] * adj_train.shape[2]
    pos_weight = (total_edges - torch.sum(adj_train)) / torch.sum(adj_train)

    # Compute normalization factor
    norm = (num_nodes * num_nodes) / ((num_nodes * num_nodes - torch.sum(adj_train)) * 2)

    return model, optimizer, pos_weight, norm




def train_model(num_nodes, model, optimizer, graph_dgl, features, adj_train, pos_weight, norm, 
                args, device, dataset, mapping_details, important_feat_ids, matrices, rules, multiples, 
                states, functors, variables, nodes, masks, base_indices, mask_indices, sort_indices, 
                stack_indices, values, keys, indices, entities, attributes, relations, rule_weight, 
                prunes, ground_truth, gt_labels, ignore_edges_inx, val_edge_idx, pltr, visulizer_step,
                utils, categorized_val_edges_pos, categorized_val_edges_neg, edge_labels):

    print(model)
    for epoch in range(args.epoch_number):

        model.train()

        # Forward propagation using all nodes
        std_z, m_z, z, reconstructed_adj_logit, reconstructed_x, reconstructed_labels = model(graph_dgl, features)
        
        reconstructed_adjacency = torch.sigmoid(reconstructed_adj_logit)
        reconstructed_x_prob = torch.sigmoid(reconstructed_x)
        reconstructed_labels_prob = torch.sigmoid(reconstructed_labels)
        
        if args.devide_rec_adj:
            reconstructed_adjacency = [(adj * (1 / args.num_nodes)) for adj in reconstructed_adjacency]

        if args.motif_obj:
            reconstructed_x_slice, matrices, reconstructed_labels_m = process_reconstructed_data(
                device, dataset, args, mapping_details, reconstructed_adjacency, reconstructed_x_prob, 
                important_feat_ids, matrices, reconstructed_labels_prob
            )
            predicted = iteration_function(
                device, dataset, args, rules, multiples, states, functors, variables, nodes, masks, 
                base_indices, mask_indices, sort_indices, stack_indices, values, keys, indices, matrices, 
                entities, attributes, relations, rule_weight, prunes, reconstructed_x_slice, 
                reconstructed_labels_m, mode='predicted'
            )
        else:
            predicted = None
                 
        # Compute loss and accuracy
        kl_loss, adj_reconstruction_loss, feat_loss, acc, adj_val_recons_loss, motif_loss, label_loss, loss = OptimizerVAE(
            args, important_feat_ids, reconstructed_adj_logit, adj_train, std_z, m_z, num_nodes, 
            pos_weight, norm, reconstructed_x, features, ground_truth, predicted, reconstructed_labels, 
            gt_labels, ignore_edges_inx, val_edge_idx
        )

        # Backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training metrics
        print("Epoch: {:03d} | Loss: {:.5f} | adj_Reconstruction_loss: {:.5f} | kl_loss: {:.5f} | Feature_Reconstruction_loss: {:.5f} | Accuracy: {:.5f} | Val_adj_Reconstruction_loss: {:.5f}".format(
            epoch + 1, loss.item(), adj_reconstruction_loss.item(), kl_loss.item(), feat_loss.item(), acc, adj_val_recons_loss
        ))
        print("Label loss: {:.5f}".format(label_loss.item()))
        print("Motif loss: {:.5f}".format(motif_loss.item()))

        # Evaluate the model on the validation set and visualize the loss
        if epoch % visulizer_step == 0:
            # pltr.redraw()
            model.eval()
            reconstructed_adj = torch.sigmoid(reconstructed_adj_logit)
            utils.Link_prection_eval(
                categorized_val_edges_pos, categorized_val_edges_neg, reconstructed_adj.detach().numpy(), edge_labels
            )
            model.train()
    # Optionally save the loss plot
    # pltr.save_plot("Loss_plot.png")


import numpy as np
import torch
import torch.optim as optim
from torch import nn
from AEmodels import *
from setup import *
from loss import OptimizerVAE


class Train_Model(Motif_Count):
    def __init__(self, num_nodes, graph_dgl, features, adj_train, args
                , gt_labels, ignore_edges_inx, val_edge_idx, utils,
                 categorized_val_edges_pos, categorized_val_edges_neg, edge_labels, feats_for_reconstruction, node_label, mapping_details, important_feat_ids):
        # Initialize the parent class
        super().__init__(args)

        # Assign variables
        self.num_nodes = num_nodes
        self.graph_dgl = graph_dgl
        self.features = features
        self.adj_train = adj_train
        self.device = args.device
        self.dataset = args.dataset
        self.mapping_details = mapping_details
        self.important_feat_ids = important_feat_ids
        self.gt_labels = gt_labels
        self.ignore_edges_inx = ignore_edges_inx
        self.val_edge_idx = val_edge_idx
        self.visulizer_step = args.Vis_step
        self.utils = utils
        self.categorized_val_edges_pos = categorized_val_edges_pos
        self.categorized_val_edges_neg = categorized_val_edges_neg
        self.edge_labels = edge_labels
        self.feats_for_reconstruction = feats_for_reconstruction
        self.node_label =  node_label 

        self.model = None
        self.optimizer = None
        self.pos_weight =None
        self.norm = None
        





    def create_model_and_optimizer(self):
        encoder = self.args.encoder_type
        decoder = self.args.decoder_type
        features = self.features
        num_of_communities = self.args.Z_dimension
        encoder_layers = [int(x) for x in self.args.encoder_layers.split()]
        DropOut_rate = self.args.DropOut_rate
        graph_dgl = self.graph_dgl
        adj_train = self.adj_train
        batch_norm = self.args.batch_norm
        lr = self.args.lr

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
        feature_decoder_model = MLPDecoder(num_of_communities, self.feats_for_reconstruction.shape[1])
        label_decoder_model = NodeClassifier(num_of_communities, np.unique(self.node_label).shape[0])

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
        norm = (self.num_nodes * self.num_nodes) / ((self.num_nodes * self.num_nodes - torch.sum(adj_train)) * 2)

        return model, optimizer, pos_weight, norm





    def train(self, ground_truth):

        # Create the model and optimizer
        self.model, self.optimizer, self.pos_weight, self.norm = self.create_model_and_optimizer()
        print(self.model)

        for epoch in range(self.args.epoch_number):
            self.model.train()

            # Forward propagation using all nodes
            std_z, m_z, z, reconstructed_adj_logit, reconstructed_x, reconstructed_labels = self.model(
                self.graph_dgl, self.features)

            reconstructed_adjacency = torch.sigmoid(reconstructed_adj_logit)
            reconstructed_x_prob = torch.sigmoid(reconstructed_x)
            reconstructed_labels_prob = torch.sigmoid(reconstructed_labels)

            if self.args.devide_rec_adj:
                reconstructed_adjacency = [
                    (adj * (1 / self.args.num_nodes)) for adj in reconstructed_adjacency
                ]

            if self.args.motif_obj:
                reconstructed_x_slice, reconstructed_labels_m = self.process_reconstructed_data(self.mapping_details, 
                    reconstructed_adjacency, reconstructed_x_prob, self.important_feat_ids,  reconstructed_labels_prob
                )
                predicted = self.iteration_function(
                    reconstructed_x_slice, reconstructed_labels_m, mode='predicted'
                )
            else:
                predicted = None

            # Compute loss and accuracy
            kl_loss, adj_reconstruction_loss, feat_loss, acc, adj_val_recons_loss, motif_loss, label_loss, loss = OptimizerVAE(
                self.args, self.important_feat_ids, reconstructed_adj_logit, self.adj_train, std_z,
                m_z, self.num_nodes, self.pos_weight, self.norm, reconstructed_x, self.features,
                ground_truth, predicted, reconstructed_labels, self.gt_labels,
                self.ignore_edges_inx, self.val_edge_idx
            )

            # Backward propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Print training metrics
            print("Epoch: {:03d} | Loss: {:.5f} | adj_Reconstruction_loss: {:.5f} | kl_loss: {:.5f} | "
                  "Feature_Reconstruction_loss: {:.5f} | Accuracy: {:.5f} | Val_adj_Reconstruction_loss: {:.5f}".format(
                epoch + 1, loss.item(), adj_reconstruction_loss.item(), kl_loss.item(),
                feat_loss.item(), acc, adj_val_recons_loss
            ))
            print("Label loss: {:.5f}".format(label_loss.item()))
            print("Motif loss: {:.5f}".format(motif_loss.item()))

            # Evaluate the model on the validation set and visualize the loss
            if epoch % self.visulizer_step == 0:
                self.model.eval()
                reconstructed_adj = torch.sigmoid(reconstructed_adj_logit)
                self.utils.Link_prection_eval(
                    self.categorized_val_edges_pos, self.categorized_val_edges_neg,
                    reconstructed_adj.detach().numpy(), self.edge_labels
                )
                self.model.train()

        return self.model, reconstructed_labels, reconstructed_adj

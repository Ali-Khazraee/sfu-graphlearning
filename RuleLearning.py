import torch
import time
start_time = time.monotonic()
import dgl
import classification as CL
from input_data import load_data
from mask_test_edges import mask_test_edges_new, roc_auc_estimator, mask_test_edges, mask_test_edges_new
import plotter
import argparse
from utils import *
import utils
import networkx as nx
from AEmodels import *
from setup import setup_function, iteration_function
import copy

np.random.seed(0)
random.seed(0)
torch.seed()
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



# torch._set_deterministic(True)

# batch_norm
# edge_Type
# decoder_type and encoder_type
# modelpath
parser = argparse.ArgumentParser(description='VGAE Framework')

parser.add_argument('-e', dest="epoch_number", type=int, default=101, help="Number of Epochs")
parser.add_argument('-v', dest="Vis_step", type=int, default=20, help="model learning rate")
parser.add_argument('-lr', dest="lr", type=float, default=0.001, help="number of epoch at which the error-plot is visualized and updated")
parser.add_argument('-dataset', dest="dataset", default="citeseer",
                    help="possible choices are: cora, citeseer, pubmed, IMDB, DBLP, ACM, IMDB-PyG")
parser.add_argument('-hemogenize', dest="hemogenize", default=False, help="either withhold the layers (edges types) during training or not")
parser.add_argument('-NofCom', dest="num_of_comunities", type=int, default=64,
                    help="Dimention of Z, i.e len(Z[0]), in the bottleNEck")
parser.add_argument('-encoder_layers', dest="encoder_layers", default="64", type=str,
                    help="a list in which each element determine the gcn size; Note: the last layer size is determine with -NofCom")
parser.add_argument('-f', dest="use_feature", default=True, help="either use features or identity matrix")
parser.add_argument('-DR', dest="DropOut_rate", default=0, help="drop out rate")
parser.add_argument('-BN', dest="batch_norm", default=True,
                    help="either use batch norm at decoder; only apply in multi relational decoders")
parser.add_argument('-Split', dest="split_the_data_to_train_test", default=True,
                    help="either use features or identity matrix; for synthasis data default is False")
parser.add_argument('-decoder_type', dest="decoder_type", default="MultiRelational_SBM",
                    help="the decoder type: InnerProductDecoder,MultiRelational_SBM")
parser.add_argument('-encoder_type', dest="encoder_type", default="RGCN_Encoder",
                    help="the encoder type:GCN_Encoder, RGCN_Encoder")
parser.add_argument('-num_node', dest="num_node", default=-1, type=str,
                    help="the size of subgraph which is sampled; -1 means use the whole graph; I added this feature to enable us to train on a small subset of graph to speed up Dev")
parser.add_argument("-downstreamTasks", dest="downstreamTasks" , default= {"nodeClassification","linkPrediction"}, help="a ser of downsteam tasks", nargs='+',)




args = parser.parse_args()

# setting
print("VGAE FRAMEWORK SETING: " + str(args))
visulizer_step = args.Vis_step
epoch_number = args.epoch_number
lr = args.lr
hemogenized = args.hemogenize
downstreamTasks = args.downstreamTasks

num_of_comunities = args.num_of_comunities  # number of comunities;
DropOut_rate = args.DropOut_rate
batch_norm = args.batch_norm
dataset = args.dataset  # possible choices are: cora, citeseer, karate, pubmed, DBIS
decoder = args.decoder_type
encoder = args.encoder_type
encoder_layers = [int(x) for x in args.encoder_layers.split()]
use_feature = args.use_feature

subgraph_size = args.num_node
if dataset in {"facebook_egoes"}:  # using subgraph will result in exception
    subgraph_size = -1

split_the_data_to_train_test = args.split_the_data_to_train_test

synthesis_graphs = {"grid", "community", "lobster", "ego"}


#============================================================'
# Tcount ground truth motif

# rules, multiples, states, functors, variables, nodes, masks, base_indices, mask_indices, sort_indices, stack_indices, values, keys, indices, matrices, entities = setup_function(dataset)

# ground_truth = iteration_function(rules, multiples, states, functors, variables, nodes, masks, base_indices, mask_indices, sort_indices, stack_indices, values, keys, indices, matrices, entities)

'edit'


# ************************************************************
# VGAE frame_work
class GVAE_FrameWork(torch.nn.Module):
    def __init__(self, encoder, decoder, node_feat_decoder, label_decoder):
        """
        :param latent_dim: the dimention of each embedded node; |z| or len(z)
        :param decoder:
        :param encoder:
        :param mlp_decoder: either apply an multi layer perceptorn on each decoeded embedings
        """
        super(GVAE_FrameWork, self).__init__()
        # self.relation_type_param = torch.nn.ParameterList(torch.nn.Parameter(torch.Tensor(2*latent_space_dim)) for x in range(latent_space_dim))

        self.decoder = decoder
        self.encoder = encoder
        self.node_feat_decoder = node_feat_decoder
        self.label_decoder = label_decoder


        # self.mlp_decoder = torch.nn.ModuleList([edge_mlp(2*latent_space_dim,[16,8,1]) for i in range(self.numb_of_rel)])

    def forward(self, adj, x):
        z, m_z, std_z = self.inference(adj, x)
        generated_adj = self.generator(z, x)
        generated_x = self.feat_generator(z)
        generated_label = self.label_decoder(z)
        return std_z, m_z, z, generated_adj, generated_x, generated_label

    # inference model q(z|adj,x)
    def inference(self, adj, x):
        z, m_q_z, std_q_z = self.encoder(adj, x)
        return z, m_q_z, std_q_z

    # generative model p(adj|z)
    def generator(self, z, x):
        adj = self.decoder(z)
        return adj

    def feat_generator(self, z):
        x = self.node_feat_decoder(z)
        return x

    def reparameterize(self, mean, std):
        eps = torch.randn_like(std)
        return eps.mul(std).add(mean)


# ************************************************************

# objective Function
def OptimizerVAE(pred, labels, std_z, mean_z, num_nodes, pos_wight, norm, x_pred, x_true, ground_truth, predicted, predicted_node_labels, gt_labels, indexes_to_ignore=None, val_edge_idx=None):
    """
    :param pred: reconstructed adj matrix by model; its a stack of dense adj matrix
    :param labels: The origianl adj matrix which shoul be reconstructed; stack of matrices
    :param std_z:
    :param mean_z:
    :param num_nodes:
    :param pos_wight: The ratio of positive to negative edges
    :param norm:
    :param indexes_to_ignore: This edges whould not effect the reconstruction loss; its a list  of lists with len(indexes_to_ignore)==2; e.g   [indexes_to_ignore[0],indexes_to_ignore[1]] wount be effect the reconstruction loss
    :param val_edge_idx: the indexces of val edges. its just for cal reconstrction loss of val edges
    :return:
    """
    val_recons_loss = None

    # 'start edit'
    
    # ground_truth = [tensor.item() for tensor in ground_truth]

    # std_dev = np.std(ground_truth)
    # for i in range(len(ground_truth)):
    #     ground_truth[i] = ground_truth[i] / std_dev
        
    # for i in range(len(ground_truth)):
    #     predicted[i] = predicted[i] / std_dev
                 

    # motif = ((a-b)**2 for a, b in zip(ground_truth, predicted))
    # motif_loss = np.sum(np.fromiter(motif, dtype=float))*(1/len(ground_truth))
    # 'end edit'
    
    reconstruction_loss = norm * F.binary_cross_entropy_with_logits(pred, labels, pos_weight=pos_wight, reduction='none')

    # the validation reconstruction loss
    if val_edge_idx:
        val_recons_loss = reconstruction_loss[:, val_edge_idx[0], val_edge_idx[1]].mean()

    # some edges wont have reconstruciton losss
    if indexes_to_ignore and len(indexes_to_ignore) > 0:
        reconstruction_loss[:, indexes_to_ignore[0], indexes_to_ignore[1]] = 0  # masking edges

    reconstruction_loss = reconstruction_loss.mean()


    # KL divergence
    kl_loss = (-0.5 / num_nodes) * torch.mean(
        torch.sum(1 + 2 * torch.log(std_z) - mean_z.pow(2) - (std_z).pow(2), dim=1))
    feat_loss = F.binary_cross_entropy_with_logits(x_pred,x_true.float())
    

    
    # label loss
    not_masked_labels = torch.where(gt_labels != -1)[0]
    criterion = nn.CrossEntropyLoss()
    label_loss = criterion(predicted_node_labels[not_masked_labels,:], gt_labels[not_masked_labels])


    acc = (torch.sigmoid(pred).round() == labels).sum() / float(pred.shape[0] * pred.shape[1]*pred.shape[2]) # accuracy on the train data
    
    motif_loss = 0
    return kl_loss, reconstruction_loss, feat_loss , acc, val_recons_loss , motif_loss, label_loss

# ============================================================
# The main procedure

# load the data
if dataset in ('grid', 'community', 'ego', 'lobster'):
    synthetic = True
    original_adj, features = load_data(dataset)
    node_label = edge_labels = circles = None
else:
    synthetic = False
    original_adj, features, node_label, edge_labels, circles, map_dic, important_feat_ids, feats_for_reconstruction= load_data(dataset)
# shuffling the data, and selecting a subset of it; subgraph_size is used to do the ecperimnet on the samller dataset to insclease development speed
if subgraph_size == -1:
    subgraph_size = original_adj.shape[-1]
elemnt = min(original_adj.shape[-1], subgraph_size)
indexes = list(range(original_adj.shape[-1]))

# -----------------------------------------
# # adj , feature matrix and  node labels  permutaion
# np.random.shuffle(indexes)
# indexes = indexes[:elemnt]
# original_adj = original_adj[indexes, :]
# original_adj = original_adj[:, indexes]
#
# features = features[indexes]
#
# if synthetic != True:
#     if node_label != None:
#         node_label = [node_label[i] for i in indexes]
#     if edge_labels != None:
#         edge_labels = edge_labels[indexes, :]
#         edge_labels = edge_labels[:, indexes]
#     if circles != None:
#         shuffles_cir = {}
#         for ego_node, circule_lists in circles.items():
#             shuffles_cir[indexes.index(ego_node)] = [[indexes.index(x) for x in circule_list] for circule_list in
#                                                      circule_lists]
#         circles = shuffles_cir
# -----------------------------------------

# instead of X used I if the switch is on
if use_feature == False:
    features = torch.eye(features.shape[0])
    features = sp.csr_matrix(features)

# make train, test and val according to kipf original implementation
if split_the_data_to_train_test == True:
    adj_train, _, val_edges_poitive, val_edges_negative, test_edges_positive, test_edges_negative, train_edges_positive, train_edges_negative, ignore_edges_inx, val_edge_idx = mask_test_edges(original_adj)#mask_test_edges_new(original_adj)
    ignore_dges = []
else:
    train_edges = val_edges = val_edges_false = test_edges = test_edges_false = ignore_edges_inx = val_edge_idx = None
    adj_train = original_adj

# I use this mudule to plot error and loss
pltr = plotter.Plotter(functions=["loss", "adj_Recons Loss", "feature_Rec Loss", "KL", ])



num_obs = 1  # number of relateion; default is hemogenous dataset with one type of edge
if hemogenized != True:
    edge_relType_train = edge_labels.multiply(adj_train)
    rel_type = np.unique(edge_labels.data)
    num_obs = len(rel_type)  # number of relateion; heterougenous setting
    # edge_relType = edge_relType + sp.eye(adj_train.shape[-1]) * (len(np.unique(edge_relType.data)) + 1)
    graph_dgl = []

    train_matrix = []
    for rel_num in rel_type:
        tm_mtrix = csr_matrix(edge_relType_train.shape)
        tm_mtrix[edge_relType_train == (rel_num)] = 1
        tr_matrix = tm_mtrix + sp.eye(adj_train.shape[-1])
        train_matrix.append(tr_matrix.todense())

        graph_dgl.append(
            dgl.graph((list(tr_matrix.nonzero()[0]), list(tr_matrix.nonzero()[1])), num_nodes=adj_train.shape[0]))

    train_matrix = [torch.tensor(mtrix) for mtrix in train_matrix]
    adj_train = torch.stack(train_matrix)

    # add the self loop; ToDO: Not the best approach
    graph_dgl.append(
        dgl.graph((list(range(adj_train.shape[-1])), list(range(adj_train.shape[-1]))), num_nodes=adj_train.shape[-1]))

    categorized_val_edges_pos, categorized_val_edges_neg = categorize(val_edges_poitive, val_edges_negative,
                                                                      edge_labels)
    # graph_dgl.append(dgl.from_scipy(adj_train))
else:
    adj_train = adj_train + sp.eye(adj_train.shape[0])  # the library does not add self-loops
    graph_dgl = dgl.from_scipy(adj_train)
    adj_train = torch.tensor(adj_train.todense())  # Todo: use sparse matix
    adj_train = torch.unsqueeze(adj_train, 0)
    categorized_val_edges_pos = {1: val_edges_poitive}
    categorized_val_edges_neg = {1: val_edges_negative}



num_nodes = adj_train.shape[-1]

if (type(features) == np.ndarray):
    features = torch.tensor(features, dtype=torch.float32)
else:
    features = torch.tensor(features.todense(), dtype=torch.float32)

# -----------------------------------------
# initialize the model
# -----------------------------------------
# Check for Encoder and redirect to appropriate function
if encoder == "GCN_Encoder":
    encoder_model = GCN_Encoder(in_feature=features.shape[1],
                                latent_dim=num_of_comunities, layers=encoder_layers, DropOut_rate=DropOut_rate)
# add your encoder
# e.g elif: myEncoder = encoder()

elif encoder == "RGCN_Encoder":
    encoder_model = RGCN_Encoder(in_feature=features.shape[1], num_relation=len(graph_dgl),
                                 latent_dim=num_of_comunities, layers=encoder_layers, DropOut_rate=DropOut_rate)
else:
    raise Exception("Sorry, this Encoder is not Impemented; check the input args")

# Check for Decoder and redirect to appropriate function
# if decoder == ""
if decoder == "MultiRelational_SBM":
    decoder_model = MultiRelational_SBM_decoder(number_of_rel=adj_train.shape[0], Lambda_dim=num_of_comunities,
                                                in_dim=num_of_comunities, normalize=batch_norm,
                                                DropOut_rate=DropOut_rate)
elif decoder == "InnerProductDecoder":  # Kipf
    decoder_model = InnerProductDecoder()
else:
    raise Exception("Sorry, this Decoder is not Impemented; check the input args")

feature_decoder_model = MLPDecoder(num_of_comunities,feats_for_reconstruction.shape[1])
label_decoder_model = NodeClassifier(num_of_comunities, np.unique(node_label).shape[0])

model = GVAE_FrameWork(encoder=encoder_model,
                       decoder=decoder_model,
                       node_feat_decoder = feature_decoder_model,
                       label_decoder = label_decoder_model)  # parameter namimng, it should be dimentionality of distriburion
# -----------------------------------------
# -----------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr)

pos_wight = torch.true_divide((adj_train.shape[0] * adj_train.shape[1] * adj_train.shape[2] - torch.sum(adj_train)),
                              torch.sum(
                                  adj_train))  # addrressing imbalance data problem: ratio between positve to negative instance

norm = torch.true_divide(adj_train.shape[-1] * adj_train.shape[-1],
                         ((adj_train.shape[-1] * adj_train.shape[-1] - torch.sum(adj_train)) * 2))

best_recorded_validation = None
best_epoch = 0


# mask 20% of the labels
gt_labels = copy.deepcopy(node_label)
masked_indexes = np.random.choice(gt_labels.shape[0], round(gt_labels.shape[0] * 2/10), replace=False)
gt_labels[masked_indexes] = -1

print(model)
for epoch in range(epoch_number):

    model.train()

    # forward propagation by using all nodes

    std_z, m_z, z, reconstructed_adj_logit, reconstructed_x, reconstructed_labels = model(graph_dgl, features)
    
    # 'start edit'

    # reconstructed_adjacency = torch.sigmoid(reconstructed_adj_logit)
    


    # if dataset == "IMDB-PyG":
    #     edge_encoding_to_node_types = {v: k for k, v in map_dic['edge_type_encoding'].items()}
    #     filtered_reconstruct_adj = []
        
    #     for idx, adj_matrix in enumerate(reconstructed_adjacency):
    #         node_types = edge_encoding_to_node_types[idx + 1]  
    #         src_type, dst_type = node_types

    #         src_start, src_end = map_dic['node_type_to_index_map'][src_type]
    #         dst_start, dst_end = map_dic['node_type_to_index_map'][dst_type]


    #         filtered_matrix = adj_matrix[src_start:src_end, dst_start:dst_end]


    #         filtered_reconstruct_adj.append(filtered_matrix)
        
    #     filtered_reconstruct_adj_tensors = [torch.tensor(matrix).to('cuda:0') for matrix in filtered_reconstruct_adj]
        
        
    #     for filtered_matrix in filtered_reconstruct_adj_tensors:
    #         filtered_shape = filtered_matrix.shape 
        
    #         for key, matrix in matrices.items():
    #             if matrix.shape == filtered_shape:
    #                 matrices[key] = filtered_matrix
    #                 break
                
        
        
    #     movie_reconstructed_x = reconstructed_x[:map_dic['node_type_to_index_map']['movie'][1]]        
    #     reconstructed_x_important_feats = movie_reconstructed_x[:,important_feat_ids]
    #     for i in range(1, len(important_feat_ids)+1):
    #         feature_name = f'feature_{i}'
    
    #         tensor_to_assign = ((reconstructed_x_important_feats[:, i-1]) > 0.5).int().cpu().numpy()
    #         entities['movies'][feature_name] = tensor_to_assign
        
    #     # only use labels of the movies
       
    #     reconstructed_labels = reconstructed_labels[:map_dic['node_type_to_index_map']['movie'][1]]
    
    #     entities['movies']['label'] = torch.argmax(reconstructed_labels, dim=1)
    
    # else:
        
    #     x_recon_important_feats = reconstructed_x[:,important_feat_ids]
        

    #     for i in range(1, len(important_feat_ids)+1):
    #         feature_name = f'feature_{i}'
        
    #         tensor_to_assign = ((x_recon_important_feats[:, i-1]) > 0.5).int().cpu().numpy()
    #         entities['nodes_table'][feature_name] = tensor_to_assign
        
    #     predicted_labels = torch.argmax(reconstructed_labels, dim=1)
    #     entities['nodes_table']['label'] = predicted_labels.cpu().detach().numpy()
    #     matrices['edges_table'] = reconstructed_adjacency[0].to('cuda:0')
        
            
    # predicted = iteration_function(rules, multiples, states, functors, variables, nodes, masks, base_indices, mask_indices, sort_indices, stack_indices, values, keys, indices, matrices, entities)

            
    # 'end edit'        
            
    # compute loss and accuracy
    ground_truth = 0
    predicted = 0
    z_kl, adj_reconstruction_loss,feat_loss, acc, adj_val_recons_loss, motif_loss, label_loss = OptimizerVAE(reconstructed_adj_logit, adj_train , std_z, m_z, num_nodes, pos_wight, norm,reconstructed_x,feats_for_reconstruction, ground_truth, predicted, reconstructed_labels, gt_labels,  ignore_edges_inx, val_edge_idx)
    loss = adj_reconstruction_loss + z_kl + torch.tensor(motif_loss) + label_loss + feat_loss

    # record the loss; to be ploted
    pltr.add_values(epoch, [ loss.item() ,adj_reconstruction_loss.item(), feat_loss.item(), z_kl.item()], [None,adj_val_recons_loss.item(), None, None], redraw=False)  # plotter.Plotter(functions=["loss", "adj_Recons Loss","feature_Rec Loss", "KL",])

    # backward propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # -------------------------------------------
    # batch detail
    # print some metrics
    print("Epoch: {:03d} | Loss: {:05f} | adj_Reconstruction_loss: {:05f} | z_kl_loss: {:05f} |".format(
        epoch + 1, loss.item(), adj_reconstruction_loss.item(), z_kl.item()),"Feature_Reconstruction_loss: {:05f} |".format(feat_loss.item()), "Acuuracy: {:05f} |".format(acc), "Val_adj_Reconstruction_loss: {:05f}".format(adj_val_recons_loss))
    # ------------------------------------------
    print("label loss: ", label_loss.item())
    # Evaluate the model on the validation and plot the loss at every visulizer_step
    if epoch % visulizer_step == 0:
        pltr.redraw()
        model.eval()
        reconstructed_adj = torch.sigmoid(reconstructed_adj_logit)
        utils.Link_prection_eval(categorized_val_edges_pos, categorized_val_edges_neg,
                                                            reconstructed_adj.detach().numpy(), edge_labels)
        model.train()
# save the loss plot in current dir
pltr.save_plot("Loss_plot.png")

test_label_decoder(reconstructed_labels, node_label,masked_indexes)


if "nodeClassification" in downstreamTasks:
    # add node classification code here
    # z = model(adj, x)
    # pred_label = Classifier(z, label)
    pass



# note: TODO: add evaluation on test set
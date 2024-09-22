import time
start_time = time.monotonic()
from input_data import load_data
from mask_test_edges import *
import argparse
from utils import *
import utils
from helper import *

np.random.seed(0)
random.seed(0)
torch.seed()
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

'you can donwload the acm graph opbject from the link below'
##https://iutbox.iut.ac.ir/index.php/s/5aHAaqEA3mfwCNG
'you should put it in data/acm_multi/multi_acm.pt directory'


# torch._set_deterministic(True)

# batch_norm
# edge_Type
# decoder_type and encoder_type
# modelpath
parser = argparse.ArgumentParser(description='VGAE Framework')

parser.add_argument('-e', dest="epoch_number", type=int, default=301, help="Number of Epochs")
parser.add_argument('-div', dest="device",  default="gpu", help="device")
parser.add_argument('-v', dest="Vis_step", type=int, default=100, help="model learning rate")
parser.add_argument('-lr', dest="lr", type=float, default=0.001, help="number of epoch at which the error-plot is visualized and updated")
parser.add_argument('-dataset', dest="dataset", default="cora",
                    help="possible choices are: cora, citeseer, pubmed, IMDB, DBLP, ACM, imdb-multi, acm-multi")
parser.add_argument('-hemogenize', dest="hemogenize", default=False, help="either withhold the layers (edges types) during training or not")
parser.add_argument('-NofCom', dest="Z_dimension", type=int, default=64,
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
parser.add_argument('-motif_obj', dest="motif_obj", default= True , help="adds motif_loss term to objective function")
parser.add_argument('-rp', dest="rule_prune",  default= True , help="Toggle rule pruning on or off")
parser.add_argument('-rw', dest="rule_weight",  default= False , help="Toggle rule weighting on or off - If you want to use rule weighting, you need to turn on rule pruning first by setting it to True.")
parser.add_argument('-dr', dest="devide_rec_adj",  default= False , help="This switch will divide reconstructed adjacency matrix by 1/n in every epoch")
parser.add_argument('-task', dest="task", default="node_classification", help="possible choices are: node_classification, link_prediction")
parser.add_argument('-graph_type', dest="graph_type", default="homogeneous", choices=["homogeneous", "heterogeneous"], help="Choose the graph type: homogeneous or heterogeneous")
parser.add_argument('-motif_weight', dest="motif_weight", type=float, default=0.01, help="Specify the weight for the motif loss term in the loss function")


#imdb-multi , acm-multi : heterogeneous
# cora , citeseer : homogeneous

args = parser.parse_args()

# setting
print("VGAE FRAMEWORK SETING: " + str(args))
visulizer_step = args.Vis_step
epoch_number = args.epoch_number
lr = args.lr
hemogenized = args.hemogenize
downstreamTasks = args.downstreamTasks

num_of_comunities = args.Z_dimension  # the dimention of the bottleneck
DropOut_rate = args.DropOut_rate
batch_norm = args.batch_norm
dataset = args.dataset  
decoder = args.decoder_type
encoder = args.encoder_type
encoder_layers = [int(x) for x in args.encoder_layers.split()]
use_feature = args.use_feature
use_motif = args.motif_obj
rule_prune = args.rule_prune
divide_ajd = args.devide_rec_adj
rule_weight = args.rule_weight



device = torch.device("cuda" if torch.cuda.is_available() and args.device =="gpu" else "cpu")

print('===============')
print(device)
print('===============')




# Load the data
original_adj, features, node_label, edge_labels, circles, mapping_details, important_feat_ids, feats_for_reconstruction,feats_for_reconstruction_count, one_hot_labe = load_data(dataset, device, args)

# This function consists of two separate functions: one splits the data into training and testing sets, and the other creates a DGL graph.
adj_train, ignore_edges_inx, val_edge_idx, graph_dgl, pre_self_loop_train_adj, categorized_val_edges_pos, categorized_val_edges_neg, categorized_Test_edges_pos, categorized_Test_edges_neg, num_nodes , gt_labels, ignored_edges, masked_indexes = process_data(args, hemogenized, original_adj, node_label, edge_labels)

# TODO : need to replace this part of the code
# # pltr = plotter.Plotter(functions=["loss", "adj_Recons Loss", "feature_Rec Loss", "KL", ])




# initialize the model

model, optimizer, pos_weight, norm = create_model_and_optimizer(
    encoder,
    decoder,
    features,
    num_of_comunities,
    encoder_layers,
    DropOut_rate,
    graph_dgl,
    adj_train,
    batch_norm,
    feats_for_reconstruction,
    node_label,
    lr,
    num_nodes
)



# TODO : this part of the code needs to be functionalized
# # mask 20% of the labels
# gt_labels = copy.deepcopy(node_label)
# masked_indexes = np.random.choice(gt_labels.shape[0], round(gt_labels.shape[0] * 2/10), replace=False)
# gt_labels[masked_indexes] = -1



#============================================================'
# count ground truth motif
# TODO : this part of the code needs to be replaced
if use_motif == True:
    rules, multiples, states, functors, variables, nodes, masks, base_indices, mask_indices, sort_indices, stack_indices, values, keys, indices, matrices, entities,attributes,relations, prunes = setup_function(dataset, rule_prune, rule_weight, device)
    
    if mapping_details != None:
        self_loop_train_adj = add_self_loops(pre_self_loop_train_adj)
        update_matrices(device, matrices, mapping_details, self_loop_train_adj)
    else:
        key = next(iter(matrices))
        pre_self_loop_train_adj1 = add_self_loops(pre_self_loop_train_adj)
        matrices[key] = torch.tensor(pre_self_loop_train_adj1[0]).to(device)
    
    ground_truth = iteration_function(device, dataset,args , rules, multiples, states, functors, variables, nodes, masks, base_indices, mask_indices, sort_indices, stack_indices, values, keys, indices, matrices, entities,attributes,relations ,rule_weight, prunes , feats_for_reconstruction_count , one_hot_labe.to(device) , mode = 'ground_truth')
else:
    ground_truth = None

# ************************************************************



# train model
model, reconstructed_labels, reconstructed_adj = train_model(
    num_nodes,
    model,
    optimizer,
    graph_dgl,
    features,
    adj_train,
    pos_weight,
    norm,
    args,
    device,
    dataset,
    mapping_details,
    important_feat_ids,
    matrices,
    rules,
    multiples,
    states,
    functors,
    variables,
    nodes,
    masks,
    base_indices,
    mask_indices,
    sort_indices,
    stack_indices,
    values,
    keys,
    indices,
    entities,
    attributes,
    relations,
    rule_weight,
    prunes,
    ground_truth,
    gt_labels,
    ignore_edges_inx,
    val_edge_idx,
    plt,
    visulizer_step,
    utils,
    categorized_val_edges_pos,
    categorized_val_edges_neg,
    edge_labels
)




#evaluation part
test_label_decoder(reconstructed_labels, node_label,masked_indexes)

# test result
print("link prediction on test set")
utils.Link_prection_eval(categorized_Test_edges_pos, categorized_Test_edges_neg,
                                                            reconstructed_adj.detach().numpy(), edge_labels)




# TODO : i need to create a module for closeness metrix                                                       
# closeness metric 
# # TODO : I can functionalize this part of the code
# num_obs = 1  
# if not hemogenized:
#     edge_relType_full = edge_labels.multiply(original_adj)
#     rel_type = np.unique(edge_labels.data)
#     num_obs = len(rel_type)  

#     graph_dgl = []
#     full_matrix = []
#     for rel_num in rel_type:
#         tm_matrix = csr_matrix(edge_relType_full.shape)
#         tm_matrix[edge_relType_full == rel_num] = 1
#         full_matrix_item = tm_matrix + sp.eye(original_adj.shape[-1])
#         full_matrix.append(full_matrix_item.todense())

#         graph_dgl.append(
#             dgl.graph((list(full_matrix_item.nonzero()[0]), list(full_matrix_item.nonzero()[1])), num_nodes= original_adj.shape[0])
#         )

#     full_matrix = [torch.tensor(matrix) for matrix in full_matrix]
#     original_adj = torch.stack(full_matrix)

#     graph_dgl.append(
#         dgl.graph((list(range(original_adj.shape[-1])), list(range(original_adj.shape[-1]))), num_nodes=original_adj.shape[-1])
#     )

# def Hemogenizer(adj_matrix):
#         return adj_matrix.sum(0)

# def generator(model, computation_graph, in_features,  num_sam = 10):

#     """use the sample and generate  attiributed graph"""



#     generate_graph = []
#     for sample_i in range(num_sam):
#         std_z, m_z, z, reconstructed_adj_logit, reconstructed_x, reconstructed_labels = model(computation_graph, in_features)
#         reconstructed_adjacency = torch.sigmoid(reconstructed_adj_logit)
#         reconstructed_x_prob = torch.sigmoid(reconstructed_x)
#         reconstructed_labels_prob = torch.sigmoid(reconstructed_labels)
#         graph =reconstructed_adjacency.detach().numpy()
#         graph = descrizer(graph)
#         graph = Hemogenizer(graph)
#         generate_graph.append([graph, reconstructed_x_prob.detach().numpy()])
#     return generate_graph

# def SaveSamples(model, computation_graph, in_features, ref_graph,ref_feature, dir,  num_sam = 10):
#     generate_graph = generator(model, computation_graph, in_features,  num_sam = 10)
#     refrence_graph = []

#     refrence_graph.append([Hemogenizer(ref_graph.detach().numpy()), ref_feature.detach().numpy()])


#     if not os.path.exists(dir):
#         os.makedirs(dir)

#     # np.save(dir + setting+'_generatedGraphs_.npy', generate_graph, allow_pickle=True)
#     # np.save(dir + setting+'refGraphs.npy', refrence_graph, allow_pickle=True)
#     with open(dir + 'generatedGraphs.npy', 'wb') as file:
#         pickle.dump(generate_graph, file)

#     with open(dir + 'refGraphs.npy', 'wb') as file:
#         pickle.dump(refrence_graph, file)

#     stat_rnn.mmd_eval([stat_rnn.to_nx(G[0]) for G in generate_graph], [stat_rnn.to_nx(G[0]) for G in refrence_graph], True)


# model.eval()
# # dir = "GeneratedSamples/"+str(dataset)
# # setting="Rule_reg" if use_motif else "Vanila"
# # dir+=setting+"/"
# # SaveSamples(model, graph_dgl, features,adj_train, features[:,important_feat_ids].float(), dir,setting)
# std_z, m_z, z, reconstructed_adj_logit, reconstructed_x, reconstructed_labels = model(graph_dgl, features)

# reconstructed_adjacency = torch.sigmoid(reconstructed_adj_logit)
# reconstructed_x_prob =   torch.sigmoid(reconstructed_x)
# reconstructed_labels_prob =  torch.sigmoid(reconstructed_labels)




# rules, multiples, states, functors, variables, nodes, masks, base_indices, mask_indices, sort_indices, stack_indices, values, keys, indices, matrices, entities,attributes,relations, prunes = setup_function(database, rule_prune, rule_weight, device)
# metric_ground_truth = iteration_function(device, dataset, heterogeneous_data, rules, multiples, states, functors, variables, nodes, masks, base_indices, mask_indices, sort_indices, stack_indices, values, keys, indices, matrices, entities,attributes,relations , rule_weight, prunes, reconstructed_x_slice = None , reconstructed_labels = None , mode = 'metric_ground_truth')
# reconstructed_x_slice, matrices,reconstructed_labels_m = process_reconstructed_data(device, dataset, heterogeneous_data, mapping_details, reconstructed_adjacency, reconstructed_x_prob, important_feat_ids, matrices,reconstructed_labels_prob)        
# metric_predicted = iteration_function(device, dataset, heterogeneous_data, rules, multiples, states, functors, variables, nodes, masks, base_indices, mask_indices, sort_indices, stack_indices, values, keys, indices, matrices, entities,attributes,relations , rule_weight, prunes, reconstructed_x_slice, reconstructed_labels_m ,mode = 'predicted')



# closeness = torch.sqrt(F.mse_loss(torch.stack(metric_ground_truth), torch.stack(metric_predicted)))



# print('closensee = ', closeness)



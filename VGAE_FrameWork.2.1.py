
import time
start_time = time.monotonic()
import dgl
import classification as CL
from input_data import load_data
from mask_test_edges import mask_test_edges, roc_auc_estimator
import plotter
import argparse
import networkx as nx
from models import *

np.random.seed(0)
random.seed(0)
torch.seed()
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch._set_deterministic(True)

# batch_norm
# edge_Type
# decoder_type and encoder_type
# modelpath
parser = argparse.ArgumentParser(description='VGAE Framework')

parser.add_argument('-e', dest="epoch_number", type=int, default=200, help="Number of Epochs")
parser.add_argument('-v', dest="Vis_step", type=int, default=180, help="model learning rate")
parser.add_argument('-lr', dest="lr", type=float, default=0.001, help="number of epoch at which the error-plot is visualized and updated")
parser.add_argument('-dataset', dest="dataset", default="ACM",
                    help="possible choices are: cora, citeseer, pubmed, IMDB, DBLP, ACM")
parser.add_argument('-hemogenize', dest="hemogenize", default=True, help="either withhold the layers (edges types) during training or not")
parser.add_argument('-NofRels', dest="num_of_relations", type=int, default=1, help="Number of latent layers (L)")
parser.add_argument('-NofCom', dest="num_of_comunities",type=int, default=64,
                    help="Dimention of Z, i.e len(Z[0]), in the bottleNEck")
parser.add_argument('-encoder_layers', dest="encoder_layers", default="64", type=str,
                    help="a list in which each element determine the gcn size; Note: the last layer size is determine with -NofCom")
parser.add_argument('-s', dest="save_embeddings_to_file", default=False, help="save the latent vector of nodes")
parser.add_argument('-f', dest="use_feature", default=True, help="either use features or identity matrix")
parser.add_argument('-DR', dest="DropOut_rate", default=.3, help="drop out rate")
parser.add_argument('-BN', dest="batch_norm", default=True,
                    help="either use batch norm at decoder; only apply in multi relational decoders")
parser.add_argument('-Split', dest="split_the_data_to_train_test", default=True,
                    help="either use features or identity matrix; for synthasis data default is False")
parser.add_argument('-decoder_type', dest="decoder_type", default="MultiLatetnt_SBM_decoder",
                    help="the decoder type, Either SBM , InnerDot , MultiLatetnt_SBM_decoder, MultiLatentLayerGraphit, multi_inner_product, ")
parser.add_argument('-encoder_type', dest="encoder_type", default="mixture_of_GCNs",
                    help="the encoder type, Either ,mixture_of_GCNs, mixture_of_sRGCNs,mixture_of_sGCNs, mixture_of_GatedGCNs , Multi_GCN or Edge_GCN ")
parser.add_argument('-edge_type_visulizer', dest="edge_type_visulizer", default=True,
                    help="either visualize the inferenced edge_type for each node or not; only possibel for mixture of decoders model inclused MapedInnerProduct_SBM and multi_inner_product")
parser.add_argument('-num_node', dest="num_node", default=-1, type=str,
                    help="the size of subgraph which is sampled; -1 means use the whole graph; I added this feature to enable us to train on a small subset of graph to speed up Dev")
parser.add_argument('-modelpath', dest="mpath", default="VGAE_FrameWork_MODEL", type=str,
                    help="The pass to save the learned model")




args = parser.parse_args()

# setting
print("VGAE FRAMEWORK SETING: " + str(args))
visulizer_step = args.Vis_step
epoch_number = args.epoch_number
lr = args.lr
hemogenized = args.hemogenize
edge_type_visulizer = args.edge_type_visulizer
PATH = args.mpath
# hidden_1 = 32  # the size of first graph convolution
num_of_relations = args.num_of_relations  # diffrent type of relation
num_of_comunities = args.num_of_comunities  # number of comunities;
DropOut_rate = args.DropOut_rate
batch_norm = args.batch_norm
dataset = args.dataset  # possible choices are: cora, citeseer, karate, pubmed, DBIS
decoder = args.decoder_type
encoder = args.encoder_type
encoder_layers = [int(x) for x in args.encoder_layers.split()]
use_feature = args.use_feature
save_embeddings_to_file = args.save_embeddings_to_file
subgraph_size = args.num_node
if dataset in {"facebook_egoes"}:  # using subgraph will result in exception
    subgraph_size = -1

split_the_data_to_train_test = args.split_the_data_to_train_test

synthesis_graphs = {"grid", "community", "lobster", "ego"}

# priorDist is prior dist i.e p(Z); 'norm' indicate Normal Distribution and 'vmf' is spherical"
if encoder in ("mixture_of_sGCNs", "mixture_of_sRGCNs"):
    priorDist= "vmf"
else: priorDist= "norm"

#since the datasets are limted We harcoded the var ToDo: fix it.
num_obs = 1
if hemogenized == False:
    num_obs = 3




# ************************************************************
# VGAE frame_work
class GVAE_FrameWork(torch.nn.Module):
    def __init__(self, numb_of_rel, encoder, decoder):
        """
        :param latent_dim: the dimention of each embedded node; |z| or len(z)
        :param numb_of_rel:
        :param decoder:
        :param encoder:
        :param mlp_decoder: either apply an multi layer perceptorn on each decoeded embedings
        """
        super(GVAE_FrameWork, self).__init__()
        # self.relation_type_param = torch.nn.ParameterList(torch.nn.Parameter(torch.Tensor(2*latent_space_dim)) for x in range(latent_space_dim))
        self.numb_of_rel = numb_of_rel
        self.decoder = decoder
        self.encoder = encoder

        self.dropout = torch.nn.Dropout(0)

        # self.mlp_decoder = torch.nn.ModuleList([edge_mlp(2*latent_space_dim,[16,8,1]) for i in range(self.numb_of_rel)])

    def forward(self, adj, x):
        z, m_z, std_z = self.inference(adj, x)
        z = self.dropout(z)
        generated_adj = self.generator(z,x)
        return std_z, m_z, z, generated_adj

    # inference model q(z|adj,x)
    def inference(self, adj, x):
        z, m_q_z, std_q_z = self.encoder(adj, x)
        return z, m_q_z, std_q_z

    # generative model p(adj|z)
    def generator(self, z, x):
        if type(self.decoder) == graphitDecoder or type(self.decoder) == MultiLatentLayerGraphit:
            adj = self.decoder(z, x)
        else: adj = self.decoder(z)
        return adj

    def reparameterize(self, mean, std):
        eps = torch.randn_like(std)
        return eps.mul(std).add(mean)
# ************************************************************

# objective Function
def OptimizerVAE(pred, labels, std_z, mean_z, num_nodes, pos_wight, norm,
                 indexes_to_ignore=None, val_edge_idx=None):
    val_poterior_cost = 0

    if indexes_to_ignore != None:
        posterior_cost = norm * F.binary_cross_entropy_with_logits(pred, labels, pos_weight=pos_wight,
                                                                   reduction='none')
        val_poterior_cost = posterior_cost[val_edge_idx].mean()
        posterior_cost[indexes_to_ignore] = 0  # masking train and test edges
        # posterior_cost[indexes_to_ignore[1], indexes_to_ignore[0]] = 0
        posterior_cost = posterior_cost.mean()

    else:
        posterior_cost = norm * F.binary_cross_entropy_with_logits(pred, labels, pos_weight=pos_wight)


    if priorDist=="norm":
        z_kl = (-0.5 / num_nodes) * torch.mean(torch.sum(1 + 2 * torch.log(std_z) - mean_z.pow(2) - (std_z).pow(2), dim=1))
    else:
        q_z = VonMisesFisher(mean_z, std_z)
        p_z = HypersphericalUniform(mean_z.shape[-1] - 1)

        z_kl = (0.5 / num_nodes) * torch.distributions.kl.kl_divergence(q_z, p_z).mean()
        # z_kl = torch.tensor(0)

    acc = (torch.sigmoid(pred).round() == labels).sum() / float(pred.shape[0] * pred.shape[1])
    return z_kl, posterior_cost, acc, val_poterior_cost


# ============================================================
# The main procedure

# load the data
if dataset in ('grid', 'community', 'ego', 'lobster'):
    synthetic = True
    original_adj, features = load_data(dataset)
    node_label = edge_labels = circles = None
else:
    synthetic = False
    original_adj, features, node_label, edge_labels, circles = load_data(dataset)

# shuffling the data, and selecting a subset of it; subgraph_size is used to do the ecperimnet on the samller dataset to insclease development speed
if subgraph_size == -1:
    subgraph_size = original_adj.shape[0]
elemnt = min(original_adj.shape[0], subgraph_size)
indexes = list(range(original_adj.shape[0]))

#-----------------------------------------
# adj , feature matrix and  node labels  permutaion
np.random.shuffle(indexes)
indexes = indexes[:elemnt]
original_adj = original_adj[indexes, :]
original_adj = original_adj[:, indexes]

features = features[indexes]

if synthetic != True:
    if node_label != None:
        node_label = [node_label[i] for i in indexes]
    if edge_labels != None:
        edge_labels = edge_labels[indexes, :]
        edge_labels = edge_labels[:, indexes]
    if circles != None:
        shuffles_cir = {}
        for ego_node, circule_lists in circles.items():
            shuffles_cir[indexes.index(ego_node)] = [[indexes.index(x) for x in circule_list] for circule_list in
                                                     circule_lists]
        circles = shuffles_cir
#-----------------------------------------

# Check for Encoder and redirect to appropriate function
if encoder == "Multi_GCN":
    encoder_model = multi_layer_GCN(in_feature=features.shape[1], latent_dim=num_of_comunities, layers=encoder_layers)
elif encoder == "mixture_of_GCNs":
    encoder_model = mixture_of_GCNs(in_feature=features.shape[1], num_relation=num_of_relations,
                                    latent_dim=num_of_comunities, layers=encoder_layers, DropOut_rate=DropOut_rate)
elif encoder == "mixture_of_NGCNs":
    encoder_model = mixture_of_NGCNs(in_feature=features.shape[1], num_relation=num_of_relations,
                                    latent_dim=num_of_comunities, layers=encoder_layers, DropOut_rate=DropOut_rate)
elif encoder == "mixture_of_sGCNs":
    encoder_model = mixture_of_sGCNs(in_feature=features.shape[1], num_relation=num_of_relations,
                                    latent_dim=num_of_comunities, layers=encoder_layers, DropOut_rate=DropOut_rate)
elif encoder == "mixture_of_sRGCNs":
    encoder_model = mixture_of_sRGCNs(in_feature=features.shape[1], num_latent_relation=num_of_relations,num_observed_relation=num_obs,
                                    latent_dim=num_of_comunities, layers=encoder_layers, DropOut_rate=DropOut_rate)

else:
    raise Exception("Sorry, this Encoder is not Impemented; check the input args")

# Check for Decoder and redirect to appropriate function
# if decoder == ""
if decoder == "MultiLatetnt_SBM_decoder":
    decoder_model = MultiLatetnt_SBM_decoder(num_of_relations, num_of_comunities, num_of_comunities, batch_norm, DropOut_rate)
elif decoder == "graphitDecoder":
    decoder_model = graphitDecoder(features.shape[1], num_of_comunities)
elif decoder == "MultiLatentLayerGraphit":
    decoder_model = MultiLatentLayerGraphit(features.shape[1], num_of_comunities, num_of_relations,DropOut_rate)
elif decoder == "multi_inner_product":
    decoder_model = MapedInnerProductDecoder([32, 32], num_of_relations, num_of_comunities, batch_norm, DropOut_rate)
elif decoder == "InnerDot":
    decoder_model = InnerProductDecoder()
else:
    raise Exception("Sorry, this Decoder is not Impemented; check the input args")

# instead of X used I if the switch is on
if use_feature == False:
    features = torch.eye(features.shape[0])
    features = sp.csr_matrix(features)

# make train, test and val according to kipf original implementation
if split_the_data_to_train_test == True:
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, train_true, train_false, ignore_edges_inx, val_edge_idx = mask_test_edges(
        original_adj)
    ignore_dges = []
else:
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, train_true, train_false, ignore_edges_inx, val_edge_idx = mask_test_edges(
        original_adj)
    train_edges = val_edges = val_edges_false = test_edges = test_edges_false = ignore_edges_inx = val_edge_idx = None
    adj_train = original_adj

# I use this mudule to plot error and loss
pltr = plotter.Plotter(functions=["loss",  "Accuracy", "Recons Loss", "KL", "AUC"])

adj_train = adj_train+ sp.eye(adj_train.shape[0])# the library does not add self-loops


if hemogenized != True:
    edge_relType = edge_labels * adj_train
    edge_relType = edge_relType + sp.eye(adj_train.shape[0]) * (len(np.unique(edge_relType.data)) + 1)
    graph_dgl = []
    for rel_num in range(num_obs):
        tm_mtrix = csr_matrix(edge_relType.shape)
        tm_mtrix[edge_relType == (rel_num+1)] = 1
        graph_dgl.append(dgl.graph((list(tm_mtrix.nonzero()[0]), list(tm_mtrix.nonzero()[1])),num_nodes=adj_train.shape[0]))

    graph_dgl.append(dgl.from_scipy(adj_train))
else:
    graph_dgl = dgl.from_scipy(adj_train)





num_nodes = adj_train.shape[0]
adj_train = torch.tensor(adj_train.todense())  # use sparse man

if (type(features) == np.ndarray):
    features = torch.tensor(features, dtype=torch.float32)
else:
    features = torch.tensor(features.todense(), dtype=torch.float32)

model = GVAE_FrameWork(num_of_relations, encoder=encoder_model,
                       decoder=decoder_model)  # parameter namimng, it should be dimentionality of distriburion

optimizer = torch.optim.Adam(model.parameters(), lr)

pos_wight = torch.true_divide((adj_train.shape[0] ** 2 - torch.sum(adj_train)), torch.sum(
    adj_train))  # addrressing imbalance data problem: ratio between positve to negative instance

norm = torch.true_divide(adj_train.shape[0] * adj_train.shape[0],
                         ((adj_train.shape[0] * adj_train.shape[0] - torch.sum(adj_train)) * 2))

best_recorded_validation = None
best_epoch = 0

print(model)
for epoch in range(epoch_number):
    model.train()
    # forward propagation by using all nodes
    std_z, m_z, z, reconstructed_adj = model(graph_dgl, features)
    # compute loss and accuracy
    z_kl, reconstruction_loss, acc, val_recons_loss = OptimizerVAE(reconstructed_adj,
                                                                   adj_train ,
                                                                   std_z, m_z, num_nodes, pos_wight, norm,
                                                                    ignore_edges_inx,
                                                                   val_edge_idx)
    loss = reconstruction_loss + z_kl

    reconstructed_adj = torch.sigmoid(reconstructed_adj).detach().numpy()
    model.eval()
    train_auc, train_acc, train_ap, train_conf = roc_auc_estimator(train_true, train_false,
                                                          reconstructed_adj, original_adj)

    if split_the_data_to_train_test == True:
        val_auc, val_acc, val_ap, val_conf = roc_auc_estimator(val_edges, val_edges_false,
                                                        reconstructed_adj, original_adj)

        # keep the history to plot
        pltr.add_values(epoch, [loss.item(), train_acc,  reconstruction_loss.item(), z_kl, train_auc],
                        [None, val_acc, val_recons_loss.item(),None, val_auc  # , val_ap
                            ], redraw=False)  # ["Accuracy", "Loss", "AUC", "AP"]
    else:
        # keep the history to plot
        pltr.add_values(epoch, [acc, loss.item(), None  # , None
                                ],
                        [None, None, None  # , None
                         ], redraw=False)  # ["Accuracy", "loss", "AUC", "AP"])

    model.train()
    # backward propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Ploting the recinstructed Graph
    if epoch % visulizer_step == 0:
        pltr.redraw()
        print("Val conf:", )
        print(val_conf, )
        print("Train Conf:")
        print(train_conf)
        # WholeTrain_auc, TT_acc, WholeTrain_ap, WholeTrain_conf = roc_auc_estimator(None, None,
        #                                                                            reconstructed_adj, original_adj)
        # print("All_train_instances_acc: {:5f} | All_train_instances_AUC: {:5f} | All_train_instances_AP: {:5f}".format(TT_acc, WholeTrain_auc, WholeTrain_ap))
        # print(" All_train_instances Train Conf:")
        # print(WholeTrain_conf)

    if epoch % visulizer_step == 0 and dataset in synthesis_graphs:
        reconstructed_adj[reconstructed_adj >= 0.5] = 1
        reconstructed_adj[reconstructed_adj < 0.5] = 0
        G = nx.from_numpy_matrix(reconstructed_adj)
        plotter.plotG(G, "generated" + dataset)

    # print some metrics
    print("Epoch: {:03d} | Loss: {:05f} | Reconstruction_loss: {:05f} | z_kl_loss: {:05f} | Accuracy: {:03f}".format(
        epoch + 1, loss.item(), reconstruction_loss.item(), z_kl.item(), acc), " | AUC:{:5f}".format(train_auc),
        " | AP:{:5f}".format(train_ap))
    if split_the_data_to_train_test == True:
        print("Val_acc: {:5f} | Val_AUC: {:5f} | Val_AP: {:5f}".format(val_acc, val_auc, val_ap))
        if best_recorded_validation == None or best_recorded_validation < val_auc:  # saving the best model
            best_recorded_validation = val_auc
            best_epoch = epoch
            torch.save(model, PATH)

# save the log plot on the current directory
pltr.save_plot("VGAE_Framework_log_plot")
model.eval()

# #Loading the best model
# if dataset not in synthesis_graphs and split_the_data_to_train_test == True:
#     model = torch.load(PATH)

print("the best Elbow on validation is " + str(best_recorded_validation) + " at epoch " + str(best_epoch))

# Link Prediction Task
print("=====================================")
print("Result on Link Prediction Task")
_, post_mean, z, re_adj = model(graph_dgl, features)
re_adj = torch.sigmoid(re_adj)
if split_the_data_to_train_test == True:
    auc, val_acc, val_ap, conf_mtrx = roc_auc_estimator(test_edges, test_edges_false, re_adj.detach().numpy(),
                                                        original_adj)
    print("Test_acc: {:03f}".format(val_acc), " | Test_auc: {:03f}".format(auc), " | Test_AP: {:03f}".format(val_ap))
    print("Confusion matrix: \n", conf_mtrx)

# save the node embedding
print("=====================================")
if (save_embeddings_to_file):  # save the embedding on the current directory   # ToDo: write both lamda and z in file
    np.savetxt(dataset + "_embeddings", post_mean.detach().numpy())

# ------------------------------------------


if edge_labels != None and edge_type_visulizer == True and (
        type(model.decoder) in  (MapedInnerProductDecoder, MultiLatetnt_SBM_decoder, MultiLatentLayerGraphit)):# and num_of_relations == 2:
    std_z, m_z, z, reconstructed_adj = model(graph_dgl, features)
    #
    Legend_labl=None
    # if dataset=="ACM":
    #     Legend_labl = ["Non-adjacent", "Paper-Subject", "Paper-Author"]
    # elif dataset=="IMDB":
    #     Legend_labl = ["Non-adjacent", "", ""]
    # elif dataset == "":
    #     Legend_labl = ["Non-adjacent", "", ""]

    if type(model.decoder)==MultiLatentLayerGraphit:
        edge_features = model.decoder.get_edge_features(z,features)
    else:
        edge_features = model.decoder.get_edge_features(z)

    edge_features = [val.detach().numpy() for val in edge_features]


    edge_label = np.array(edge_labels.todense()).flatten()
    # edge_labels = edge_labels+ sp.eye(edge_labels.shape[0])* (len(np.unique(edge_labels.data))+1)
    z = z.detach().numpy()
    pair_features = []
    label = []

    none_edges = []
    samplesize =  len(edge_labels.nonzero()[0])

    index = np.where(edge_labels.todense()==0)
    i = random.sample(range(0, len(index[0])), len(edge_labels.nonzero()[0]))
    j = random.sample(range(0, len(index[0])), len(edge_labels.nonzero()[0]))
    none_edges = [index[0][i], index[1][i]]



    for i,j in zip(*edge_labels.nonzero()):
        pair_features.append(np.concatenate((z[i],z[j])))
        label.append(edge_labels[i,j])

    if True:
        for i,j in zip(*none_edges):
            pair_features.append(np.concatenate((z[i],z[j])))
            label.append(edge_labels[i,j])

    # plot edge embedding for dyed; concating approach
    fname = dataset+str(num_of_relations)+decoder
    rndm_indec = list(range(len(label)))
    random.shuffle(rndm_indec)
    resized = rndm_indec[:int(len(rndm_indec)*.10)]
    if dataset=="IMDB":
        Legend_labl = ["Non-adjacent", "Actor-Movie", "Director-Movie"]
    else:
        Legend_labl = ["Non-adjacent", "Paper-Author", "Paper-Subject"]
    if num_of_relations==1:

        plotter.featureVisualizer( [pair_features[i] for i in resized], [label[i] for i in resized], [label[i] for i in resized], filename = "pairVisulizer"+ fname, per = 10,legend_label=Legend_labl)
        if type(model.decoder)== MultiLatetnt_SBM_decoder:
            Nodelabel = []
            Node_pair_features = []

            _, _, z_, _ = model(graph_dgl, features)
            node_feature = model.decoder.get_node_features(z_)[0].detach().numpy()
            for i,j in zip(*edge_labels.nonzero()):
                Node_pair_features.append(np.concatenate((node_feature[i],node_feature[j])))
                Nodelabel.append(edge_labels[i,j])
            for i,j in zip(*none_edges):
                Node_pair_features.append(np.concatenate((node_feature[i],node_feature[j])))
                Nodelabel.append(edge_labels[i,j])
            plotter.featureVisualizer( [Node_pair_features[i] for i in resized], [Nodelabel[i] for i in resized], [Nodelabel[i] for i in resized], filename = "NodeFeaturepairVisulizer"+ fname, per = 10,legend_label=Legend_labl)



    # labels_test, labels_pred, accuracy, micro_recall, macro_recall, micro_precision, macro_precision, micro_f1, macro_f1, conf_matrix, report = CL.NN(
    #     np.array([pair_features[i] for i in resized]), [label[i] for i in resized])
    #
    # print("Accuracy:{}".format(accuracy),
    #       "Macro_AvgPrecision:{}".format(macro_precision), "Micro_AvgPrecision:{}".format(micro_precision),
    #       "Macro_AvgRecall:{}".format(macro_recall), "Micro_AvgRecall:{}".format(micro_recall),
    #       "F1 - Macro,Micro: {} {}".format(macro_f1, micro_f1),
    #       "confusion matrix:{}".format(conf_matrix))
    #-----------------
    # plot edge embedding for dyed; LL approach
    edges_feature = []
    for i,j in zip(*edge_labels.nonzero()):
        feature = np.array([f_i[i,j] for f_i in edge_features])
        edges_feature.append(feature)

    if True:
        for i,j in zip(*none_edges):
            feature = np.array([f_i[i,j] for f_i in edge_features])
            edges_feature.append(feature)
    if num_of_relations>1:
        plotter.featureVisualizer([edges_feature[i] for i in resized], [label[i] for i in resized],[label[i] for i in resized], filename = "edge_representationVisulizer"+ fname,legend_label=Legend_labl)
    # labels_test, labels_pred, accuracy, micro_recall, macro_recall, micro_precision, macro_precision, micro_f1, macro_f1, conf_matrix, report = CL.NN(
    #     np.array([edges_feature[i] for i in resized]), [label[i] for i in resized])
    #
    # print("Accuracy:{}".format(accuracy),
    #       "Macro_AvgPrecision:{}".format(macro_precision), "Micro_AvgPrecision:{}".format(micro_precision),
    #       "Macro_AvgRecall:{}".format(macro_recall), "Micro_AvgRecall:{}".format(micro_recall),
    #       "F1 - Macro,Micro: {} {}".format(macro_f1, micro_f1),
    #       "confusion matrix:{}".format(conf_matrix))

    #---------------------------------------------
    # dyad classification in inductive setting
    selected_nodes = list(np.unique(np.concatenate(list(edge_labels.nonzero())+none_edges)))
    random.shuffle(selected_nodes)

    train_nodes=set(selected_nodes[:int(len(selected_nodes)*.60)])
    Val_nodes = set(selected_nodes[int(len(selected_nodes)*.60):int(len(selected_nodes)*.75)])
    test_nodes= set(selected_nodes[int(len(selected_nodes)*.75):])
    train_feature = []
    train_label = []
    val_feature = []
    val_label = []
    test_feature = []
    test_label = []
    edges_feature_train = []
    edges_feature_val = []
    edges_feature_test = []
    for i, j in zip(*(np.concatenate((edge_labels.nonzero()[0],none_edges[0])),np.concatenate((edge_labels.nonzero()[1],none_edges[1])))):
        if i in train_nodes and j in train_nodes:
            train_feature.append(np.concatenate((z[i], z[j])))
            edges_feature_train.append(np.array([f_i[i, j] for f_i in edge_features]))
            train_label.append(edge_labels[i, j])
        if i in Val_nodes and j in Val_nodes:
            val_feature.append(np.concatenate((z[i], z[j])))
            edges_feature_val.append(np.array([f_i[i, j] for f_i in edge_features]))
            val_label.append(edge_labels[i, j])
        if i in test_nodes and j in test_nodes:
            test_feature.append(np.concatenate((z[i], z[j])))
            edges_feature_test.append(np.array([f_i[i, j] for f_i in edge_features]))
            test_label.append(edge_labels[i, j])



    print("inductive setting")
    if num_of_relations==1:
        print("Concat model:")
        labels_test, labels_pred, accuracy, micro_recall, macro_recall, micro_precision, macro_precision, micro_f1, macro_f1, conf_matrix, report = CL.NN(
            np.array(train_feature), train_label,
            np.array(val_feature), val_label,
            np.array(test_feature), test_label)

        print("Accuracy:{}".format(accuracy),
              "Macro_AvgPrecision:{}".format(macro_precision), "Micro_AvgPrecision:{}".format(micro_precision),
              "Macro_AvgRecall:{}".format(macro_recall), "Micro_AvgRecall:{}".format(micro_recall),
              "F1 - Macro,Micro: {} {}".format(macro_f1, micro_f1),
              "confusion matrix:{}".format(conf_matrix))

    print("dyad model:")
    if num_of_relations>1:
        labels_test, labels_pred, accuracy, micro_recall, macro_recall, micro_precision, macro_precision, micro_f1, macro_f1, conf_matrix, report = CL.NN(
            np.array(edges_feature_train), train_label,
            np.array(edges_feature_val), val_label,
            np.array(edges_feature_test), test_label)

        print("Accuracy:{}".format(accuracy),
          "Macro_AvgPrecision:{}".format(macro_precision), "Micro_AvgPrecision:{}".format(micro_precision),
          "Macro_AvgRecall:{}".format(macro_recall), "Micro_AvgRecall:{}".format(micro_recall),
          "F1 - Macro,Micro: {} {}".format(macro_f1, micro_f1),
          "confusion matrix:{}".format(conf_matrix))



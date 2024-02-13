


import time
start_time = time.monotonic()
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv as GraphConv
from collections import *
from util import *
from scipy.sparse import csr_matrix
from visualization import *
import torch.nn as nn
from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform

np.random.seed(0)
random.seed(0)
torch.seed()
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# torch._set_deterministic(True)
import pickle as pickle
import scipy.sparse

# ************************************************************
# encoders
# ************************************************************
# This is RGCN, Modeling Relational Data with Graph Convolutional Networks, implemetation without regularizer
class RGCN_layer(torch.nn.Module):
    def __init__(self, in_feature, out=32, numRel=1):
        super(RGCN_layer, self).__init__()
        self.GCNLayer= torch.nn.ModuleList([GraphConv(in_feature, out, activation=None, bias=False, weight=True, allow_zero_in_degree=True) for i in range(numRel)])
        self.numRel = numRel

    def forward(self, A, X):
        mes = []
        for i in range(self.numRel):
            if type(A) == list:
                mes.append(self.GCNLayer[i](A[i],X))
            else:
                mes.append(self.GCNLayer[i](A,X))
        Z = torch.stack(mes, dim=0).sum(0)

        return Z

# ------------------------------------------------------------------

#  This class  create a multi-layer of GCNs, stacked on each other
class GCN(torch.nn.Module):
    def __init__(self, in_feature, layers=[64], drop_out_rate=0):
        """
        :param in_feature: the size of input feature; X.shape()[1]
        :param layers: a list in which each element determine the size of corresponding GCNN Layer.
        """
        super(GCN, self).__init__()
        layers = [in_feature] + layers
        if len(layers) < 1: raise Exception("sorry, you need at least two layer")
        self.ConvLayers = torch.nn.ModuleList(
            GraphConv(layers[i], layers[i + 1], activation=None, bias=False, weight=True) for i in
            range(len(layers) - 1))
        self.dropout = torch.nn.Dropout(drop_out_rate)

    def forward(self, adj, x):
        for conv_layer in self.ConvLayers:
            x = torch.tanh(conv_layer(adj, x))
            x = self.dropout(x)

        return x

#-------------------------------------------------------------

# LL implemenation of multi layer GCNs
class GCN_Encoder(torch.nn.Module):
    def __init__(self, in_feature, latent_dim=32, layers=[64, 64], DropOut_rate=0):
        """
        :param in_feature: the size of input feature; X.shape()[1]
        :param num_relation: Number of Latent Layers
        :param layers: a list in which each element determine the size of corresponding GCNN Layer.
        """
        super(GCN_Encoder, self).__init__()

        self.GCN_layers = GCN(in_feature, layers, DropOut_rate)

        self.q_z_mean = GraphConv(layers[-1] , latent_dim, activation=None, bias=False, weight=True)

        self.q_z_std = GraphConv(layers[-1]  , latent_dim, activation=None, bias=False, weight=True)

    def forward(self, adj, x):

        Z = self.GCN_layers(adj, x)



        m_q_z = self.q_z_mean(adj, Z)
        std_q_z = torch.relu(self.q_z_std(adj, Z)) + .0001
        # m_q_z = self.q_z_mean( Z, activation= lambda a : a)
        # std_q_z = torch.relu(self.q_z_std( Z, activation= lambda a : a)) + .0001

        z = self.reparameterize(m_q_z, std_q_z)
        return z, m_q_z, std_q_z,

    def reparameterize(self, mean, std):
        eps = torch.randn_like(std)
        return eps.mul(std).add(mean)

#-------------------------------------------------------------

#-------------------------------------------------------------
 # this class can be used to creae a atack of RGCN layers
class RGCN(torch.nn.Module):
    def __init__(self, in_feature, rel_num,  layers=[64], drop_out_rate=0):
        """
        :param in_feature: the size of input feature; X.shape()[1]
        :param rel_num: Number of Observed Layers (L')
        :param layers: a list in which each element determine the size of corresponding GCNN Layer.
        """
        super(RGCN, self).__init__()
        layers = [in_feature] + layers
        if len(layers) < 1: raise Exception("sorry, you need at least two layer")

        self.ConvLayers = torch.nn.ModuleList(
            RGCN_layer(layers[i], layers[i + 1], rel_num) for i in range(len(layers) - 1))
        self.dropout = torch.nn.Dropout(drop_out_rate)

    def forward(self, adj, x):

        for conv_layer in self.ConvLayers:
            x = torch.tanh(conv_layer(adj, x))
            x = self.dropout(x)

        return x

# LL implementation of S-VGAE+
class RGCN_Encoder(torch.nn.Module):
    def __init__(self, in_feature,  num_relation, latent_dim=32, layers=[64, 64], DropOut_rate=0):
        """
        :param in_feature: the size of input feature; X.shape()[1]
        :param num_relation: Number of  Layers or edge types (L)
        :param layers: a list in which each element determine the size of corresponding GCNN Layer.
        """
        # num_relation = 1
        super(RGCN_Encoder, self).__init__()

        self.RGCNs = RGCN(in_feature, num_relation, layers, DropOut_rate)

        self.q_z_mean = RGCN_layer(layers[-1] , latent_dim,num_relation)
        self.q_z_std = RGCN_layer(layers[-1] , 1,num_relation)

    def forward(self, adj, x, edge_type=None):

        Z = self.RGCNs(adj, x)

        m_q_z = self.q_z_mean(adj, Z)
        m_q_z = m_q_z / (.0001+m_q_z.norm(dim=-1, keepdim=True))

        # the `+ 1` prevent collapsing behaviors
        std_q_z = F.softplus(self.q_z_std( adj,Z)) + 1

        z = self.reparameterize(m_q_z, std_q_z)
        return z, m_q_z, std_q_z,

    def reparameterize(self, mean, std):
        q_z = VonMisesFisher(mean, std)
        return q_z.rsample()




# ************************************************************
# decoders
# ************************************************************


# ------------------------------------------------------------------

class MultiRelational_SBM_decoder(torch.nn.Module):
    """This decoder is implemetation of DGLFRM decoder id which A = f(Z) Lambda F(Z)^{t} """

    def __init__(self, number_of_rel, Lambda_dim, in_dim, normalize, DropOut_rate, node_trns_layers= [32] ):
        """
        :param in_dim: the size of input feature; X.shape()[1]
        :param number_of_rel: Number of Latent Layers
        :param Lambda_dim: dimention of Lambda matrix in sbm model, or the dimention of Z in the decoder after final transformation
        :param node_trns_layers: a list in which each element determine the size of corresponding GCNN Layer.
        :param normalize: bool which indicate either use norm layer or not
        """
        super(MultiRelational_SBM_decoder, self).__init__()

        self.nodeTransformer = torch.nn.ModuleList(
            node_mlp(in_dim, node_trns_layers +[Lambda_dim], normalize, DropOut_rate) for i in range(number_of_rel))

        self.lambdas = torch.nn.ParameterList(
            torch.nn.Parameter(torch.Tensor(Lambda_dim, Lambda_dim)) for i in range(number_of_rel))
        self.numb_of_rel = number_of_rel
        self.reset_parameters()

    def reset_parameters(self):
        for i, weight in enumerate(self.lambdas):
            self.lambdas[i] = init.xavier_uniform_(weight)

    def forward(self, in_tensor):
        gen_adj = []
        for i in range(self.numb_of_rel):
            z = self.nodeTransformer[i](in_tensor)
            h = torch.mm(z, (torch.mm(self.lambdas[i], z.t())))
            gen_adj.append(h)
        return torch.stack(gen_adj)

    def get_node_features(self, in_tensor):
        Z = []
        for i in range(self.numb_of_rel):
            Z.append(self.nodeTransformer[i](in_tensor))
        return Z

    def get_edge_features(self, in_ten):
        A = []
        for i in range(self.numb_of_rel):
            z = self.nodeTransformer[i](in_ten)
            layer_i = torch.mm(z, (torch.mm(self.lambdas[i], z.t())))
            A.append(layer_i)
        return A



# ------------------------------------------------------------------
# Added an Inner Product Decoder for reproducing VGAE Framework
class InnerProductDecoder(torch.nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self):
        # , dropout
        super(InnerProductDecoder, self).__init__()
        # self.dropout = dropout

    def forward(self, z):
        adj = torch.mm(z, z.t())
        return adj

# ------------------------------------------------------------------
class MapedInnerProductDecoder(torch.nn.Module):
# we used this decoder in VGAE*, its extention of VGAE in which decoder is definede as DEC(Z) = f(Z)f(Z)^t
    def __init__(self, layers, num_of_relations, in_size, normalize, DropOut_rate):
        """
        :param in_size: the size of input feature; X.shape()[1]
        :param num_of_relations: Number of Latent Layers
        :param layers: a list in which each element determine the size of corresponding GCNN Layer.
        :param normalize: a bool which indicate either use norm layer or not
        """
        super(MapedInnerProductDecoder, self).__init__()
        self.models = torch.nn.ModuleList(
            node_mlp(in_size, layers, normalize, DropOut_rate) for i in range(num_of_relations))

    def forward(self, z, activation = torch.nn.LeakyReLU(0.01)):
        A = []
        for trans_model in self.models:
            tr_z = trans_model(z)
            layer_i = torch.mm(tr_z, tr_z.t(), )
            A.append(layer_i)
        return torch.sum(torch.stack(A), 0)

    def get_edge_features(self, z):
        A = []
        for trans_model in self.models:
            tr_z = trans_model(z)
            layer_i = torch.mm(tr_z, tr_z.t(), )
            A.append(layer_i)
        return A







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

torch._set_deterministic(True)
import pickle as pickle
import scipy.sparse

# ************************************************************
# encoders
# ************************************************************
# This is RGCN, Modeling Relational Data with Graph Convolutional Networks, implemetation without regularizer
class KIARGCN(torch.nn.Module):
    def __init__(self, in_feature, out=32, numRel=1):
        super(KIARGCN, self).__init__()
        self.GCNLayer= torch.nn.ModuleList([GraphConv(in_feature, out, activation=None, bias=False, weight=True, allow_zero_in_degree=True) for i in range(numRel)])
        self.numRel = numRel

    def forward(self, A, X, ed=None):
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
class mixture_of_GCNs(torch.nn.Module):
    def __init__(self, in_feature, num_relation, latent_dim=32, layers=[64, 64], DropOut_rate=0):
        """
        :param in_feature: the size of input feature; X.shape()[1]
        :param num_relation: Number of Latent Layers
        :param layers: a list in which each element determine the size of corresponding GCNN Layer.
        """
        super(mixture_of_GCNs, self).__init__()

        self.gcns = torch.nn.ModuleList(GCN(in_feature, layers, DropOut_rate) for i in range(num_relation))
        # self.fcNN = node_mlp(latent_dim*num_relation,[latent_dim])
        # self.q_z_mean = node_mlp(latent_dim,[latent_dim])
        #
        # self.q_z_std = node_mlp(latent_dim,[latent_dim])
        self.q_z_mean = GraphConv(layers[-1] * num_relation, latent_dim, activation=None, bias=False, weight=True)

        self.q_z_std = GraphConv(layers[-1]  * num_relation, latent_dim, activation=None, bias=False, weight=True)

    def forward(self, adj, x):
        Z = []
        for GCN in self.gcns:
            Z.append(GCN(adj, x))
        # concat embedings
        Z = torch.cat(Z, 1)
        # multi-layer perceptron
        # Z = self.fcNN(Z)

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

# implementation of Graph Normalized Convolutional Network, based on original paper
# Variational Graph Normalized AutoEncoders
# this class create an stack of GNCN layers
class NGCN(torch.nn.Module):
    def __init__(self, in_feature, layers=[64], drop_out_rate=0):
        """
        :param in_feature: the size of input feature; X.shape()[1]
        :param layers: a list in which each element determine the size of corresponding GNCN Layer.
        """
        super(NGCN, self).__init__()
        layers = [in_feature] + layers
        if len(layers) < 1: raise Exception("sorry, you need at least two layer")

        self.ConvLayers = torch.nn.ModuleList(
            GraphConv(layers[i], layers[i + 1], activation=None, bias=False, weight=False) for i in
            range(len(layers) - 1))

        self.trans = torch.nn.ModuleList(
            torch.nn.Linear(layers[i], layers[i + 1]) for i in
            range(len(layers) - 1))

        self.dropout = torch.nn.Dropout(drop_out_rate)

    def forward(self, adj, x, act=None):

        for i, conv_layer in enumerate(self.ConvLayers):
            x = self.trans[i](x)
            x= torch.nn.functional.normalize( x,  dim =1)*1.8
            x = conv_layer(adj, x)
            if act!=None:
                x = act(x)
                x =self.dropout(x)

        return x

#-------------------------------------------------------------

# LL implemenation of multi layer NGCNs
class mixture_of_NGCNs(torch.nn.Module):
    def __init__(self, in_feature, num_relation, latent_dim=32, layers=[64, 64], DropOut_rate=0):
        """
        :param in_feature: the size of input feature; X.shape()[1]
        :param num_relation: Number of Latent Layers
        :param layers: a list in which each element determine the size of corresponding GCNN Layer.
        """
        # num_relation = 1
        super(mixture_of_NGCNs, self).__init__()
        self.gcns = None
        if len(layers)>0:# if there is prppagation layer before the mean and strd layer; the original paper only have std and mean layer
            self.gcns = torch.nn.ModuleList(NGCN(in_feature, layers, DropOut_rate) for i in range(num_relation))
        else:
            layers=[in_feature]

        self.q_z_mean = NGCN(layers[-1] * num_relation, [latent_dim])
        # self.q_z_mean = torch.nn.ModuleList(NGCN(layers[-1] , [latent_dim]) for i in range(num_relation))

        self.q_z_std = GraphConv(layers[-1] * num_relation, latent_dim, activation=None, bias=False, weight=True)
        # self.q_z_std = torch.nn.ModuleList(GraphConv(layers[-1] , latent_dim, activation=None, bias=False, weight=True) for i in range(num_relation))
    def forward(self, adj, x):
        Z = []

        if self.gcns!=None:
            for NGCN in self.gcns:
                Z.append(NGCN(adj, x, act=torch.tanh))
            Z = torch.cat(Z, 1)
        else:
            Z = x

        m_q_z = self.q_z_mean(adj, Z)
        std_q_z = torch.relu(self.q_z_std(adj, Z)) + .0001


        z = self.reparameterize(m_q_z, std_q_z)
        return z, m_q_z, std_q_z,
        # m_q_z = []
        # std_q_z = []
        #
        # for i in range(len(self.q_z_mean)) :
        #     m_q_z.append(self.q_z_mean[i](adj, Z))
        #     std_q_z.append(torch.relu(self.q_z_std[i](adj, Z)) + .0001)

        # m_q_z = torch.cat(m_q_z, 1)
        # std_q_z = torch.cat(std_q_z, 1)
        #
        # z = self.reparameterize(m_q_z, std_q_z)
        # return z, m_q_z, std_q_z

    def reparameterize(self, mean, std):
        eps = torch.randn_like(std)
        return eps.mul(std).add(mean)
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
        # self.ConvLayers = torch.nn.ModuleList(
        #     RelGraphConv(layers[i], layers[i + 1], rel_num, bias=False, self_loop=False) for i in range(len(layers) - 1))
        self.ConvLayers = torch.nn.ModuleList(
            KIARGCN(layers[i], layers[i + 1], rel_num) for i in range(len(layers) - 1))
        self.dropout = torch.nn.Dropout(drop_out_rate)

    def forward(self, adj, x, edge_type):

        for conv_layer in self.ConvLayers:
            x = torch.tanh(conv_layer(adj, x, edge_type))
            x = self.dropout(x)

        return x

# LL implementation of S-VGAE+
class mixture_of_sRGCNs(torch.nn.Module):
    def __init__(self, in_feature,  num_latent_relation,num_observed_relation, latent_dim=32, layers=[64, 64], DropOut_rate=0):
        """
        :param in_feature: the size of input feature; X.shape()[1]
        :param num_latent_relation: Number of Latent Layers (L)
        :param num_observed_relation: Number of Observed Layers (L')
        :param layers: a list in which each element determine the size of corresponding GCNN Layer.
        """
        # num_relation = 1
        super(mixture_of_sRGCNs, self).__init__()

        self.rgcns = torch.nn.ModuleList(RGCN(in_feature, num_observed_relation, layers, DropOut_rate) for i in range(num_latent_relation))
        # self.fcNN = node_mlp(latent_dim*num_relation,[latent_dim])
        # self.q_z_mean = node_mlp(latent_dim,[latent_dim])
        #
        # self.q_z_std = node_mlp(latent_dim,[latent_dim])
        # self.q_z_mean = RelGraphConv(layers[-1] * num_latent_relation, latent_dim, num_observed_relation)
        #
        # self.q_z_std = RelGraphConv(layers[-1]  * num_latent_relation, latent_dim, num_observed_relation)
        # self.q_z_mean = GraphConv(layers[-1] * num_latent_relation, latent_dim)
        #
        # self.q_z_std = GraphConv(layers[-1]  * num_latent_relation, latent_dim)
        self.q_z_mean = KIARGCN(layers[-1] * num_latent_relation, latent_dim,num_observed_relation)

        self.q_z_std = KIARGCN(layers[-1]  * num_latent_relation, 1,num_observed_relation)

    def forward(self, adj, x, edge_type):
        dropout = torch.nn.Dropout(0)
        Z = []
        for RGCN in self.rgcns:
            Z.append(RGCN(adj, x, edge_type))
        # concat embedings
        Z = torch.cat(Z, 1)
        # multi-layer perceptron
        # Z = self.fcNN(Z)



        m_q_z = self.q_z_mean(adj, Z)
        m_q_z = m_q_z / (.0001+m_q_z.norm(dim=-1, keepdim=True))
        # the `+ 1` prevent collapsing behaviors
        std_q_z = F.softplus(self.q_z_std( adj,Z)) + 1

        z = self.reparameterize(m_q_z, std_q_z)
        return z, m_q_z, std_q_z,

    def reparameterize(self, mean, std):
        q_z = VonMisesFisher(mean, std)
        # p_z = HypersphericalUniform(self.z_dim - 1)
        return q_z.rsample()



#-------------------------------------------------------------

# this class is LL implementation of hyperspherical VAE
# original approach can be found at Hyperspherical Variational Auto-Encoders
class mixture_of_sGCNs(torch.nn.Module):
    def __init__(self, in_feature, num_relation, latent_dim=32, layers=[64, 64], DropOut_rate=0):
        """
        :param in_feature: the size of input feature; X.shape()[1]
        :param num_relation: Number of Latent Layers
        :param layers: a list in which each element determine the size of corresponding GCNN Layer.
        """
        super(mixture_of_sGCNs, self).__init__()

        self.gcns = torch.nn.ModuleList(GCN(in_feature, layers, DropOut_rate) for i in range(num_relation))
        # self.fcNN = node_mlp(latent_dim*num_relation,[latent_dim])
        # self.q_z_mean = node_mlp(latent_dim,[latent_dim])
        #
        # self.q_z_std = node_mlp(latent_dim,[latent_dim])
        # self.q_z_mean = nn.Linear(layers[-1] * num_relation, latent_dim, )
        #
        # self.q_z_std = nn.Linear(layers[-1]  * num_relation, 1)
        self.q_z_mean = GraphConv(layers[-1] * num_relation, latent_dim, activation=None, bias=False, weight=True)

        self.q_z_std = GraphConv(layers[-1]  * num_relation, 1, activation=None, bias=False, weight=True)

    def forward(self, adj, x):
        dropout = torch.nn.Dropout(0)
        Z = []
        for GCN in self.gcns:
            Z.append(GCN(adj, x))
        # concat embedings
        Z = torch.cat(Z, 1)
        # multi-layer perceptron
        # Z = self.fcNN(Z)

        z_mean = self.q_z_mean(adj, Z)
        z_mean = z_mean / (.0001+z_mean.norm(dim=-1, keepdim=True))
        # the `+ 1` prevent collapsing behaviors
        z_var = F.softplus(self.q_z_std( adj,Z)) + 1
        z = self.reparameterize(z_mean, z_var)
        return z, z_mean, z_var,

    def reparameterize(self, mean, std):
        q_z = VonMisesFisher(mean, std)
        # p_z = HypersphericalUniform(self.z_dim - 1)
        return q_z.rsample()


# ************************************************************
# decoders
# ************************************************************

class MultiLatentLayerGraphit(torch.nn.Module):
    """Decoder for using inner product of multiple transformed Z"""

    def __init__(self,  X_dim, Z_dim, num_of_relations, DropOut_rate):
        #
        super(MultiLatentLayerGraphit, self).__init__()

        self.models = torch.nn.ModuleList(
            graphitDecoder(X_dim, Z_dim) for i in range(num_of_relations))
        self.dropout = torch.nn.Dropout(DropOut_rate)

    def forward(self, z, x):
        gen_adj = []
        for i, model in enumerate(self.models):
            tr_z = model(z,x)
            gen_adj.append(tr_z)
        return torch.sum(torch.stack(gen_adj), 0)

    def get_edge_features(self, z,x):
        gen_adj = []
        for i, model in enumerate(self.models):
            tr_z = model(z,x)
            gen_adj.append(tr_z)

        return gen_adj

class graphitDecoder(torch.nn.Module):
    def __init__(self, X_dim, Z_dim):
        super(graphitDecoder, self).__init__()


        self.GCN1 = GraphConvNN(Z_dim, Z_dim)
        self.GCN2 = GraphConvNN(X_dim, Z_dim)
        self.GCN3 = GraphConvNN(Z_dim, Z_dim)

    def forward(self, Z, X):
        # for i,layer in enumerate(self.layers):
        #     Z = layer(Z,X)
        #     # if i!=(len(self.layers)-1):
        #     #     Z = torch.relu(Z)
        # return torch.matmul(Z,Z.permute(0, 2, 1))
        # A = torch.matmul((Z+1) / (torch.norm((Z+1), dim=2, keepdim=True, p=2) + .001), ((Z+1) / (torch.norm((Z+1), dim=2, keepdim=True, p=2) + .001)).permute(0, 2, 1))
        # A = torch.relu(torch.matmul(Z, Z.permute(0, 2, 1)))


        recon_1 = Z / (torch.norm((Z), dim=1, keepdim=True, p=2))
        recon_2 = torch.ones_like(recon_1)
        recon_2 /= torch.sqrt(torch.sum(recon_2, axis = 1, keepdim=True))


        A = torch.matmul(recon_1, recon_1.permute(1,0)) + torch.matmul(recon_2, recon_2.permute(1,0))
        Z1 = torch.relu(self.GCN1(A,Z)) + torch.relu(self.GCN2(A,X))
        # Z1 = torch.relu(Z1)
        Z2 = self.GCN3(A,Z1)
        # Z2 = torch.sigmoid(Z2)
        Z = .5*Z+.5*Z2
        return torch.matmul(Z, Z.permute(1,0))

# ------------------------------------------------------------------

class MultiLatetnt_SBM_decoder(torch.nn.Module):
    """This decoder is implemetation of DGLFRM decoder id which A = f(Z) Lambda F(Z)^{t} """

    def __init__(self, number_of_rel, Lambda_dim, in_dim, normalize, DropOut_rate, node_trns_layers= [32] ):
        """
        :param in_dim: the size of input feature; X.shape()[1]
        :param number_of_rel: Number of Latent Layers
        :param Lambda_dim: dimention of Lambda matrix in sbm model, or the dimention of Z in the decoder after final transformation
        :param node_trns_layers: a list in which each element determine the size of corresponding GCNN Layer.
        :param normalize: bool which indicate either use norm layer or not
        """
        super(MultiLatetnt_SBM_decoder, self).__init__()

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
            # gen_adj.append((h) * self.p_of_relation(z, i))
            gen_adj.append(h)
            # gen_adj.append(self.mlp_decoder[i](self.to_3D(z)))
        return torch.sum(torch.stack(gen_adj), 0)

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




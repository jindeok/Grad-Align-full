import os.path as osp

from utils import *
from data import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import random_walk
from sklearn.linear_model import LogisticRegression

import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, GraphConv, GCNConv, GINConv, global_mean_pool
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN

from torch.nn import init

import networkx as nx
import numpy as np
import pandas as pd
import copy 

from torch_geometric.utils import *
from torch_geometric.data import NeighborSampler as RawNeighborSampler

from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from tqdm import tqdm
from tqdm import trange
import time
import IPython

'''GraphConv -> add Activation / Normalization '''

class GradAlign:
    def __init__(self, G1, G2, att_s, att_t, k_hop, hid, alignment_dict, alignment_dict_reversed, train_ratio, idx1_dict, idx2_dict, alpha,beta):

        self.G1 = G1
        self.G2 = G2
        self.layer = k_hop
        
        self.att_s = att_s
        self.att_t = att_t
        
        self.epochs = 30       
        self.hid_channel = hid
        
        self.default_weight = 1.0
        
        self.device = torch.device('cpu')    
        #self.device = torch.device('cuda')   
        self.alignment_dict = alignment_dict
        self.alignment_dict_reversed =alignment_dict_reversed

        self.train_ratio = train_ratio
        self.idx1_dict = idx1_dict 
        self.idx2_dict = idx2_dict 

        self.gamma = 1
        self.lp_thresh = 0.7
        
        # average degree ratio
        #self.ratio = ratio
        self.alpha = alpha
        self.beta = beta
        #self.ratio = self.G1.number_of_nodes()/self.G2.number_of_nodes()
        
        #mode config
        self.eval_mode = True
        self.cea_mode = False

        
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # disable temporarily because of error
        
    def run_algorithm(self): #anchor is not considered yet
        
        iteration = 0
        
        #Construct GNN

       # model = myGNN_hidden(len(self.att_s.T), hidden_channels=self.hid_channel, num_layers = self.layer)
       # model = myGCN(len(self.att_s.T), hidden_channels=self.hid_channel, num_layers = self.layer)
        model = myGIN(len(self.att_s.T), hidden_channels=self.hid_channel, num_layers = self.layer)
        
        model = model.to(self.device)
        
        if self.train_ratio == 0:
            seed_list1 = []
            seed_list2 = []
        else:
            seed_list1 = list(np.random.choice(list(self.alignment_dict.keys()), int(self.train_ratio * len(self.alignment_dict)), replace = False))
            seed_list2 = [self.alignment_dict[seed_list1[x]] for x in range(len(seed_list1))]
            self.pre_seed_list1 = seed_list1
            self.pre_seed_list2 = seed_list2
            self.G1, self.G2 = seed_link(seed_list1, seed_list2, self.G1, self.G2)
            
            self.H = self.calculateH(self.gamma)
            
        nx.set_edge_attributes(self.G1, values = self.default_weight, name = 'weight')
        nx.set_edge_attributes(self.G2, values = self.default_weight, name = 'weight')
        
        index = sorted(list(self.G1.nodes()))
        columns = sorted(list(self.G2.nodes()))
        
        start = time.time()
        
        # Start iteration

        while True:
                
            self.attr_norm_s, self.attr_norm_t =self.normalized_attribute(self.G1, self.G2)
            
            index = list(set(index) - set(seed_list1))
            columns = list(set(columns) - set(seed_list2))
            seed_n_id_list = seed_list1 + seed_list2
            if len(columns) == 0 or len(index) == 0:
                break
            if len(self.alignment_dict) == len(seed_list1):
                break
            print('\n ------ The current iteration : {} ------'.format(iteration))
            # GNN Embedding Update
            data_s, x_s, edge_index_s, edge_weight_s = self.convert2torch_data(self.G1, self.attr_norm_s) 
            data_t, x_t, edge_index_t, edge_weight_t = self.convert2torch_data(self.G2, self.attr_norm_t)

            if self.cea_mode == False:
                if iteration == 0:
                    embedding1, embedding2 = self.embedding(seed_list1, seed_list2, iteration, self.epochs, x_s, edge_index_s, edge_weight_s, x_t, edge_index_t, edge_weight_t, model, data_s, data_t)
                    
            elif self.cea_mode == True:
                embedding1, embedding2 = self.embedding(seed_list1, seed_list2, iteration, self.epochs, x_s, edge_index_s, edge_weight_s, x_t, edge_index_t, edge_weight_t, model, data_s, data_t)
           
            # Update graph
            print('\n start adding a seed nodes')
            seed_list1, seed_list2, anchor, S, adj2 = self.AddSeeds(embedding1, embedding2, index, columns, seed_list1, seed_list2, iteration)
            embedding_fin1 = embedding1[self.layer].detach().numpy()
            embedding_fin2 = embedding2[self.layer].detach().numpy()
            if self.cea_mode == True:                
                print('\n Edge augmentation...')
                self.G1, self.G2 = self.EvolveGraph(seed_list1, seed_list2, S)
            iteration += 1
            
            
        # Evaluate Performance
        print("total time : {}sec".format(int(time.time() - start)))
        print('\n Start evaluation...')
        self.Evaluation(seed_list1, seed_list2)
        S_prime, result = self.FinalEvaluation(S, embedding1, embedding2, seed_list1, seed_list2, self.idx1_dict, self.idx2_dict, adj2)
        
        return S, S_prime, seed_list1, seed_list2, result

    
    def convert2torch_data(self, G, att):      
        
        data = from_networkx(G)
        att = torch.from_numpy(att)
        data.x = att
        x, edge_index = data.x.to(self.device), data.edge_index.to(self.device)
        data.edge_attr = data['weight']
        edge_weight = data.edge_attr
        x = x.float()
        edge_weight = edge_weight.float()

        
        return data, x, edge_index, edge_weight
    
    def normalized_attribute(self,G1,G2):
        
        self.degarr_s = normalized_adj(G1)
        self.degarr_t = normalized_adj(G2)
        
        # attr1_norm = self.degarr_s * self.att_s
        # attr2_norm = self.degarr_t * self.att_t
        
        attr1_norm = self.att_s
        attr2_norm = self.att_t     #for ablation
        
        return attr1_norm, attr2_norm
        
    def embedding(self, seed_list1, seed_list2, match_iter, epoch, x_s, edge_index_s, edge_weight_s, x_t, edge_index_t, edge_weight_t, model, data_s, data_t):

        seed_1_idx_list = [self.idx1_dict[a] for a in seed_list1]
        seed_1_idx_list = torch.LongTensor(seed_1_idx_list)        
        seed_2_idx_list = [self.idx2_dict[b] for b in seed_list2]        
        seed_2_idx_list = torch.LongTensor(seed_2_idx_list)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)    
        
        A_s = nx.adjacency_matrix(self.G1)
        A_t = nx.adjacency_matrix(self.G2)
        A_hat_s_list = self.distinctive_loss(A_s.todense())
        A_hat_t_list = self.distinctive_loss(A_t.todense())
        
        t = trange(epoch, desc='EMB')            
        model.train()
        for ep in t:     
            
            total_loss = 0
            
            #for loss test GIN
            embedding_s = model.full_forward(x_s, edge_index_s)
            embedding_t = model.full_forward(x_t, edge_index_t)
            # others
            # embedding_s = model.full_forward(x_s, edge_index_s, edge_weight_s)
            # embedding_t = model.full_forward(x_t, edge_index_t, edge_weight_t)
            
            optimizer.zero_grad()
            loss = 0
            for i, (emb_s, emb_t, A_hat_s, A_hat_t) in enumerate(zip(embedding_s, embedding_t, A_hat_s_list, A_hat_t_list)):
                #multi-layer-loss
                if i == 0:
                    continue                
                consistency_loss_s = self.linkpred_loss(emb_s, A_hat_s)
                consistency_loss_t = self.linkpred_loss(emb_t, A_hat_t)
                loss += consistency_loss_s + consistency_loss_t
   
            loss.backward()
            optimizer.step()                
            total_loss += float(loss)
            t.set_description('EMB (total_loss=%g)' % (total_loss))
            
        #for test GIN
        embedding_s = model.full_forward(x_s, edge_index_s)        
        embedding_t = model.full_forward(x_t, edge_index_t)
        
        # embedding_s = model.full_forward(x_s, edge_index_s, edge_weight_s)        
        # embedding_t = model.full_forward(x_t, edge_index_t, edge_weight_t)
              
        return embedding_s, embedding_t
    
    def calculateH(self, gamma):
        self.H = np.zeros((self.G1.number_of_nodes(),self.G2.number_of_nodes()))
        for i, j in zip(self.pre_seed_list1,self.pre_seed_list2):
            self.H[self.idx1_dict[i],self.idx2_dict[j]] = gamma            
        return self.H

    def AddSeeds(self, embedding1, embedding2, index, columns, seed_list1, seed_list2, iteration):

        S_fin = np.zeros((self.G1.number_of_nodes(),self.G2.number_of_nodes()))
        
        for i, (emb1, emb2) in enumerate(zip(embedding1, embedding2)):            
            S = torch.matmul(F.normalize(emb1), F.normalize(emb2).t())
            S = S.detach().numpy()
            S_fin += (1/(self.layer+1)) * S     
            
        try:
            S_fin = S_fin + self.H
        except:
            print("no prior anchors")
            pass
        
        sim_matrix = np.zeros((len(index) * len(columns), 3))
        for i in range(len(index)):
            for j in range(len(columns)):
                sim_matrix[i * len(columns) + j, 0] = index[i] 
                sim_matrix[i * len(columns) + j, 1] = columns[j]
                sim_matrix[i * len(columns) + j, 2] = S_fin[self.idx1_dict[index[i]],self.idx2_dict[columns[j]]] 
        if len(seed_list1) != 0:
            print("Tversky sim calculation..")
            sim_matrix2 = calculate_Tversky_coefficient(self.G1, self.G2, seed_list1, seed_list2, index, columns, alpha = self.alpha, beta = self.beta)
            sim_matrix[:, 2] *= sim_matrix2[:, 2]
        else:
            sim_matrix2 = 1 # no effect
        sim_matrix = sim_matrix[np.argsort(-sim_matrix[:, 2])]
            
        seed1 = []
        seed2 = []
        len_sim_matrix = len(sim_matrix)
        if len_sim_matrix != 0:            
    
            len_sim_matrix = len(sim_matrix)
           # T = np.max([5, int(len(self.alignment_dict) / 100 * (1.1 ** (iteration)))]) #여기부분도 바꿔볼수잇을듯? "매칭 넘버 펑션"
           # T = align_func(version = 'lin', a = 20, b = 5, i = iteration) 
           # T = align_func(version = 'exp', a = 2, b = 5, i = iteration) 
           # T = align_func(version = 'log', a = 3**(100), b = 5, i = iteration) 
            T = align_func(version = 'const', a = int(len(self.alignment_dict)/15), b = 0, i = iteration) 

            #lin, exp, log
           # T = 1500

            while len(sim_matrix) > 0 and T > 0:
                T -= 1
                node1, node2 = int(sim_matrix[0, 0]), int(sim_matrix[0, 1])
                seed1.append(node1)
                seed2.append(node2)
                sim_matrix = sim_matrix[sim_matrix[:, 0] != node1, :]
                sim_matrix = sim_matrix[sim_matrix[:, 1] != node2, :]
            anchor = len(seed_list1)

        anchor = len(seed_list1)
        seed_list1 += seed1
        seed_list2 += seed2
        print('Add seed nodes : {}'.format(len(seed1)))
        self.Evaluation(seed_list1, seed_list2)
        
        return seed_list1, seed_list2, anchor, S_fin, sim_matrix2
    
    def EvolveGraph(self, seed_list1, seed_list2, S):

        pred1, pred2 = self.cross_link(self.lp_thresh, self.G1, self.G2, seed_list1, seed_list2, S)
        
        print("{} edges are added in total".format(len(pred1)+len(pred2)))
        self.G1.add_edges_from(pred1, weight = self.default_weight)
        self.G2.add_edges_from(pred2, weight = self.default_weight)

        return self.G1, self.G2
        
    def Evaluation(self, seed_list1, seed_list2):
        count = 0

        for i in range(len(seed_list1)):
            try:
                if self.alignment_dict[seed_list1[i]] == seed_list2[i]:
                    count += 1
            except:
                continue

        train_len = int(self.train_ratio * len(self.alignment_dict))
        print('Prediction accuracy  at this iteration : %.2f%%'%(100 * (count-train_len) / (len(seed_list1)-train_len)))        
        print('All accuracy : %.2f%%'%(100*(count / len(self.alignment_dict))))    
        print('All prediction accuracy : %.2f%%'%(100*((count - train_len) /(len(self.alignment_dict)-train_len))))    
        
    def FinalEvaluation(self, S, embedding1, embedding2, seed_list1, seed_list2, idx1_dict, idx2_dict, adj2):
        
        count = 0

        for i in range(len(seed_list1)):
            try:
                if self.alignment_dict[seed_list1[i]] == seed_list2[i]:
                    count += 1
            except:
                continue

        train_len = int(self.train_ratio * len(self.alignment_dict))      
        print('All accuracy : %.2f%%'%(100*(count / len(self.alignment_dict)))) 
        acc = count / len(self.alignment_dict)
        
        #input embeddings are final embedding
        index = list(self.G1.nodes())
        columns = list(self.G2.nodes())
        if self.eval_mode == True:
            adj2 = calculate_Tversky_coefficient_final(self.G1, self.G2, seed_list1, seed_list2, index, columns, alpha = self.alpha, beta = self.beta)
            S_prime = self.adj2S(adj2, self.G1.number_of_nodes(), self.G2.number_of_nodes())
            S *= S_prime
    
        gt_dict = self.alignment_dict
        
        top_1 = top_k(S,1)
        top_5 = top_k(S,5)
        top_10 = top_k(S,10)
        
        top1_eval = compute_precision_k(top_1, gt_dict, idx1_dict, idx2_dict)
        top5_eval = compute_precision_k(top_5, gt_dict, idx1_dict, idx2_dict)
        top10_eval = compute_precision_k(top_10, gt_dict, idx1_dict, idx2_dict)

        print('Success@1 : {:.4f}'.format(top1_eval))
        print('Success@5 : {:.4f}'.format(top5_eval))
        print('Success@10 : {:.4f}'.format(top10_eval))
        
        result = '@1:' + str(round(top1_eval,4)) + ',  @5:' + str(round(top5_eval,4))+ ',  @10:' + str(round(top10_eval,4))+ ',  Acc:'+ str(round(acc,4))
        
        return S, result
        

    def adj2S(self, adj, m, n):
        # m = # of nodes in G_s
        S = np.zeros((m,n))
        index = list(self.G1.nodes())
        columns = list(self.G2.nodes())
        for i in range(m):
            for j in range(n):
                S[self.idx1_dict[index[i]], self.idx2_dict[columns[j]]] = adj[i * n + j, 2]
        return S
    
    def linkpred_loss(self, embedding, A):
        
        pred_adj = torch.matmul(F.normalize(embedding), F.normalize(embedding).t())
        pred_adj = F.normalize((torch.min(pred_adj, torch.Tensor([1]))), dim = 1)
        
        linkpred_losss = (pred_adj - A) ** 2
        linkpred_losss = linkpred_losss.sum() / A.shape[1]
        
        return linkpred_losss
    
    def edge_weight_update(self, G, seed_list):
    
        for seed in seed_list:
            for nbr in list(nx.neighbors(G, seed)):
                G.edges[seed,nbr]['weight'] *= self.p 
                
    def distinctive_loss(self, A):
    
        A_hat_list = []
        A_hat_list.append(None) #empty element for future iteration
        for i in range(len(A)):
            A[i, i] = 1
        A = torch.FloatTensor(A)        
        A_cand = A
        
        for l in range(self.layer):
            
            D_ = torch.diag(torch.sum(A, 0)**(-0.5))         
            A_hat = torch.matmul(torch.matmul(D_,A),D_)
            A_hat = A_hat.float()
            A_hat_list.append(A_hat)    
            A_cand = torch.matmul(A,A_cand)
            A = A + A_cand
            
        return A_hat_list
    
    def cross_link(self, thresh, G1, G2, seed_list1, seed_list2, S):
    
        pred1 = []
        pred2 = []
    
        for i in range(len(seed_list1)):
            for j in range(i+1, len(seed_list1)):
                if (S[self.idx1_dict[seed_list1[i]],self.idx2_dict[seed_list2[i]]] > thresh 
                        and S[self.idx1_dict[seed_list1[j]],self.idx2_dict[seed_list2[j]]] > thresh):
                    
                    if not G1.has_edge(seed_list1[i], seed_list1[j]) and G2.has_edge(seed_list2[i], seed_list2[j]):
                        
                            pred1.append([min(seed_list1[i], seed_list1[j]), max(seed_list1[i], seed_list1[j])])
                        
                    if not G2.has_edge(seed_list2[i], seed_list2[j]) and G1.has_edge(seed_list1[i], seed_list1[j]):
                        pred2.append([min(seed_list2[i], seed_list2[j]), max(seed_list2[i], seed_list2[j])])
        
        return pred1, pred2

    

        
    
class myGNN_hidden(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):        
        super(myGNN_hidden, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(GraphConv(in_channels, hidden_channels, aggr = 'add'))
        init_weight(self.modules())

    def forward(self, x, adjs, edge_weight):
        for i, (edge_index, e_id, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index, edge_weight[e_id])
            if i == self.num_layers - 1:
                x = x.tanh()
           
        return x

    def full_forward(self, x, edge_index, edge_weight):
        emb_list = []
        emb_list.append(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i == self.num_layers - 1:
                x = x.tanh()
            emb_list.append(x)
            
        return emb_list
    
    
class myGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):        
        super(myGCN, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(GCNConv(in_channels, hidden_channels))
        init_weight(self.modules())

    def full_forward(self, x, edge_index, edge_weight):
        emb_list = []
        emb_list.append(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            x = x.tanh()
            emb_list.append(x)
            
        return emb_list
    

class myGIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):        
        super(myGIN, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(in_channels, hidden_channels),
                        ReLU(),
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        BN(hidden_channels),
                    ), train_eps=False, aggr = 'max'))
        init_weight(self.modules())

    def full_forward(self, x, edge_index):
        emb_list = []
        emb_list.append(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = x.tanh()
            emb_list.append(x)
            
        return emb_list
    
    
    
class mySAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):        
        super(mySAGE, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))
        init_weight(self.modules())

    def full_forward(self, x, edge_index, edge_weight):
        emb_list = []
        emb_list.append(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            x = x.tanh()
            emb_list.append(x)
            
        return emb_list
    
    
def init_weight(modules):
    """
    Weight initialization
    :param modules: Iterable of modules
    :param activation: Activation function.
    """
    
    for m in modules:
        if isinstance(m, nn.Linear):
            m.weight.data = init.xavier_uniform_(m.weight.data) #, gain=nn.init.calculate_gain(activation.lower()))
    


def seed_link(seed_list1, seed_list2, G1, G2):
    k = 0
    for i in range(len(seed_list1) - 1):
        for j in range(np.max([1, i + 1]), len(seed_list1)):
            if G1.has_edge(seed_list1[i], seed_list1[j]) and not G2.has_edge(seed_list2[i], seed_list2[j]):
                G2.add_edges_from([[seed_list2[i], seed_list2[j]]])
                k += 1
            if not G1.has_edge(seed_list1[i], seed_list1[j]) and G2.has_edge(seed_list2[i], seed_list2[j]):
                G1.add_edges_from([[seed_list1[i], seed_list1[j]]])
                k += 1
    print('Add seed links : {}'.format(k), end = '\t')
    return G1, G2

def normalized_adj(G):
    # make sure ordering has ascending order
    deg = dict(G.degree)
    deg = sorted(deg.items())
    deglist = [math.pow(b, -0.5) for (a,b) in deg]
    degarr = np.array(deglist)
    degarr = np.expand_dims(degarr, axis = 0)
    return degarr.T


def align_func(version, a, b, i):
    
    if version == "lin":
        return int(a*i + b)
    elif version == "exp":
        return int(a**i + b)
    elif version == "log":    
        return int(math.log(a*i+b) + b)
    elif version == "const":
        return a

def calculate_Tversky_coefficient(G1, G2, seed_list1, seed_list2, index, columns, alpha, beta, alignment_dict = None):
    shift = int(np.max([np.max(G1.nodes()), np.max(G2.nodes())]))
    seed1_dict = {}
    seed1_dict_reversed = {}
    seed2_dict = {}
    seed2_dict_reversed = {}
    for i in range(len(seed_list1)):
        seed1_dict[i + 2 * (shift + 1)] = seed_list1[i]
        seed1_dict_reversed[seed_list1[i]] = i + 2 * (shift + 1)
        seed2_dict[i + 2 * (shift + 1)] = seed_list2[i] + shift + 1
        seed2_dict_reversed[seed_list2[i] + shift + 1] = i + 2 * (shift + 1)
    G1_edges = pd.DataFrame(G1.edges())
    G1_edges.iloc[:, 0] = G1_edges.iloc[:, 0].apply(lambda x:to_seed(x, seed1_dict_reversed))
    G1_edges.iloc[:, 1] = G1_edges.iloc[:, 1].apply(lambda x:to_seed(x, seed1_dict_reversed))
    G2_edges = pd.DataFrame(G2.edges())
    G2_edges += shift + 1
    G2_edges.iloc[:, 0] = G2_edges.iloc[:, 0].apply(lambda x:to_seed(x, seed2_dict_reversed))
    G2_edges.iloc[:, 1] = G2_edges.iloc[:, 1].apply(lambda x:to_seed(x, seed2_dict_reversed))
    adj = nx.Graph()
    adj.add_edges_from(np.array(G1_edges))
    adj.add_edges_from(np.array(G2_edges))
    Tversky_dict = {}
    for G1_node in index:
        for G2_node in columns:
            if (G1_node, G2_node) not in Tversky_dict.keys():
                Tversky_dict[G1_node, G2_node] = 0
            try:
                Tversky_dict[G1_node, G2_node] += calculate_Tversky(adj.neighbors(G1_node), adj.neighbors(G2_node + shift + 1), alpha, beta)
            except:
                continue
    Tversky_dict = [[x[0][0], x[0][1], x[1]] for x in Tversky_dict.items()]
    sim_matrix = np.array(Tversky_dict)
    
    return sim_matrix

def calculate_Tversky_coefficient_final(G1, G2, seed_list1, seed_list2, index, columns, alpha, beta):
    shift = int(np.max([np.max(G1.nodes()), np.max(G2.nodes())]))
    seed1_dict = {}
    seed1_dict_reversed = {}
    seed2_dict = {}
    seed2_dict_reversed = {}
    for i in range(len(seed_list1)):
        seed1_dict[i + 2 * (shift + 1)] = seed_list1[i]
        seed1_dict_reversed[seed_list1[i]] = i + 2 * (shift + 1)
        seed2_dict[i + 2 * (shift + 1)] = seed_list2[i] + shift + 1
        seed2_dict_reversed[seed_list2[i] + shift + 1] = i + 2 * (shift + 1)
    G1_edges = pd.DataFrame(G1.edges())
    G1_edges.iloc[:, 0] = G1_edges.iloc[:, 0].apply(lambda x:to_seed(x, seed1_dict_reversed))
    G1_edges.iloc[:, 1] = G1_edges.iloc[:, 1].apply(lambda x:to_seed(x, seed1_dict_reversed))
    G2_edges = pd.DataFrame(G2.edges())
    G2_edges += shift + 1
    G2_edges.iloc[:, 0] = G2_edges.iloc[:, 0].apply(lambda x:to_seed(x, seed2_dict_reversed))
    G2_edges.iloc[:, 1] = G2_edges.iloc[:, 1].apply(lambda x:to_seed(x, seed2_dict_reversed))
    adj = nx.Graph()
    adj.add_edges_from(np.array(G1_edges))
    adj.add_edges_from(np.array(G2_edges))
    Tversky_dict = {}  
    for G1_node in index:
        for G2_node in columns:
            Tversky_dict[G1_node, G2_node] = 0
            g1 = to_seed(G1_node, seed1_dict_reversed)
            g2 = to_seed(G2_node + shift + 1, seed2_dict_reversed)
            Tversky_dict[G1_node, G2_node] += calculate_Tversky(adj.neighbors(g1), adj.neighbors(g2), alpha, beta)
    Tversky_dict = [[x[0][0], x[0][1], x[1]] for x in Tversky_dict.items()]
    sim_matrix = np.array(Tversky_dict)
    return sim_matrix

def to_seed(x, dictionary):
    try:
        return dictionary[x]
    except:
        return x

def calculate_Tversky(setA, setB, alpha, beta):
    setA = set(setA)
    setB = set(setB)   
    ep = 0.01
        
    inter = len(setA & setB) + ep
    #union = len(setA | setB) + ep
    diffA = len(setA - setB) 
    diffB = len(setB - setA)
     
    Tver = inter / (inter + alpha*diffA + beta*diffB)
    
    return Tver


def top_k(S, k=1):
    """
    S: scores, numpy array of shape (M, N) where M is the number of source nodes,
        N is the number of target nodes
    k: number of predicted elements to return
    """
    top = np.argsort(-S)[:,:k]
    result = np.zeros(S.shape)
    for idx, target_elms in enumerate(top):
        for elm in target_elms:
            result[idx,elm] = 1
        
    return result

def compute_precision_k(top_k_matrix, gt, idx1_dict, idx2_dict):
    n_matched = 0

    if type(gt) == dict:
        for key, value in gt.items():
            if top_k_matrix[idx1_dict[key], idx2_dict[value]] == 1:
                n_matched += 1
        return n_matched/len(gt)
    
    return n_matched/n_nodes
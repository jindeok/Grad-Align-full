import networkx as nx


import torch_geometric.utils.convert as cv
from torch_geometric.data import NeighborSampler as RawNeighborSampler

import pandas as pd
from utils import *
import warnings
import argparse
warnings.filterwarnings('ignore')
import collections
import networkx as nx
import copy 
from sklearn.metrics import roc_auc_score
import os
from models import *
import numpy as np
import random
import torch

'''
Please note that some function codes are apopted and revised from
https://github.com/vinhsuhi/GAlign
https://github.com/deepopo/CENALP
'''

def na_dataloader(args):
    
    if args.mode == 'not_perturbed': 
        # for douban dataset
        G1, G2= loadG(args.data_folder, args.graphname)           
      
        if args.graphname == 'am-td': # will be revised later 
        
            source_dataset = Dataset('dataset\\DataProcessing\\allmv_tmdb\\allmv')
            target_dataset = Dataset('dataset\\DataProcessing\\allmv_tmdb\\tmdb')
            gt_dr = 'dataset\\DataProcessing\\allmv_tmdb\\dictionaries\\groundtruth'
            gt_dict = graph_utils.load_gt(gt_dr, source_dataset.id2idx, target_dataset.id2idx, 'dict')
            gt_dict = DeleteDuplicatedElement(gt_dict)
    
            attr1, attr2, attribute_sim  = AttributeProcessing(args, G1, G2, gt_dict)
            G2, alignment_dict, alignment_dict_reversed = preprocessing(G1, G2, gt_dict)
            idx1_dict, idx2_dict = create_idx_dict_pair(G1,G2,alignment_dict)
            
            return G1, G2, attr1, attr2, alignment_dict, alignment_dict_reversed, idx1_dict, idx2_dict
        
        elif args.graphname == 'fl-my': # will be revised later 
        
            source_dataset = Dataset('dataset\\DataProcessing\\fl-my\\flickr')
            target_dataset = Dataset('dataset\\DataProcessing\\fl-my\\myspace')
            gt_dr = 'dataset\\DataProcessing\\fl-my\\dictionaries\\groundtruth'
            gt_dict = graph_utils.load_gt(gt_dr, source_dataset.id2idx, target_dataset.id2idx, 'dict')
    
            attr1, attr2, attribute_sim  = AttributeProcessing(args, G1, G2, gt_dict)
            G2, alignment_dict, alignment_dict_reversed = preprocessing(G1, G2, gt_dict)
            idx1_dict, idx2_dict = create_idx_dict_pair(G1,G2,alignment_dict)
            
            return G1, G2, attr1, attr2, alignment_dict, alignment_dict_reversed, idx1_dict, idx2_dict

            
        else:        
            
            alignment_dict, alignment_dict_reversed = read_alignment(args.alignment_folder, args.graphname)
            attr1, attr2, attribute_sim = AttributeProcessing(args, G1, G2, alignment_dict)
            G2, alignment_dict, alignment_dict_reversed = preprocessing(G1, G2, alignment_dict)
            idx1_dict, idx2_dict = create_idx_dict_pair(G1,G2,alignment_dict)
            
            return G1, G2, attr1, attr2, alignment_dict, alignment_dict_reversed, idx1_dict, idx2_dict

            
    else: 
        # for perturbed
        G1, G2 = loadG(args.data_folder, args.graphname)
        alignment_dict, alignment_dict_reversed = read_alignment(args.alignment_folder, args.graphname)
        G3, G4, Ggt = PerturbedProcessing(G1, G2, 0, args.edge_portion, args.graphname)
        
        attr1, attr2, attribute_sim = AttributeProcessing(args, G3, G4, alignment_dict)    
        #alignment_dict, alignment_dict_reversed = read_alignment(args.alignment_folder, args.graphname)  
        idx1_dict, idx2_dict = create_idx_dict_pair(G1,G2,alignment_dict)
        
        return G3, G4, attr1, attr2, alignment_dict, alignment_dict_reversed, idx1_dict, idx2_dict
    
    
'''Data preprocessing'''

def Reordering(G1,G2):
    G1 = nx.convert_node_labels_to_integers(G1, first_label=1, ordering='default', label_attribute=None)
    G2 = nx.convert_node_labels_to_integers(G2, first_label=G1.number_of_nodes()+1, ordering='default', label_attribute=None)
    return G1,G2

def ReorderingSame(G1,G2):
    G1 = nx.convert_node_labels_to_integers(G1, first_label=0, ordering='default', label_attribute=None)
    G2 = nx.convert_node_labels_to_integers(G2, first_label=0, ordering='default', label_attribute=None)
    return G1,G2


def PerturbedProcessing(G1, G2, com_portion, rand_portion, graphname):

    G3 = copy.deepcopy(G1)
    #Groundtruth graph
    Ggt = copy.deepcopy(G1)
    #Input 2 graphs
    G3, G4 = perturb_edge_pair(G3, com_portion, rand_portion)
    #G3, G4 = ReorderingSame(G3,G4) #0505 disabled
    
    #export to the files
    nx.write_edgelist(G3, "{}1_ran{}.edges".format(graphname, rand_portion),delimiter=',',data=False)
    nx.write_edgelist(G4, "{}2_ran{}.edges".format(graphname, rand_portion),delimiter=',',data=False)
    print('exporting data complete')
    
    return G3, G4, Ggt


    
def AttributeProcessing(args,G1,G2, alignment_dict):

    attribute_sim, attr1, attr2, attr1_pd, attr2_pd = read_attribute(args.attribute_folder, args.graphname, G1, G2, alignment_dict)

    featpd1 = pd.DataFrame(attr1)
    featpd2 = pd.DataFrame(attr2)
    featnpy1 = featpd1.to_numpy() 
    featnpy2 = featpd2.to_numpy() 
    print(featpd1)
    np.save("feats1.npy",featnpy1)
    np.save("feats2.npy",featnpy2)
    print("feats exporting complete")
    
    return attr1, attr2, attribute_sim



def DeleteDuplicatedElement(gt_dict):
    #for am-td
    dellist = [1759,1702,3208,3255,1311,2126,4249,2892,4657,5738,4990,2076]
    #[1786,1725,3231,3275,1349,2148,5641,2915,4661,5605,4991,2077]
    for key in dellist:
        del gt_dict[key]
    return gt_dict

''' perturbation '''
def perturb_edge_pair(G, com_portion = 0.05, rand_portion = 0.1):
    
    G_copy = copy.deepcopy(G)
    
    edgelist = list(G.edges)
    
    num_mask_common = int(len(edgelist)*com_portion)
    num_mask_rand = int(len(edgelist)*rand_portion)
    
    for _ in range(num_mask_common):
        e = sample(list(edgelist),1)
        
        start_vertex = e[0][0]
        end_vertex = e[0][1]
        
        if G.degree[start_vertex] >= 2 and G.degree[end_vertex] >= 2:
            G.remove_edges_from(e)
            G_copy.remove_edges_from(e)
            
    for _ in range(num_mask_rand):
        e = sample(list(edgelist),1)
        
        start_vertex = e[0][0]
        end_vertex = e[0][1]
        
        if G.degree[start_vertex] >= 2 and G.degree[end_vertex] >= 2:
            G.remove_edges_from(e)
            
    for _ in range(num_mask_rand):
        e = sample(list(edgelist),1)
        
        start_vertex = e[0][0]
        end_vertex = e[0][1]
        
        if G_copy.degree[start_vertex] >= 2 and G_copy.degree[end_vertex] >= 2:
            G_copy.remove_edges_from(e)
            
    return G, G_copy

def perturb_edge_pair_real(G1, G2, dictionary, com_portion = 0.1, rand_portion = 0.05):
    
    edgelist1 = list(G1.edges)
    edgelist2 = list(G2.edges)
    
    edgelist_com = list(set(edgelist1) & set(edgelist2))
    
    num_mask_common = int(len(edgelist2)*com_portion)
    num_mask_rand = int(len(edgelist1)*rand_portion)
    
    for _ in range(num_mask_common):
        e = sample(list(edgelist1),1)
   
        start_vertex = e[0][0]
        end_vertex = e[0][1]
        
        try:
            e2 = [(dictionary[start_vertex], dictionary[end_vertex])]
        except:
            pass
        
        if G1.degree[start_vertex] >= 2 and G1.degree[end_vertex] >= 2:
            G1.remove_edges_from(e)
            try:
                G2.remove_edges_from(e2)
            except:
                pass
            
    for _ in range(num_mask_rand):
        e = sample(list(edgelist1),1)
        
        start_vertex = e[0][0]
        end_vertex = e[0][1]
        
        if G1.degree[start_vertex] >= 2 and G1.degree[end_vertex] >= 2:
            G1.remove_edges_from(e)
            
    for _ in range(num_mask_rand):
        e = sample(list(edgelist2),1)
        
        start_vertex = e[0][0]
        end_vertex = e[0][1]
        
        if G2.degree[start_vertex] >= 2 and G2.degree[end_vertex] >= 2:
            G2.remove_edges_from(e)
            
    return G1, G2

class Dataset:
    """
    this class are copied from the repo: https://github.com/vinhsuhi/GAlign
    this class receives input from graphsage format with predefined folder structure, the data folder must contains these files:
    G.json, id2idx.json, features.npy (optional)

    Arguments:
    - data_dir: Data directory which contains files mentioned above.
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self._load_id2idx()
        self._load_G()
        self._load_features()
        graph_utils.construct_adjacency(self.G, self.id2idx, sparse=False, file_path=self.data_dir + "/edges.edgelist")
        # self.load_edge_features()
        print("Dataset info:")
        print("- Nodes: ", len(self.G.nodes()))
        print("- Edges: ", len(self.G.edges()))

    def _load_G(self):
        G_data = json.load(open(os.path.join(self.data_dir, "G.json")))
        G_data['links'] = [{'source': self.idx2id[G_data['links'][i]['source']], 'target': self.idx2id[G_data['links'][i]['target']]} for i in range(len(G_data['links']))]
        self.G = json_graph.node_link_graph(G_data)


    def _load_id2idx(self):
        id2idx_file = os.path.join(self.data_dir, 'id2idx.json')
        self.id2idx = json.load(open(id2idx_file))
        self.idx2id = {v:k for k,v in self.id2idx.items()}


    def _load_features(self):
        self.features = None
        feats_path = os.path.join(self.data_dir, 'feats.npy')
        if os.path.isfile(feats_path):
            self.features = np.load(feats_path)
        else:
            self.features = None
        return self.features

    def load_edge_features(self):
        self.edge_features= None
        feats_path = os.path.join(self.data_dir, 'edge_feats.mat')
        if os.path.isfile(feats_path):
            edge_feats = loadmat(feats_path)['edge_feats']
            self.edge_features = np.zeros((len(edge_feats[0]),
                                           len(self.G.nodes()),
                                           len(self.G.nodes())))
            for idx, matrix in enumerate(edge_feats[0]):
                self.edge_features[idx] = matrix.toarray()
        else:
            self.edge_features = None
        return self.edge_features

    def get_adjacency_matrix(self, sparse=False):
        return graph_utils.construct_adjacency(self.G, self.id2idx, sparse=False, file_path=self.data_dir + "/edges.edgelist")

    def get_nodes_degrees(self):
        return graph_utils.build_degrees(self.G, self.id2idx)

    def get_nodes_clustering(self):
        return graph_utils.build_clustering(self.G, self.id2idx)

    def get_edges(self):
        return graph_utils.get_edges(self.G, self.id2idx)

    def check_id2idx(self):
        # print("Checking format of dataset")
        for i, node in enumerate(self.G.nodes()):
            if (self.id2idx[node] != i):
                print("Failed at node %s" % str(node))
                return False
        # print("Pass")
        return True
''' file loading '''
def read_alignment(alignment_folder, filename):
    alignment = pd.read_csv(alignment_folder + filename + '.csv', header = None)
    alignment_dict = {}
    alignment_dict_reversed = {}
    for i in range(len(alignment)):
        alignment_dict[alignment.iloc[i, 0]] = alignment.iloc[i, 1]
        alignment_dict_reversed[alignment.iloc[i, 1]] = alignment.iloc[i, 0]
    return alignment_dict, alignment_dict_reversed

def get_reversed(alignment_dict):
    alignment_dict_reversed = {}
    reversed_dictionary = {value : key for (key, value) in alignment_dict.items()}
    return alignment_dict, reversed_dictionary

def read_attribute(attribute_folder, filename, G1, G2, alignment_dict):
    try:
        attribute, attr1, attr2, attr1_pd, attr2_pd = load_attribute(attribute_folder, filename, G1, G2, alignment_dict)
       # attribute, attr1, attr2, attr1_pd, attr2_pd = load_attribute_for_perturb(attribute_folder, filename, G1, G2,alignment_dict, 0.2)
        attribute = attribute.transpose()
    except:
        attr1 = []
        attr2 = []
        attr1_pd = []
        attr2_pd = []
        attribute = []
        print('Attribute files not found.')
    return attribute, attr1, attr2, attr1_pd, attr2_pd




def load_gt(path, id2idx_src=None, id2idx_trg=None, format='matrix'):    
    if id2idx_src:
        conversion_src = type(list(id2idx_src.keys())[0])
        conversion_trg = type(list(id2idx_trg.keys())[0])
    if format == 'matrix':
        # Dense
        """
        gt = np.zeros((len(id2idx_src.keys()), len(id2idx_trg.keys())))
        with open(path) as file:
            for line in file:
                src, trg = line.strip().split()                
                gt[id2idx_src[conversion_src(src)], id2idx_trg[conversion_trg(trg)]] = 1
        return gt
        """
        # Sparse
        row = []
        col = []
        val = []
        with open(path) as file:
            for line in file:
                src, trg = line.strip().split()
                row.append(id2idx_src[conversion_src(src)])
                col.append(id2idx_trg[conversion_trg(trg)])
                val.append(1)
        gt = csr_matrix((val, (row, col)), shape=(len(id2idx_src), len(id2idx_trg)))
    else:
        gt = {}
        with open(path) as file:
            for line in file:
                src, trg = line.strip().split()
                # print(src, trg)
                if id2idx_src:
                    gt[id2idx_src[conversion_src(src)]] = id2idx_trg[conversion_trg(trg)]
                else:
                    gt[str(src)] = str(trg)
    return gt


def loadG(data_folder, filename):

    G1 = nx.Graph()
    G2 = nx.Graph()
    G1_edges = pd.read_csv(data_folder + filename + '1.edges', names = ['0', '1'])
    G1.add_edges_from(np.array(G1_edges))
    G2_edges = pd.read_csv(data_folder + filename + '2.edges', names = ['0', '1'])
    G2.add_edges_from(np.array(G2_edges))
    
    return G1, G2

def loadG_link(data_folder, test_frac, filename):
    G1 = nx.Graph()
    G2 = nx.Graph()
    G1_edges = pd.read_csv(data_folder + filename + '1.edges', names = ['0', '1'])
    G1.add_edges_from(np.array(G1_edges))
    G2_edges = pd.read_csv(data_folder + filename + '2_' + str(test_frac) + '.edges', names = ['0', '1'])
    G2.add_edges_from(np.array(G2_edges))
    test_edges = pd.read_csv(data_folder + filename + '2_' + str(test_frac) + '_test.edges', names = ['0', '1'])
    return G1, G2, test_edges

def load_attribute(attribute_folder, filename, G1, G2, alignment_dict):

    G1_nodes = list(G1.nodes()) 
    G2_nodes = list(G2.nodes())
    
    attribute1_pd = pd.read_csv(attribute_folder + filename + 'attr1.csv', header = None, index_col = 0)
    attribute2_pd = pd.read_csv(attribute_folder + filename + 'attr2.csv', header = None, index_col = 0)
    #print("sadasdas",attribute2)

    attribute1 = np.array(attribute1_pd.loc[G1_nodes, :])
    attribute2 = np.array(attribute2_pd.loc[G2_nodes, :])   
    
    attr_cos = cosine_similarity(attribute1, attribute2)
    return attr_cos, attribute1, attribute2, attribute1_pd, attribute2_pd


def load_attribute_for_perturb(attribute_folder, filename, G1, G2, alignment_dict, percent):
    G1_nodes = list(G1.nodes()) 
    G2_nodes = list(G2.nodes())
    noise = '_' + str(percent)

    attribute1_pd = pd.read_csv(attribute_folder +'noise/' + filename + noise + 'attr1.csv', header = None, index_col = 0)
    attribute2_pd = pd.read_csv(attribute_folder + 'noise/'+ filename + noise + 'attr2.csv', header = None, index_col = 0)
    #print("sadasdas",attribute2)

    attribute1 = np.array(attribute1_pd.loc[G1_nodes, :])
    attribute2 = np.array(attribute2_pd.loc[G2_nodes, :])   
    
    attr_cos = cosine_similarity(attribute1, attribute2)
    return attr_cos, attribute1, attribute2, attribute1_pd, attribute2_pd

def preprocessing(G1, G2, alignment_dict):
    '''
    Parameters
    ----------
    G1 : source graph
    G2 : target graph
    alignment_dict : grth dict

    '''
    # shift index for constructing union
    # construct shifted dict
    shift = G1.number_of_nodes()
    G2_list = list(G2.nodes())
    G2_shiftlist = list(idx + shift for idx in list(G2.nodes()))
    shifted_dict = dict(zip(G2_list,G2_shiftlist))
    
    #relable idx for G2
    G2 = nx.relabel_nodes(G2, shifted_dict)
    
    #update alignment dict
    align1list = list(alignment_dict.keys())
    align2list = list(alignment_dict.values())   
    shifted_align2list = [a+shift for a in align2list]
    
    groundtruth_dict = dict(zip(align1list, shifted_align2list))
    groundtruth_dict, groundtruth_dict_reversed = get_reversed(groundtruth_dict)
    
    return G2, groundtruth_dict, groundtruth_dict_reversed


def create_idx_dict_pair(G1,G2,alignment_dict):
    '''
    Make sure that this function is followed after preprocessing dict.

    '''
    
    G1list = list(G1.nodes())
    #G1list.sort()
    idx1_list = list(range(G1.number_of_nodes()))
    #make dict for G1
    idx1_dict = {a : b for b, a in zip(idx1_list,G1list)}

    
    G2list = list(G2.nodes())
    #G2list.sort()
    idx2_list = list(range(G2.number_of_nodes()))
    #make dict for G2
    idx2_dict = {c : d for d, c in zip(idx2_list,G2list)}
    
    return idx1_dict, idx2_dict
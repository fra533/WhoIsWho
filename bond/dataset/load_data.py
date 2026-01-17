import os
import json
import codecs
import torch
import random
import pickle
import numpy as np
from os.path import join
from params import set_params
from torch_geometric.data import Data

args = set_params()

def load_data(rfname):
    with open(rfname, 'rb') as rf:
        return pickle.load(rf)


def dump_data(obj, wfname):
    with open(wfname, 'wb') as wf:
        pickle.dump(obj, wf)
       
        
def load_json(rfname):
    with codecs.open(rfname, 'r', encoding='utf-8') as rf:
        return json.load(rf)


def load_dataset(mode):
    """
    Load dataset by mode.

    Args:
        mode.
    Returns:
        names(list):[guanhua_du, ...]
        pubs(dict): {author:[...], title: xxx, ...}
    """
    if mode == "train":
        data_path = join(args.save_path, "src", "train", "train_author.json")
    elif mode == "valid":
        data_path = join(args.save_path, "src", "sna-valid", "sna_valid_raw.json")
    elif mode == "test":
        data_path = join(args.save_path, "src", "sna-test", "sna_test_raw.json")

    pubs = load_json(data_path)
    names = []
    for name in pubs:
        names.append(name)
    
    return names, pubs


def load_graph(name, th_a=args.coa_th, th_o=args.coo_th, th_v=args.cov_th, th_c=args.coc_th,
               th_i=args.coi_th, cite_out_weight=args.cite_out_weight, cite_in_weight=args.cite_in_weight):    
    """
    Args:
        name(str): author
        th_a(int): threshold of coA
        th_o(float): threshold of coO
        th_v(int): threshold of coV
        th_c(int): threshold for citation OUT edges     
        th_i(int): threshold for citation IN edges   
        cite_out_weight(float): weight multiplier for outgoing citations
        cite_in_weight(float): weight multiplier for incoming citations    
    Returns:
        label(list): true label
        ft_tensor(tensor): node feature
        data(Pyg Graph Data): graph
        
        Returns (None, None, None) if graph files are missing
    """
    data_path = join(args.save_path, 'graph')
    datapath = join(data_path, args.mode, name)

    # Check if graph files exist
    feats_path = join(datapath, 'feats_p.npy')
    adj_path = join(datapath, 'adj_attr.txt')
    
    if not os.path.exists(feats_path) or not os.path.exists(adj_path):
        print(f"⚠️  WARNING: Missing graph files for {name}")
        print(f"    feats_p.npy exists: {os.path.exists(feats_path)}")
        print(f"    adj_attr.txt exists: {os.path.exists(adj_path)}")
        return None, None, None
    # ======================================================

    # Load label
    if args.mode == "train":
        p_label = np.load(join(datapath, 'p_label.npy'), allow_pickle=True)
        p_label_list = []
        for pid in p_label.item():
            p_label_list.append(p_label.item()[pid])
        label = torch.LongTensor(p_label_list)
    else:
        label = []
    
    # Load node feature
    feats = np.load(join(datapath, 'feats_p.npy'), allow_pickle=True)
    ft_list = []
    for idx in feats.item():
        ft_list.append(feats.item()[idx])
    ft_tensor = torch.stack(ft_list)

    # Load edge
    temp = set()
    with open(join(datapath, 'adj_attr.txt'), 'r', encoding='utf-8') as f:
        for line in f:
            temp.add(line)

    srcs, dsts, value, attr = [], [], [], []
    for line in temp:
        toks = line.strip().split("\t")
        
        if len(toks) == 11:
            src, dst = int(toks[0]), int(toks[1])
            val_a = int(toks[2])
            val_o = int(toks[3])
            attr_o = float(toks[4])
            val_v = int(toks[5])
            attr_v = float(toks[6])
            val_cite_out = int(toks[7])      
            attr_cite_out = float(toks[8])   
            val_cite_in = int(toks[9])       
            attr_cite_in = float(toks[10])   
        # ===========================================================
        else:
            #print('read adj_attr ERROR!\n')
            continue

        if args.rel_on == 'a':
            if val_a > th_a:
                srcs.append(src)
                dsts.append(dst)
                value.append(val_a)
                attr.append(val_a)
        elif args.rel_on == 'o':
            if val_o > th_o:
                srcs.append(src)
                dsts.append(dst)
                value.append(val_o)
                attr.append(val_o)
        elif args.rel_on == 'v':
            if val_v > th_v:
                srcs.append(src)
                dsts.append(dst)
                value.append(val_v)
                attr.append(val_v)  
        elif args.rel_on == 'aov':
            prob_v = random.random()
            if (prob_v >= args.prob_v):
                val_v = val_v
            else:
                val_v = 0
            
            if attr_o >= args.coo_th:
                val_o = val_o
            else:
                val_o = 0

            has_relation = (
                (val_a > th_a) or 
                (val_o > th_o) or 
                (val_v > th_v) or 
                (val_cite_out > th_c) or  
                (val_cite_in > th_i)      
            )

            if has_relation:
                srcs.append(src)
                dsts.append(dst)
                
                total_weight = (
                    val_a * 2.0 + 
                    val_o * 1.0 + 
                    val_v * 1.0 + 
                    val_cite_out * cite_out_weight +
                    val_cite_in * cite_in_weight
                )
                value.append(total_weight)
                
                # Attributi multi-dimensionali (5D)
                attr.append([
                    float(val_a), 
                    float(attr_o), 
                    float(attr_v),
                    float(attr_cite_out),  
                    float(attr_cite_in)    
                ])
            # ======================================================
        else:
            print('wrong relation set\n')
            break

    temp.clear()

    # Build graph
    edge_index = torch.cat([torch.tensor(srcs).unsqueeze(0), torch.tensor(dsts).unsqueeze(0)], dim=0)
    edge_index = edge_index.long() 
    edge_attr = torch.tensor(attr, dtype=torch.float32)
    edge_weight = torch.tensor(value, dtype=torch.float32)
    data = Data(edge_index=edge_index, edge_attr=edge_attr, edge_weight=edge_weight)

    return label, ft_tensor, data

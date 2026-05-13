import os
import json
import codecs
import torch
import random
import pickle
import numpy as np
from os.path import join
from torch_geometric.data import Data


def load_data(rfname):
    with open(rfname, 'rb') as rf:
        return pickle.load(rf)

def dump_data(obj, wfname):
    with open(wfname, 'wb') as wf:
        pickle.dump(obj, wf)
       
def load_json(rfname):
    with codecs.open(rfname, 'r', encoding='utf-8') as rf:
        return json.load(rf)

def load_dataset(args, mode):
    if mode == "train":
        data_path = join(args.save_path, "src", "train", "train_author.json")
    elif mode == "valid":
        data_path = join(args.save_path, "src", "sna-valid", "sna_valid_raw.json")
    elif mode == "test":
        data_path = join(args.save_path, "src", "sna-test", "sna_test_raw.json")

    pubs = load_json(data_path)
    names = list(pubs.keys())
    return names, pubs

def load_graph(args, name):    
    """
    Load graph with proper threshold filtering.
    Tutti i parametri vengono letti da args, passato esplicitamente dal chiamante.
    """
    # Leggi soglie e pesi da args
    th_a            = args.coa_th
    th_o            = args.coo_th
    th_v            = args.cov_th
    th_c            = args.coc_th
    th_i            = args.coi_th
    cite_out_weight = args.cite_out_weight
    cite_in_weight  = args.cite_in_weight

    datapath = join(args.save_path, 'graph', args.mode, name)

    feats_path = join(datapath, 'feats_p.npy')
    adj_path   = join(datapath, 'adj_attr.txt')
    
    if not os.path.exists(feats_path) or not os.path.exists(adj_path):
        print(f"⚠️  WARNING: Missing graph files for {name}")
        return None, None, None

    # Load label
    if args.mode == "train":
        p_label = np.load(join(datapath, 'p_label.npy'), allow_pickle=True)
        label = torch.LongTensor([p_label.item()[pid] for pid in p_label.item()])
    else:
        label = []
    
    # Load node features
    feats = np.load(feats_path, allow_pickle=True)
    ft_tensor = torch.stack([feats.item()[idx] for idx in feats.item()])

    # Load edges
    temp = set()
    with open(adj_path, 'r', encoding='utf-8') as f:
        for line in f:
            temp.add(line)

    srcs, dsts, value, attr = [], [], [], []
    for line in temp:
        toks = line.strip().split("\t")
        
        if len(toks) != 11:
            continue

        src, dst        = int(toks[0]),   int(toks[1])
        val_a           = int(toks[2])
        val_o           = int(toks[3]);   attr_o       = float(toks[4])
        val_v           = int(toks[5]);   attr_v       = float(toks[6])
        val_cite_out    = int(toks[7]);   attr_cite_out = float(toks[8])
        val_cite_in     = int(toks[9]);   attr_cite_in  = float(toks[10])

        if args.rel_on == 'a':
            if val_a > th_a:
                srcs.append(src); dsts.append(dst)
                value.append(val_a); attr.append(val_a)
                
        elif args.rel_on == 'o':
            if val_o > th_o:
                srcs.append(src); dsts.append(dst)
                value.append(val_o); attr.append(val_o)
                
        elif args.rel_on == 'v':
            if val_v > th_v:
                srcs.append(src); dsts.append(dst)
                value.append(val_v); attr.append(val_v)
                
        elif args.rel_on == 'aov':
            has_relation = (
                (val_a > th_a) or 
                (val_o > th_o and attr_o >= args.coo_th) or
                (val_v > th_v)
            )
            if args.use_citations:
                has_relation = has_relation or (val_cite_out > th_c) or (val_cite_in > th_i)

            if has_relation:
                weight_a = (val_a * 2.0) if val_a > th_a else 0.0
                weight_o = (val_o * 1.0) if (val_o > th_o and attr_o >= args.coo_th) else 0.0
                weight_v = (val_v * 1.0) if val_v > th_v else 0.0
                
                weight_cite_out = 0.0
                weight_cite_in  = 0.0
                if args.use_citations:
                    if val_cite_out > th_c: weight_cite_out = val_cite_out * cite_out_weight
                    if val_cite_in  > th_i: weight_cite_in  = val_cite_in  * cite_in_weight
                
                total_weight = weight_a + weight_o + weight_v + weight_cite_out + weight_cite_in
                
                if total_weight > 0:
                    srcs.append(src); dsts.append(dst)
                    value.append(total_weight)
                    attr.append([
                        float(val_a), 
                        float(attr_o), 
                        float(attr_v),
                        float(attr_cite_out) if args.use_citations else 0.0,  
                        float(attr_cite_in)  if args.use_citations else 0.0,
                    ])
        else:
            print('wrong relation set\n')
            break

    temp.clear()

    if len(srcs) > 0:
        edge_index  = torch.cat([torch.tensor(srcs).unsqueeze(0), torch.tensor(dsts).unsqueeze(0)], dim=0).long()
        edge_attr   = torch.tensor(attr,  dtype=torch.float32)
        edge_weight = torch.tensor(value, dtype=torch.float32)
        data = Data(edge_index=edge_index, edge_attr=edge_attr, edge_weight=edge_weight)
    else:
        data = Data(
            edge_index  = torch.empty((2, 0), dtype=torch.long),
            edge_attr   = torch.empty((0, 5), dtype=torch.float32),
            edge_weight = torch.empty(0,       dtype=torch.float32),
        )

    return label, ft_tensor, data
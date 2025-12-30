import torch
import os
import numpy as np
import scipy.sparse as sp
import pickle
from gensim.models import word2vec
from os.path import join
from tqdm import tqdm
from bond.dataset.load_data import load_json
from bond.dataset.save_results import check_mkdir
from bond.params import set_params

args = set_params()

def gen_relations(name, mode, target):
    dirpath = join(args.save_path, 'relations', mode, name)
    temp = set()
    paper_info = dict()

    if target == 'author': filename = "paper_author.txt"
    elif target == 'org': filename = "paper_org.txt"
    elif target == 'venue': filename = "paper_venue.txt"
    elif target == 'cite_out': filename = "paper_cite_out.txt"
    elif target == 'cite_in': filename = "paper_cite_in.txt"
    else: return {}
    
    file_path = join(dirpath, filename)
    if not os.path.exists(file_path): return {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f: temp.add(line)

    for line in temp:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            p, a = toks[0], toks[1]     
            if p not in paper_info: paper_info[p] = []
            paper_info[p].append(a)
    return paper_info

def save_label_pubs(mode, name, raw_pubs, save_path):
    if mode == "train":
        label_dict = {}
        pubs=[]
        ilabel = 0
        for aid in raw_pubs[name]:
            pubs.extend(raw_pubs[name][aid])
            for pid in raw_pubs[name][aid]:
                label_dict[pid] = ilabel
            ilabel += 1
        np.save(join(save_path, "p_label.npy"), label_dict)
    else:
        pubs = [pid for pid in raw_pubs[name]]
    return pubs

def save_emb(mode, name, pubs, save_path):
    """
    Crea feats_p.npy con embeddings W2V per ogni paper.
    Include TUTTI i paper, anche quelli senza embedding valido (usa zero vector).
    """
    import torch  # Import necessario
    
    # Percorso agli embeddings W2V pre-generati
    emb_path = join(args.save_path, 'paper_emb', mode, name, 'ptext_emb.pkl')
    tcp_path = join(args.save_path, 'paper_emb', mode, name, 'tcp.pkl')
    
    # Dimensione embedding
    ft_dim = 256
    
    # Carica embeddings se esistono
    ptext_emb = {}
    tcp = set()
    
    if os.path.exists(emb_path):
        with open(emb_path, 'rb') as f:
            ptext_emb = pickle.load(f)
        
        if os.path.exists(tcp_path):
            with open(tcp_path, 'rb') as f:
                tcp = pickle.load(f)
    else:
        print(f"  ⚠️  No embeddings for {name}, using zero vectors")
    
    # ===== CREA feats_p.npy CON TUTTI I PAPER (COME TENSOR) =====
    feats_dict = {}
    missing_count = 0
    
    for idx, pid in enumerate(pubs):
        if pid in ptext_emb:
            # Converti numpy array → torch tensor
            feats_dict[idx] = torch.tensor(ptext_emb[pid], dtype=torch.float32)
        else:
            # Zero vector come tensor
            feats_dict[idx] = torch.zeros(ft_dim, dtype=torch.float32)
            missing_count += 1
    
    # Salva come numpy dict
    np.save(join(save_path, 'feats_p.npy'), feats_dict)
    
    if missing_count > 0:
        print(f"  ⚠️  {name}: {missing_count}/{len(pubs)} papers with zero vectors")
    # ============================================================

def save_graph(name, pubs, save_path, mode):
    paper_dict = {pid: idx for idx, pid in enumerate(pubs)}
    cp_a, cp_o = set(), set()

    # ===== AGGIUNGI QUESTO ALL'INIZIO =====
    # Salva pids.txt (mapping indici → paper IDs)
    with open(join(save_path, 'pids.txt'), 'w', encoding='utf-8') as f:
        for pid in pubs:
            f.write(f'{pid}\n')
    # ======================================

    # Carica relazioni
    rels = {
        'auth': gen_relations(name, mode, 'author'),
        'org': gen_relations(name, mode, 'org'),
        'ven': gen_relations(name, mode, 'venue'),
        'cout': gen_relations(name, mode, 'cite_out'),
        'cin': gen_relations(name, mode, 'cite_in')
    }

    # Identifica outlier (paper senza relazioni base)
    for pid in paper_dict:
        if pid not in rels['auth']: cp_a.add(paper_dict[pid])
        if pid not in rels['org']: cp_o.add(paper_dict[pid])
    cp = cp_a & cp_o 

    with open(join(save_path, 'adj_attr.txt'), 'w') as f:  
        for p1 in paper_dict:
            p1_idx = paper_dict[p1]
            for p2 in paper_dict:
                p2_idx = paper_dict[p2] 
                if p1 == p2: continue

                # Helper per calcolare count e jaccard
                def calc_rel(key):
                    if p1 in rels[key] and p2 in rels[key]:
                        s1, s2 = set(rels[key][p1]), set(rels[key][p2])
                        cnt = len(s1 & s2)
                        jac = cnt / len(s1 | s2) if (s1 | s2) else 0.0
                        return cnt, jac
                    return 0, 0.0

                # Calcola metriche
                co_a, _ = calc_rel('auth')
                co_o, jac_o = calc_rel('org')
                co_v, jac_v = calc_rel('ven')
                co_cout, jac_cout = calc_rel('cout')
                co_cin, jac_cin = calc_rel('cin')

                # Scrivi se c'è almeno una relazione
                if (co_a + co_o + co_v + co_cout + co_cin) > 0:
                    f.write(f'{p1_idx}\t{p2_idx}\t'
                            f'{co_a}\t'
                            f'{co_o}\t{jac_o:.4f}\t'
                            f'{co_v}\t{jac_v:.4f}\t'
                            f'{co_cout}\t{jac_cout:.4f}\t'
                            f'{co_cin}\t{jac_cin:.4f}\n')
                
    with open(join(save_path, 'rel_cp.txt'), 'w') as out_f:
        for i in cp: out_f.write(f'{i}\n')

def build_graph():
    for mode in ["train", "valid", "test"]:
        print(f"Building graphs for {mode}...")
        data_base = join(args.save_path, "src")
        if mode == "train":
            raw_pubs = load_json(join(data_base, "train", "train_author.json"))
        elif mode == "valid":
            raw_pubs = load_json(join(data_base, "sna-valid", "sna_valid_raw.json"))
        elif mode == "test":
            raw_pubs = load_json(join(data_base, "sna-test", "sna_test_raw.json"))
        
        for name in tqdm(raw_pubs):
            save_path = join(args.save_path, 'graph', mode, name)
            check_mkdir(save_path)
            
            # Step 1: Estrai lista paper
            pubs = save_label_pubs(mode, name, raw_pubs, save_path)
            
            # Step 2: Crea feats_p.npy con embeddings W2V
            save_emb(mode, name, pubs, save_path)
            
            # Step 3: Costruisci grafo (adj_attr.txt, pids.txt, rel_cp.txt)
            save_graph(name, pubs, save_path, mode) 


if __name__ == "__main__":
    build_graph()
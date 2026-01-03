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

# ============================================================
# [AGGIUNTO] FUNZIONE DI NORMALIZZAZIONE ID
# ============================================================
def clean_id(identifier):
    """
    Rimuove prefissi comuni (doi:, omid:, ecc.) e pulisce la stringa
    per permettere il confronto tra metadati e citazioni.
    """
    if not identifier:
        return ""
    identifier = str(identifier).lower().strip()
    prefixes = ["doi:", "omid:", "openalex:", "pmid:", "https://doi.org/", "http://doi.org/"]
    for p in prefixes:
        if identifier.startswith(p):
            identifier = identifier[len(p):]
    return identifier

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
    emb_path = join(args.save_path, 'paper_emb', mode, name, 'ptext_emb.pkl')
    tcp_path = join(args.save_path, 'paper_emb', mode, name, 'tcp.pkl')
    ft_dim = 256
    ptext_emb = {}
    if os.path.exists(emb_path):
        with open(emb_path, 'rb') as f:
            ptext_emb = pickle.load(f)
    
    feats_dict = {}
    for idx, pid in enumerate(pubs):
        if pid in ptext_emb:
            feats_dict[idx] = torch.tensor(ptext_emb[pid], dtype=torch.float32)
        else:
            feats_dict[idx] = torch.zeros(ft_dim, dtype=torch.float32)
    
    np.save(join(save_path, 'feats_p.npy'), feats_dict)

# ============================================================
# [MODIFICATO] COSTRUZIONE RESOLVER CON PULIZIA
# ============================================================
def build_id_resolver(pubs_dict):
    """
    Crea una mappa che associa ogni DOI o OMID pulito all'ID interno (pid).
    """
    resolver = {}
    for pid, pub in pubs_dict.items():
        # Mappa il DOI pulito
        if pub.get('doi'):
            resolver[clean_id(pub['doi'])] = pid
        # Mappa l'OMID pulito
        if pub.get('omid'):
            resolver[clean_id(pub['omid'])] = pid
    return resolver

# ============================================================
# [MODIFICATO] SALVATAGGIO GRAFO CON LOGICA DI MATCH ROBUSTA
# ============================================================
def save_graph(name, pubs, save_path, mode):
    paper_dict = {pid: idx for idx, pid in enumerate(pubs)}
    cp_a, cp_o = set(), set()

    # Carichiamo i metadati JSON dell'autore per il resolver
    pubs_json_path = join(args.save_path, 'names_pub', mode, name + '.json')
    pubs_dict = load_json(pubs_json_path)
    id_resolver = build_id_resolver(pubs_dict)

    # DEBUG opzionale: print(f"Resolver per {name}: {len(id_resolver)} chiavi")

    rels = {
        'auth': gen_relations(name, mode, 'author'),
        'org': gen_relations(name, mode, 'org'),
        'ven': gen_relations(name, mode, 'venue'),
        'cout': gen_relations(name, mode, 'cite_out'),
        'cin': gen_relations(name, mode, 'cite_in')
    }

    for pid in paper_dict:
        if pid not in rels['auth']: cp_a.add(paper_dict[pid])
        if pid not in rels['org']: cp_o.add(paper_dict[pid])
    cp = cp_a & cp_o 

    with open(join(save_path, 'adj_attr.txt'), 'w') as f:  
        for p1 in paper_dict:
            p1_idx = paper_dict[p1]
            
            # --- TRADUZIONE RIFERIMENTI DI P1 ---
            # Convertiamo i DOI/OMID citati da p1 in ID interni presenti nel set
            p1_internal_references = set()
            if p1 in rels['cout']:
                for raw_id in rels['cout'][p1]:
                    # Puliamo l'ID della citazione (rimuove "doi:", ecc.)
                    target_id = clean_id(raw_id)
                    internal_id = id_resolver.get(target_id)
                    if internal_id:
                        p1_internal_references.add(internal_id)

            for p2 in paper_dict:
                p2_idx = paper_dict[p2] 
                if p1 == p2: continue

                def calc_rel(key):
                    if p1 in rels[key] and p2 in rels[key]:
                        s1, s2 = set(rels[key][p1]), set(rels[key][p2])
                        cnt = len(s1 & s2)
                        jac = cnt / len(s1 | s2) if (s1 | s2) else 0.0
                        return cnt, jac
                    return 0, 0.0

                co_a, _ = calc_rel('auth')
                co_o, jac_o = calc_rel('org')
                co_v, jac_v = calc_rel('ven')

                # --- CITAZIONE DIRETTA ---
                # Se l'ID di p2 è tra i riferimenti risolti di p1
                direct_cite = 1 if p2 in p1_internal_references else 0
                
                # --- CO-CITAZIONE / BIBLIOGRAPHIC COUPLING ---
                co_cout, jac_cout = calc_rel('cout')
                co_cin, jac_cin = calc_rel('cin')

                # Valori finali per colonne 8-9 (Cite Out)
                val_cite_out = co_cout + direct_cite
                attr_cite_out = max(jac_cout, 1.0 if direct_cite else 0.0)

                # Scrittura 11 colonne
                if (co_a + co_o + co_v + val_cite_out + co_cin) > 0:
                    f.write(f'{p1_idx}\t{p2_idx}\t'
                            f'{co_a}\t'
                            f'{co_o}\t{jac_o:.4f}\t'
                            f'{co_v}\t{jac_v:.4f}\t'
                            f'{val_cite_out}\t{attr_cite_out:.4f}\t'
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
            pubs = save_label_pubs(mode, name, raw_pubs, save_path)
            save_emb(mode, name, pubs, save_path)
            save_graph(name, pubs, save_path, mode) 

if __name__ == "__main__":
    build_graph()
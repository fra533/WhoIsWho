import torch
import os
import numpy as np
import scipy.sparse as sp
from gensim.models import word2vec
from os.path import join
from tqdm import tqdm
from dataset.load_data import load_data, load_json
from dataset.save_results import check_mkdir

from params import set_params

args = set_params()


def gen_relations(name, mode, target):
    dirpath = join(args.save_path, 'relations', mode, name)

    temp = set()
    paper_info = dict()
    info_paper = dict()

    # Mapping nomi file
    if target == 'author':
        filename = "paper_author.txt"
    elif target == 'org':
        filename = "paper_org.txt"
    elif target == 'venue':
        filename = "paper_venue.txt"
    elif target == 'cite_out':  # NUOVO
        filename = "paper_cite_out.txt"
    elif target == 'cite_in':   # NUOVO
        filename = "paper_cite_in.txt"
    else:
        raise ValueError(f"Unknown target: {target}")
    
    file_path = join(dirpath, filename)
    
    # Se file non esiste (es. nessuna citazione), ritorna dict vuoto
    if not os.path.exists(file_path):
        return {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            temp.add(line)

    for line in temp:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            p, a = toks[0], toks[1]     
            if p not in paper_info:
                paper_info[p] = []
            paper_info[p].append(a)

    temp.clear()
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

        file_path = join(save_path, "p_label.npy")
        np.save(file_path, label_dict)
    else:
        pubs = []
        for pid in raw_pubs[name]:
            pubs.append(pid)

    return pubs


def save_graph(name, pubs, save_path, mode):

    # init node mapping & edge mapping
    paper_dict = {pid: idx for idx, pid in enumerate(pubs)}
    cp_a, cp_o = set(), set()

    paper_rel_ath = gen_relations(name, mode, 'author')
    paper_rel_org = gen_relations(name, mode, 'org')
    paper_rel_ven = gen_relations(name, mode, 'venue')
    
    # ===== NUOVO: Carica relazioni citazioni =====
    paper_rel_cite_out = gen_relations(name, mode, 'cite_out')
    paper_rel_cite_in = gen_relations(name, mode, 'cite_in')
    # ==============================================
    
    for pid in paper_dict:
        if pid not in paper_rel_ath:
            cp_a.add(paper_dict[pid])
    for pid in paper_dict:
        if pid not in paper_rel_org:
            cp_o.add(paper_dict[pid])

    # mark paper w/o coauthor and coorg as outlier
    cp = cp_a & cp_o 

    with open(join(save_path, 'adj_attr.txt'), 'w') as f:  
        
        for p1 in paper_dict:
            p1_idx = paper_dict[p1]
            for p2 in paper_dict:
                p2_idx = paper_dict[p2] 
                if p1 != p2:
                    co_aths, co_orgs, co_vens = 0, 0, 0
                    co_cite_out, co_cite_in = 0, 0  # NUOVO
                    org_attr, org_attr_jaccard, org_jaccard2 = 0, 0, 0
                    ven_attr, ven_attr_jaccard, venue_jaccard2 = 0, 0, 0
                    cite_out_attr, cite_in_attr = 0, 0  # NUOVO
                    org_idf_sum, org_idf_sum1, org_idf_sum2 = 0, 0, 0
                    ven_idf_sum, ven_idf_sum1, ven_idf_sum2 = 0, 0, 0
                    co_org_idf, co_ven_idf = 0, 0
                    co_org_idf_2, co_ven_idf_2 = 0, 0
                    
                    # Co-authors
                    if p1 in paper_rel_ath:
                        for k in paper_rel_ath[p1]:
                            if p2 in paper_rel_ath:
                                if k in paper_rel_ath[p2]:
                                    co_aths += 1
                    
                    # Co-orgs
                    if p1 in paper_rel_org:
                        for k in paper_rel_org[p1]:
                            if p2 in paper_rel_org:
                                if k in paper_rel_org[p2]:
                                    co_orgs += 1

                    # Co-venues
                    if p1 in paper_rel_ven:
                        for k in paper_rel_ven[p1]:
                            if p2 in paper_rel_ven:
                                if k in paper_rel_ven[p2]:
                                    co_vens += 1

                    # ===== NUOVO: Co-citazioni =====
                    # Co-citazioni outgoing (citano gli stessi paper)
                    if p1 in paper_rel_cite_out and p2 in paper_rel_cite_out:
                        common_refs = set(paper_rel_cite_out[p1]) & set(paper_rel_cite_out[p2])
                        co_cite_out = len(common_refs)
                    
                    # Co-citazioni incoming (sono citati dagli stessi paper)
                    if p1 in paper_rel_cite_in and p2 in paper_rel_cite_in:
                        common_citing = set(paper_rel_cite_in[p1]) & set(paper_rel_cite_in[p2])
                        co_cite_in = len(common_citing)
                    # ===================================

                    # Calcola attributi org
                    if co_orgs > 0:
                        all_words_p1 = len(paper_rel_org[p1])
                        all_words_p2 = len(paper_rel_org[p2])
                        org_attr = co_orgs / max(all_words_p1, all_words_p2)
                        org_attr_jaccard = co_orgs / (all_words_p1 + all_words_p2 - co_orgs)

                    # Calcola attributi venue
                    if co_vens > 0:
                        all_words_p1 = len(paper_rel_ven[p1])
                        all_words_p2 = len(paper_rel_ven[p2])
                        ven_attr = co_vens / max(all_words_p1, all_words_p2)
                        ven_attr_jaccard = co_vens / (all_words_p1 + all_words_p2 - co_vens)

                    # ===== NUOVO: Calcola attributi citazioni =====
                    # Similarità bibliografica (citano paper simili)
                    if co_cite_out > 0:
                        all_refs_p1 = len(paper_rel_cite_out.get(p1, []))
                        all_refs_p2 = len(paper_rel_cite_out.get(p2, []))
                        if all_refs_p1 + all_refs_p2 - co_cite_out > 0:
                            cite_out_attr = co_cite_out / (all_refs_p1 + all_refs_p2 - co_cite_out)
                    
                    # Accoppiamento bibliografico (citati insieme)
                    if co_cite_in > 0:
                        all_citing_p1 = len(paper_rel_cite_in.get(p1, []))
                        all_citing_p2 = len(paper_rel_cite_in.get(p2, []))
                        if all_citing_p1 + all_citing_p2 - co_cite_in > 0:
                            cite_in_attr = co_cite_in / (all_citing_p1 + all_citing_p2 - co_cite_in)
                    # ==============================================

                    # Scrivi edge con TUTTI gli attributi (incluse citazioni)
                    if (co_aths + co_orgs + co_cite_out + co_cite_in) > 0:  # MODIFICATO
                        f.write(f'{p1_idx}\t{p2_idx}\t{co_aths}\t'
                                f'{co_orgs}\t{org_attr_jaccard}\t'
                                f'{co_vens}\t{ven_attr_jaccard}\t'
                                f'{co_cite_out}\t{cite_out_attr}\t'  # NUOVO
                                f'{co_cite_in}\t{cite_in_attr}\n')   # NUOVO
    
    f.close()
                
    with open(join(save_path, 'rel_cp.txt'), 'w') as out_f:
        for i in cp:
            out_f.write(f'{i}\n')
    out_f.close()


def save_emb(mode, name, pubs, save_path):
    # build mapping of paper & index-id
    mapping = dict()
    for idx, pid in enumerate(pubs):
        mapping[idx] = pid

    # load paper embedding
    ptext_emb = load_data(join(args.save_path, 'paper_emb', mode, name, 'ptext_emb.pkl'))

    # init node feature matrix(n * dim_size)
    ft = dict()
    for pidx_1 in mapping:
        pid_1 = mapping[pidx_1]
        ft[pidx_1] = torch.from_numpy(ptext_emb[pid_1])

    feats_file_path = join(save_path, 'feats_p.npy')
    np.save(feats_file_path, ft)


def build_graph():
    
    for mode in ["train", "valid", "test"]:
        print("preprocess dataset: ", mode)
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
            save_graph(name, pubs, save_path, mode)
            save_emb(mode, name, pubs, save_path)


if __name__ == "__main__":
    build_graph()



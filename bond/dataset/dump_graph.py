import torch
import os
import numpy as np
import pickle
from os.path import join
from tqdm import tqdm
from bond.dataset.load_data import load_json
from bond.dataset.save_results import check_mkdir


def clean_id(identifier):
    """Normalizza un DOI: lowercase e strip."""
    if not identifier:
        return ""
    return str(identifier).lower().strip()


def gen_relations(args, name, mode, target):
    dirpath = join(args.save_path, 'relations', mode, name)
    temp = set()
    paper_info = dict()

    filenames = {
        'author':   "paper_author.txt",
        'org':      "paper_org.txt",
        'venue':    "paper_venue.txt",
        'cite_out': "paper_cite_out.txt",
        'cite_in':  "paper_cite_in.txt",
    }
    if target not in filenames:
        return {}

    file_path = join(dirpath, filenames[target])
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

    return paper_info


def save_label_pubs(mode, name, raw_pubs, save_path):
    if mode == "train":
        label_dict = {}
        pubs = []
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


def save_emb(args, mode, name, pubs, save_path):
    emb_folder = getattr(args, 'emb_dir_name', 'paper_emb')
    emb_path = join(args.save_path, emb_folder, mode, name, 'ptext_emb.pkl')

    ft_dim = 768 if args.emb_type.lower() in ['specter', 'sbert'] else 256

    ptext_emb = {}
    if os.path.exists(emb_path):
        with open(emb_path, 'rb') as f:
            ptext_emb = pickle.load(f)

    feats_dict = {}
    for idx, pid in enumerate(pubs):
        if pid in ptext_emb:
            vec = torch.tensor(ptext_emb[pid], dtype=torch.float32)
            if vec.shape[0] != ft_dim:
                vec = torch.zeros(ft_dim, dtype=torch.float32)
            feats_dict[idx] = vec
        else:
            feats_dict[idx] = torch.zeros(ft_dim, dtype=torch.float32)

    np.save(join(save_path, 'feats_p.npy'), feats_dict)


def build_id_resolver(pubs_dict):
    """Crea una mappa DOI (lowercase) -> pid."""
    resolver = {}
    for pid, pub in pubs_dict.items():
        if pub.get('doi'):
            doi_clean = pub['doi'].lower().strip()
            resolver[doi_clean] = pid
    return resolver


def save_graph(args, name, pubs, save_path, mode):
    paper_dict = {pid: idx for idx, pid in enumerate(pubs)}
    cp_a, cp_o = set(), set()

    pubs_json_path = join(args.save_path, 'names_pub', mode, name + '.json')
    pubs_dict  = load_json(pubs_json_path)
    id_resolver = build_id_resolver(pubs_dict)

    rels = {
        'auth': gen_relations(args, name, mode, 'author'),
        'org':  gen_relations(args, name, mode, 'org'),
        'ven':  gen_relations(args, name, mode, 'venue'),
        'cout': gen_relations(args, name, mode, 'cite_out'),
        'cin':  gen_relations(args, name, mode, 'cite_in'),
    }

    for pid in paper_dict:
        if pid not in rels['auth']: cp_a.add(paper_dict[pid])
        if pid not in rels['org']:  cp_o.add(paper_dict[pid])
    cp = cp_a & cp_o

    with open(join(save_path, 'adj_attr.txt'), 'w') as f:
        for p1 in paper_dict:
            p1_idx = paper_dict[p1]

            p1_internal_references = set()
            if p1 in rels['cout']:
                for doi in rels['cout'][p1]:
                    internal_id = id_resolver.get(doi)
                    if internal_id:
                        p1_internal_references.add(internal_id)

            for p2 in paper_dict:
                p2_idx = paper_dict[p2]
                if p1 == p2:
                    continue

                def calc_rel(key):
                    if p1 in rels[key] and p2 in rels[key]:
                        s1, s2 = set(rels[key][p1]), set(rels[key][p2])
                        cnt = len(s1 & s2)
                        jac = cnt / len(s1 | s2) if (s1 | s2) else 0.0
                        return cnt, jac
                    return 0, 0.0

                co_a,    jac_a    = calc_rel('auth')
                co_o,    jac_o    = calc_rel('org')
                co_v,    jac_v    = calc_rel('ven')
                co_cout, jac_cout = calc_rel('cout')
                co_cin,  jac_cin  = calc_rel('cin')

                direct_cite  = 1 if p2 in p1_internal_references else 0
                val_cite_out = co_cout + direct_cite
                attr_cite_out = max(jac_cout, 1.0 if direct_cite else 0.0)

                is_safe_coauthor  = (co_a    >= 3)
                is_direct_citation = (direct_cite > 0)
                is_strong_coupling = (co_cout >= 3)

                if is_safe_coauthor or is_direct_citation or is_strong_coupling:
                    f.write(
                        f'{p1_idx}\t{p2_idx}\t'
                        f'{co_a}\t'
                        f'{co_o}\t{jac_o:.4f}\t'
                        f'{co_v}\t{jac_v:.4f}\t'
                        f'{val_cite_out}\t{attr_cite_out:.4f}\t'
                        f'{co_cin}\t{jac_cin:.4f}\n'
                    )

    with open(join(save_path, 'rel_cp.txt'), 'w') as out_f:
        for i in cp:
            out_f.write(f'{i}\n')


def build_graph(args):
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
            save_emb(args, mode, name, pubs, save_path)
            save_graph(args, name, pubs, save_path, mode)


if __name__ == "__main__":
    from bond.params import set_params
    args = build_graph(set_params())
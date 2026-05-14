import torch
import random
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances

from torch_geometric.nn import GAE
from loadmodel.att_gnn import ATTGNN
from dataset.load_data import load_dataset, load_graph
from dataset.save_results import save_results
from os.path import join, dirname
import os
import json
from .generate_pair import generate_pair


class BONDTrainer:
    def __init__(self, args, no_gnn=False):
        """
        Args:
            args:    oggetto Namespace restituito da set_params(), passato da PipelineA.
            no_gnn:  se True bypassa il training GNN e usa gli embedding originali.
                     Passato esplicitamente da PipelineA — non più letto da env var.
        """
        self.args   = args
        self.no_gnn = no_gnn

        # Seed — impostato qui, una sola volta, con i valori di args corretti
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        self.device = torch.device(
            ("cuda:" + str(args.gpu))
            if torch.cuda.is_available() and args.cuda
            else "cpu"
        )

        if self.no_gnn:
            print("🚀 MODALITÀ NO_GNN ATTIVA: La GNN verrà bypassata.")
        else:
            print("🧠 MODALITÀ GNN ATTIVA: Procedo con il training.")

    def onehot_encoder(self, label_list):
        if isinstance(label_list, list):
            labels_arr = np.array(label_list)
        elif isinstance(label_list, torch.Tensor):
            labels_arr = label_list.detach().cpu().numpy()
        else:
            labels_arr = np.array(label_list)

        num_classes = max(0, int(max(labels_arr)) + 1)
        onehot_mat = np.zeros((len(labels_arr), num_classes))
        for i in range(len(labels_arr)):
            if labels_arr[i] != -1:
                onehot_mat[i, labels_arr[i]] = 1
        return onehot_mat

    def matx2list(self, adj):
        """
        Transform similarity matrix to cluster labels.
        """
        if not isinstance(adj, np.ndarray):
            adj = adj.cpu().detach().numpy() if torch.is_tensor(adj) else np.array(adj)

        n_papers = adj.shape[0]
        labels   = [-1] * n_papers
        assigned = [False] * n_papers
        current_label = 0

        for i in range(n_papers):
            if assigned[i]:
                continue
            cluster_members = [j for j in range(n_papers) if adj[i][j] == 1]
            if cluster_members:
                for j in cluster_members:
                    labels[j]   = current_label
                    assigned[j] = True
                current_label += 1
            else:
                labels[i]   = -1
                assigned[i] = True

        return labels

    def post_match(self, pred, pubs, name, mode):
        """
        Post-match outliers.
        Args:
            pred(list): prediction e.g. [0, 0, -1, 1]
            pubs(list): paper-ids
            name(str):  author name
            mode(str):  train/valid/test
        Return:
            pred(list): after post-match e.g. [0, 0, 0, 1]
        """
        args = self.args

        outlier = {i for i, p in enumerate(pred) if p == -1}

        datapath = join(args.save_path, 'graph', mode, name)
        with open(join(datapath, 'rel_cp.txt'), 'r') as f:
            rel_outlier = [int(x) for x in f.read().split('\n')[:-1]]
        for i in rel_outlier:
            outlier.add(i)

        print(f"post matching {len(outlier)} outliers")
        paper_pair  = generate_pair(pubs, name, outlier, mode)
        paper_pair1 = paper_pair.copy()

        K = len(set(pred))

        for i in range(len(pred)):
            if i not in outlier:
                continue
            j = np.argmax(paper_pair[i])
            while j in outlier:
                paper_pair[i][j] = -1
                last_j = j
                j = np.argmax(paper_pair[i])
                if j == last_j:
                    break
            if paper_pair[i][j] >= 1.5:
                pred[i] = pred[j]
            else:
                pred[i] = K
                K += 1

        for ii, i in enumerate(outlier):
            for jj, j in enumerate(outlier):
                if jj <= ii:
                    continue
                if paper_pair1[i][j] >= 1.5:
                    pred[j] = pred[i]

        return pred

    def labels_to_clusters(self, pred, paper_ids):
        """
        Ogni outlier (-1) diventa un cluster singleton separato.
        """
        from collections import defaultdict
        cluster_dict  = defaultdict(list)
        outlier_count = 0

        for idx, label in enumerate(pred):
            l_val = label.item() if hasattr(label, 'item') else label
            if l_val == -1:
                cluster_dict[f"outlier_{outlier_count}"].append(paper_ids[idx])
                outlier_count += 1
            else:
                cluster_dict[l_val].append(paper_ids[idx])

        return list(cluster_dict.values())

    def fit(self, datatype):
        args   = self.args
        device = self.device

        names, pubs = load_dataset(args, datatype)

        if datatype in ['valid', 'test']:
            if datatype == 'valid':
                gt_file = join(args.save_path, 'src', 'sna-valid', 'sna_valid_ground_truth.json')
            else:
                gt_file = join(args.save_path, 'src', 'sna-test', 'sna_test_ground_truth.json')

            if os.path.exists(gt_file):
                with open(gt_file, 'r', encoding='utf-8') as f:
                    ground_truth = json.load(f)
                gt_names = set(ground_truth.keys())
                names = [n for n in names if n in gt_names]
                pubs  = {n: pubs[n] for n in names if n in pubs}
                print(f"\n{'='*70}")
                print(f"FILTERED TO GROUND TRUTH: {len(names)} authors")
                print(f"{'='*70}\n")
            else:
                print(f"⚠️  Ground truth file not found: {gt_file}")

        results              = {}
        total_authors        = len(names)
        processed_authors    = 0
        special_case_authors = 0
        special_case_info    = []

        for name in names:
            print("training:", name)
            results[name] = []

            # ==== Load data ====
            label, ft_list, data = load_graph(args, name)

            if label is None or ft_list is None or data is None:
                print(f"  ❌ Failed to load graph files")
                name_pubs = []
                if datatype == 'train':
                    for aid in pubs[name]:
                        name_pubs.extend(pubs[name][aid])
                else:
                    name_pubs = list(pubs[name])

                results[name] = [[pid] for pid in name_pubs] if name_pubs else []
                special_case_authors += 1
                special_case_info.append((name, 0, 0, "Load failed"))
                continue

            # Get paper IDs list
            name_pubs = []
            if datatype == 'train':
                for aid in pubs[name]:
                    name_pubs.extend(pubs[name][aid])
            else:
                name_pubs = list(pubs[name])

            n_nodes = ft_list.shape[0]
            n_edges = data.edge_index.shape[1]

            # ===== CHECK DI SICUREZZA =====
            if n_nodes != len(name_pubs):
                print(f"  ❌ CRITICAL ERROR: Graph({n_nodes}) != Papers({len(name_pubs)})")
                results[name] = [[pid] for pid in name_pubs]
                special_case_authors += 1
                special_case_info.append((name, n_nodes, len(name_pubs), "Build error"))
                continue

            print(f"  Graph: {n_nodes} nodes, {n_edges} edges")

            # ===== CASI SPECIALI =====

            if n_nodes == 1:
                print(f"  → Single paper, creating 1 cluster")
                results[name] = [[name_pubs[0]]]
                special_case_authors += 1
                special_case_info.append((name, n_nodes, n_edges, "Single paper"))
                continue

            if n_edges == 0:
                print(f"  → No edges, creating {n_nodes} singleton clusters")
                results[name] = [[pid] for pid in name_pubs]
                special_case_authors += 1
                special_case_info.append((name, n_nodes, n_edges, "No edges"))
                continue

            if n_nodes <= 4:
                print(f"  → Very small graph, using simplified clustering")
                try:
                    from sklearn.metrics.pairwise import euclidean_distances
                    distances = euclidean_distances(ft_list.cpu().numpy())
                    labels    = DBSCAN(eps=args.db_eps * 2, min_samples=1, metric='precomputed').fit_predict(distances)
                    results[name] = self.labels_to_clusters(labels.tolist(), name_pubs)
                    print(f"  → Created {len(results[name])} clusters")
                    special_case_authors += 1
                    special_case_info.append((name, n_nodes, n_edges, "Simplified"))
                    continue
                except Exception as e:
                    print(f"  → Simplified clustering failed: {e}, using singletons")
                    results[name] = [[pid] for pid in name_pubs]
                    special_case_authors += 1
                    special_case_info.append((name, n_nodes, n_edges, "Failed"))
                    continue

            if n_nodes < 10:
                print(f"  ⚠️  Small graph, results may be suboptimal")

            # ===== GNN O BYPASS =====
            if self.no_gnn:
                print(f"  → GNN BYPASS: Clustering su feature originali")
                embd = F.normalize(ft_list.float().to(device), p=2, dim=1)

            else:
                print(f"  → GNN TRAINING: In corso...")
                data.edge_index = data.edge_index.long()

                num_cluster        = int(ft_list.shape[0] * args.compress_ratio)
                input_layer_shape  = ft_list.shape[1]
                layer_shape        = [input_layer_shape] + args.hidden_dim + [num_cluster]

                model = GAE(ATTGNN(layer_shape))
                ft_list = ft_list.float().to(device)
                data    = data.to(device)
                model.to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)

                for epoch in range(args.epochs):
                    model.train()
                    optimizer.zero_grad()

                    logits, embd = model.encode(ft_list, data.edge_index, data.edge_attr)
                    embd         = F.normalize(embd, p=2, dim=1)

                    loss_recon = model.recon_loss(embd, data.edge_index)

                    dis      = pairwise_distances(embd.cpu().detach().numpy(), metric='cosine')
                    db_label = DBSCAN(eps=args.db_eps, min_samples=args.db_min, metric='precomputed').fit_predict(dis)
                    db_label = torch.from_numpy(db_label).to(device)

                    class_matrix  = torch.from_numpy(self.onehot_encoder(db_label)).float().to(device)
                    local_label   = torch.mm(class_matrix, class_matrix.t())
                    global_label  = torch.matmul(logits, logits.t())
                    loss_cluster  = F.binary_cross_entropy_with_logits(global_label, local_label)

                    loss_train = args.cluster_w * loss_cluster + (1 - args.cluster_w) * loss_recon

                    if epoch % 5 == 0:
                        print(
                            f'epoch: {epoch:3d}',
                            f'cluster loss: {loss_cluster.item():.4f}',
                            f'recon loss: {loss_recon.item():.4f}',
                            f'ALL loss: {loss_train.item():.4f}',
                        )

                    loss_train.backward()
                    optimizer.step()

                with torch.no_grad():
                    model.eval()
                    logits, embd = model.encode(ft_list, data.edge_index, data.edge_attr)

            # ===== CLUSTERING FINALE =====
            with torch.no_grad():
                lc_dis = pairwise_distances(embd.cpu().detach().numpy(), metric='cosine')
                pred   = DBSCAN(eps=args.db_eps, min_samples=args.db_min, metric='precomputed').fit_predict(lc_dis)
                pred   = pred.tolist()

            if args.post_match:
                pred = self.post_match(pred, name_pubs, name, datatype)

            clusters = self.labels_to_clusters(pred, name_pubs)

            # ===== DEBUG (primi 3 autori) =====
            if processed_authors < 3:
                n_singletons = sum(1 for c in clusters if len(c) == 1)
                print(f"\n{'='*70}")
                print(f"CLUSTERING DEBUG: {name}")
                print(f"{'='*70}")
                print(f"  Papers: {len(name_pubs)}")
                print(f"  Unique labels: {len(set(pred))}")
                print(f"  Outliers (label=-1): {sum(1 for l in pred if l == -1)}")
                print(f"  Clusters created: {len(clusters)}")
                print(f"  Cluster sizes: {sorted([len(c) for c in clusters], reverse=True)[:10]}")
                print(f"  Singletons: {n_singletons}/{len(clusters)} ({n_singletons/len(clusters)*100:.1f}%)")
                print(f"{'='*70}\n")

            results[name] = clusters
            processed_authors += 1

        # ===== STATISTICHE FINALI =====
        print("\n" + "="*70)
        print("TRAINING COMPLETED - STATISTICS")
        print("="*70)
        print(f"Total authors: {total_authors}")
        print(f"GNN trained:   {processed_authors}")
        print(f"Special cases: {special_case_authors}")

        if special_case_info:
            case_types = {}
            for _, _, _, reason in special_case_info:
                case_types[reason] = case_types.get(reason, 0) + 1
            print("\nSpecial case breakdown:")
            for reason, count in case_types.items():
                print(f"  {reason}: {count}")

        result_path = save_results(args, names, pubs, results)
        print(f"\nResults saved: {result_path}")
        print("="*70)
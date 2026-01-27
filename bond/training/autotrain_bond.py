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
from os.path import join,dirname
import os
import json
from .generate_pair import generate_pair

from params import set_params


args = set_params()

seed = args.seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device(("cuda:"+str(args.gpu)) if torch.cuda.is_available() and args.cuda else "cpu")

class BONDTrainer:
    def __init__(self) -> None:
        pass

    def onehot_encoder(self, label_list):
        if isinstance(label_list, np.ndarray):
            labels_arr = label_list
        else:
            labels_arr = np.array(label_list.cpu().detach().numpy())
        
        num_classes = max(labels_arr) + 1
        # Se max è -1 (solo outliers), num_classes diventa 0. Gestiamo il caso:
        num_classes = max(0, num_classes)
        
        # Creiamo una matrice dove gli outlier (-1) hanno una riga di soli zeri
        # così non influenzano positivamente la similarità tra loro
        onehot_mat = np.zeros((len(labels_arr), num_classes))

        for i in range(len(labels_arr)):
            if labels_arr[i] != -1:
                onehot_mat[i, labels_arr[i]] = 1

        return onehot_mat
    
    def matx2list(self, adj):
        """
        Transform similarity matrix to cluster labels.
        
        Args:
            adj: N x N similarity matrix where adj[i][j] = 1 if i and j are in same cluster
            
        Returns:
            labels: List of cluster labels, e.g. [0, 0, 1, 2, 2]
        """
        # Converti a numpy se necessario
        if not isinstance(adj, np.ndarray):
            if torch.is_tensor(adj):
                adj = adj.cpu().detach().numpy()
            else:
                adj = np.array(adj)
        
        n_papers = adj.shape[0]
        labels = [-1] * n_papers
        current_label = 0
        assigned = [False] * n_papers
        
        for i in range(n_papers):
            if assigned[i]:
                continue
            
            # Trova tutti i paper connessi a i (stesso cluster)
            cluster_members = []
            for j in range(n_papers):
                if adj[i][j] == 1:
                    cluster_members.append(j)
            
            # Assegna label
            if len(cluster_members) > 0:
                for j in cluster_members:
                    labels[j] = current_label
                    assigned[j] = True
                current_label += 1
            else:
                # Paper isolato (non dovrebbe succedere se matrice è simmetrica)
                labels[i] = -1
                assigned[i] = True
        
        return labels
        
    def post_match(self, pred, pubs, name, mode):
        """
        Post-match outliers.
        Args:
            pred(list): prediction e.g. [0, 0, -1, 1]
            pubs(list): paper-ids
            name(str): author name
            mode(str): train/valid/test
        Return:
            pred(list): after post-match e.g. [0, 0, 0, 1] 
        """
        #1 outlier from dbscan labels
        outlier = set()
        for i in range(len(pred)):
            if pred[i] == -1:
                outlier.add(i)

        #2 outlier from building graphs (relational)
        datapath = join(args.save_path, 'graph', mode, name)
        with open(join(datapath, 'rel_cp.txt'), 'r') as f:
            rel_outlier = [int(x) for x in f.read().split('\n')[:-1]] 

        for i in rel_outlier:
            outlier.add(i)
        
        print(f"post matching {len(outlier)} outliers")
        paper_pair = generate_pair(pubs, name, outlier, mode)
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
                K = K + 1

        for ii, i in enumerate(outlier):
            for jj, j in enumerate(outlier):
                if jj <= ii:
                    continue
                else:
                    if paper_pair1[i][j] >= 1.5:
                        pred[j] = pred[i]
        return pred

    def labels_to_clusters(self, pred, paper_ids):
        """
        Versione corretta: ogni outlier (-1) diventa un cluster separato (singleton).
        """
        from collections import defaultdict
        cluster_dict = defaultdict(list)
        
        # Contatore per creare ID univoci per gli outlier
        outlier_count = 0
        
        for idx, label in enumerate(pred):
            l_val = label.item() if hasattr(label, 'item') else label
            
            if l_val == -1:
                # Invece di usare -1 come chiave comune, creiamo una chiave univoca
                # Questo garantisce che il paper finisca in un cluster da solo (Precision!)
                unique_key = f"outlier_{outlier_count}"
                cluster_dict[unique_key].append(paper_ids[idx])
                outlier_count += 1
            else:
                cluster_dict[l_val].append(paper_ids[idx])
        
        return list(cluster_dict.values())

    def fit(self, datatype):
        names, pubs = load_dataset(datatype)
        
        if datatype in ['valid', 'test']:
            # Carica ground truth
            if datatype == 'valid':
                gt_file = join(args.save_path, 'src', 'sna-valid', 'sna_valid_ground_truth.json')
            else:
                gt_file = join(args.save_path, 'src', 'sna-test', 'sna_test_ground_truth.json')
            
            if os.path.exists(gt_file):
                import json
                with open(gt_file, 'r', encoding='utf-8') as f:
                    ground_truth = json.load(f)
                
                # Filtra
                gt_names = set(ground_truth.keys())
                names = [n for n in names if n in gt_names]
                pubs = {n: pubs[n] for n in names if n in pubs}
                
                print(f"\n{'='*70}")
                print(f"FILTERED TO GROUND TRUTH: {len(names)} authors")
                print(f"{'='*70}\n")
            else:
                print(f"⚠️  Ground truth file not found: {gt_file}")
        # =========================================================
        
        results = {}
        
        # Statistiche per monitorare
        total_authors = len(names)
        processed_authors = 0
        special_case_authors = 0
        special_case_info = []

        f1_list = []
        for name in names:
            print("training:", name)
            results[name] = []

            # ==== Load data ====
            label, ft_list, data = load_graph(name)
            
            # Check se load fallito
            if label is None or ft_list is None or data is None:
                print(f"  ❌ Failed to load graph files")
                # Crea cluster per tutti i paper
                name_pubs = []
                if datatype == 'train':
                    for aid in pubs[name]:
                        name_pubs.extend(pubs[name][aid])
                else:
                    for pid in pubs[name]:
                        name_pubs.append(pid)
                
                clusters = [[pid] for pid in name_pubs] if name_pubs else []
                results[name] = clusters
                special_case_authors += 1
                special_case_info.append((name, 0, 0, "Load failed"))
                continue
            
            # Get paper IDs list
            name_pubs = []
            if datatype == 'train':
                for aid in pubs[name]:
                    name_pubs.extend(pubs[name][aid])
            else:
                for pid in pubs[name]:
                    name_pubs.append(pid)

            n_nodes = ft_list.shape[0]
            n_edges = data.edge_index.shape[1]

            # ===== CHECK DI SICUREZZA =====
            if n_nodes != len(name_pubs):
                print(f"  ❌ CRITICAL ERROR: Graph({n_nodes}) != Papers({len(name_pubs)})")
                print(f"     Graph building failed for {name}!")
                print(f"     This should NOT happen if graphs were built correctly.")
                print(f"     → SKIPPING {name}")
                
                # Crea singleton clusters come fallback
                clusters = [[pid] for pid in name_pubs]
                results[name] = clusters
                special_case_authors += 1
                special_case_info.append((name, n_nodes, len(name_pubs), "Build error"))
                continue
            # ==============================
            
            print(f"  Graph: {n_nodes} nodes, {n_edges} edges")
            
            # ===== GESTIONE CASI SPECIALI =====
            
            # CASO 1: Single paper
            if n_nodes == 1:
                print(f"  → Single paper, creating 1 cluster")
                clusters = [[name_pubs[0]]]
                results[name] = clusters
                special_case_authors += 1
                special_case_info.append((name, n_nodes, n_edges, "Single paper"))
                continue
            
            # CASO 2: No edges
            if n_edges == 0:
                print(f"  → No edges, creating {n_nodes} singleton clusters")
                clusters = [[pid] for pid in name_pubs]
                results[name] = clusters
                special_case_authors += 1
                special_case_info.append((name, n_nodes, n_edges, "No edges"))
                continue
            
            # CASO 3: Very small graph (2-4 nodes)
            if n_nodes <= 4:
                print(f"  → Very small graph, using simplified clustering")
                try:
                    from sklearn.metrics.pairwise import euclidean_distances
                    
                    distances = euclidean_distances(ft_list.cpu().numpy())
                    labels = DBSCAN(eps=args.db_eps*2, min_samples=1, metric='precomputed').fit_predict(distances)
                    
                    clusters = self.labels_to_clusters(labels.tolist(), name_pubs)
                    results[name] = clusters
                    
                    print(f"  → Created {len(clusters)} clusters")
                    special_case_authors += 1
                    special_case_info.append((name, n_nodes, n_edges, "Simplified"))
                    continue
                except Exception as e:
                    print(f"  → Simplified clustering failed: {e}, using singletons")
                    clusters = [[pid] for pid in name_pubs]
                    results[name] = clusters
                    special_case_authors += 1
                    special_case_info.append((name, n_nodes, n_edges, "Failed"))
                    continue
            
            # CASO 4: Small graph (5-9 nodes) - warning ma continua
            if n_nodes < 10:
                print(f"  ⚠️  Small graph, results may be suboptimal")
            
            # ===== TRAINING NORMALE (>=5 nodes con edges) =====
            
            # Assicura tipo corretto
            data.edge_index = data.edge_index.long()
            
            num_cluster = int(ft_list.shape[0]*args.compress_ratio)
            layer_shape = []
            input_layer_shape = ft_list.shape[1]
            hidden_layer_shape = args.hidden_dim
            output_layer_shape = num_cluster
            
            layer_shape.append(input_layer_shape)
            layer_shape.extend(hidden_layer_shape)
            layer_shape.append(output_layer_shape)

            # ==== Init model ====
            model = GAE(ATTGNN(layer_shape))
            ft_list = ft_list.float()
            ft_list = ft_list.to(device)
            data = data.to(device)
            model.to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)

            for epoch in range(args.epochs):
                # ==== Train ====
                model.train()
                optimizer.zero_grad()
                
                logits, embd = model.encode(ft_list, data.edge_index, data.edge_attr)
                dis = pairwise_distances(embd.cpu().detach().numpy(), metric='cosine')
                db_label = DBSCAN(eps=args.db_eps, min_samples=args.db_min, metric='precomputed').fit_predict(dis) 
                db_label = torch.from_numpy(db_label)
                db_label = db_label.to(device) 
                '''
                print(f"\n[DEBUG] {name}:")
                #print(f"  -> DBSCAN Labels trovate: {unique_labels}")
                if -1 in unique_labels:
                    print(f"  -> ATTENZIONE: DBSCAN ha trovato RUMORE (-1).")

                if args.post_match:
                    print("  -> Eseguo POST-MATCH (che potrebbe separare i cluster).")
                else:
                    print("  -> POST-MATCH disabilitato.")

                unique_labels = set(db_label.cpu().numpy())
                '''
                # change to one-hot form
                class_matrix = torch.from_numpy(self.onehot_encoder(db_label))
                # get N * N matrix
                local_label = torch.mm(class_matrix, class_matrix.t())
                local_label = local_label.float()
                local_label = local_label.to(device)

                global_label = torch.matmul(logits, logits.t())
                
                loss_cluster = F.binary_cross_entropy_with_logits(global_label, local_label)
                loss_recon = model.recon_loss(embd, data.edge_index)
                
                # Controllo per NaN
                if torch.isnan(loss_cluster) or torch.isnan(loss_recon):
                    print(f"  WARNING: NaN detected at epoch {epoch}")
                    continue

                w_cluster = args.cluster_w
                w_recon = 1 - w_cluster
                loss_train = w_cluster * loss_cluster + w_recon * loss_recon
                
                if (epoch % 5) == 0:
                    print(
                        'epoch: {:3d}'.format(epoch),
                        'cluster loss: {:.4f}'.format(loss_cluster.item()),
                        'recon loss: {:.4f}'.format(loss_recon.item()),
                        'ALL loss: {:.4f}'.format(loss_train.item())
                    )

                loss_train.backward()
                optimizer.step()
            
            # ==== Evaluate ====
            with torch.no_grad():
                model.eval()
                logits, embd = model.encode(ft_list, data.edge_index, data.edge_attr)
                gl_label = torch.matmul(logits, logits.t())
                
                lc_dis = pairwise_distances(embd.cpu().detach().numpy(), metric='cosine')
                local_label = DBSCAN(eps=args.db_eps, min_samples=args.db_min, metric='precomputed').fit_predict(lc_dis) 
                gl_dis = pairwise_distances(gl_label.cpu().detach().numpy(), metric='cosine')
                gl_label = DBSCAN(eps=args.db_eps, min_samples=args.db_min, metric='precomputed').fit_predict(gl_dis) 
                
                # ===== USA DIRETTAMENTE I LABEL =====
                pred = local_label.tolist()

                if args.post_match:
                    pred = self.post_match(pred, name_pubs, name, datatype)

                # ===== CONVERTI LABEL IN CLUSTER =====
                clusters = self.labels_to_clusters(pred, name_pubs)
                
                # ===== DEBUG (primi 3 autori processati normalmente) =====
                if processed_authors < 3:
                    print(f"\n{'='*70}")
                    print(f"CLUSTERING DEBUG: {name}")
                    print(f"{'='*70}")
                    print(f"  Papers: {len(name_pubs)}")
                    print(f"  Unique labels: {len(set(pred))}")
                    print(f"  Outliers (label=-1): {sum(1 for l in pred if l == -1)}")
                    print(f"  Clusters created: {len(clusters)}")
                    print(f"  Cluster sizes: {sorted([len(c) for c in clusters], reverse=True)[:10]}")
                    
                    n_singletons = sum(1 for c in clusters if len(c) == 1)
                    print(f"  Singletons: {n_singletons}/{len(clusters)} ({n_singletons/len(clusters)*100:.1f}%)")
                    print(f"{'='*70}\n")
                # ==================================================
                
                # Save results
                results[name] = clusters
                processed_authors += 1

        # Stampa statistiche finali
        print("\n" + "="*70)
        print("TRAINING COMPLETED - STATISTICS")
        print("="*70)
        print(f"Total authors: {total_authors}")
        print(f"GNN trained: {processed_authors}")
        print(f"Special cases: {special_case_authors}")
        
        if special_case_info:
            print(f"\nSpecial case breakdown:")
            case_types = {}
            for name, nodes, edges, reason in special_case_info:
                case_types[reason] = case_types.get(reason, 0) + 1
            
            for reason, count in case_types.items():
                print(f"  {reason}: {count}")
        
        result_path = save_results(names, pubs, results)
        print(f"\nResults saved: {result_path}")
        print("="*70)
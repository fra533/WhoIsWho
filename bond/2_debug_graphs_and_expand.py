"""
Feature extraction da FILE RAW DEI GRAFI
Legge adj_attr.txt, feats_p.npy, p_label.npy, rel_cp.txt  dataset/graph
In input prende le F1 calcolate per il train e il test calcolato su un dataset splittato casualmente
e gli embeddings. 
"""

import json
import numpy as np
import pandas as pd
import os
from os.path import join, exists
import pickle
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist
import networkx as nx

# ======================== CONFIGURAZIONE ========================
BASE_PATH = r"C:\Users\franc\OneDrive - Alma Mater Studiorum Università di Bologna\Desktop\BOND-OC\WhoIsWho\bond"

TRAIN_RESULTS = join(BASE_PATH, "out_train", "res.json")
TEST_RESULTS = join(BASE_PATH, "out", "res.json")

TRAIN_GT = join(BASE_PATH, "dataset", "data", "src", "train", "train_author.json")
TEST_GT = join(BASE_PATH, "dataset", "data", "src", "sna-valid", "sna_valid_ground_truth.json")

DATA_PATH = join(BASE_PATH, "dataset", "data")
OUTPUT_DIR = join(BASE_PATH, "bond_full_features_raw")
# ================================================================


class RawGraphAnalyzer:
    """Analizza features da file raw grafi + embeddings"""
    
    def __init__(self):
        self.author_features = {}
        self.author_results = {}
        print("="*80)
        print("FULL FEATURE EXTRACTION - RAW GRAPH FILES")
        print("Legge: adj_attr.txt, feats_p.npy, embeddings")
        print("="*80)
    
    def load_all_results(self):
        """Carica risultati e GT"""
        print("\n[1/4] Caricamento risultati...")
        
        all_results = {}
        all_gt = {}
        
        # Risultati
        with open(TRAIN_RESULTS, 'r', encoding='utf-8') as f:
            all_results.update(json.load(f))
        with open(TEST_RESULTS, 'r', encoding='utf-8') as f:
            all_results.update(json.load(f))
        
        print(f"  Risultati: {len(all_results)} autori")
        
        # Ground truth
        with open(TRAIN_GT, 'r', encoding='utf-8') as f:
            all_gt.update(json.load(f))
        with open(TEST_GT, 'r', encoding='utf-8') as f:
            all_gt.update(json.load(f))
        
        print(f"  Ground truth: {len(all_gt)} autori")
        
        # Calcola F1
        for name in all_results:
            if name in all_gt:
                metrics = self._compute_f1(all_results[name], all_gt[name])
                self.author_results[name] = metrics
        
        print(f"  F1 calcolato: {len(self.author_results)} autori")
        
        return pd.DataFrame.from_dict(self.author_results, orient='index')
    
    def extract_all_features(self, author_names):
        """Estrae features per tutti gli autori"""
        print(f"\n[2/4] Estrazione features (embeddings + raw grafi)...")
        print(f"  Target: {len(author_names)} autori")
        
        success_count = 0
        skipped = []
        
        for i, name in enumerate(author_names):
            if (i + 1) % 20 == 0:
                print(f"    Processati: {i+1}/{len(author_names)} (success: {success_count})")
            
            features = self._extract_author_features(name)
            if features:
                self.author_features[name] = features
                success_count += 1
            else:
                skipped.append(name)
        
        print(f"  ✓ Features estratte: {success_count} autori")
        print(f"  ⚠️  Saltati: {len(skipped)} autori")
        
        if len(skipped) > 0 and len(skipped) <= 5:
            print(f"  Autori saltati: {', '.join(skipped)}")
        
        return pd.DataFrame.from_dict(self.author_features, orient='index')
    
    def _extract_author_features(self, name):
        """Estrae features per un singolo autore"""
        features = {}
        
        # Trova directory autore
        author_dir = None
        mode_found = None
        
        for mode in ['train', 'valid', 'test']:
            candidate = join(DATA_PATH, "graph", mode, name)
            if exists(candidate):
                author_dir = candidate
                mode_found = mode
                break
        
        if not author_dir:
            return None
        
        # ====== FEATURES DA EMBEDDINGS ======
        emb_path = join(DATA_PATH, "paper_emb", mode_found, name, "ptext_emb.pkl")
        
        if exists(emb_path):
            try:
                with open(emb_path, 'rb') as f:
                    emb_data = pickle.load(f)
                
                embeddings = list(emb_data.values())
                
                if len(embeddings) > 0:
                    features['num_papers'] = len(embeddings)
                    
                    if len(embeddings) == 1:
                        features['emb_mean_similarity'] = 0.0
                        features['emb_std_similarity'] = 0.0
                        features['emb_effective_dim'] = 1.0
                        features['emb_intra_variance'] = 0.0
                        features['emb_density'] = 0.0
                    else:
                        emb_matrix = np.array(embeddings)
                        
                        # Similarity
                        sim_matrix = cosine_similarity(emb_matrix)
                        np.fill_diagonal(sim_matrix, 0)
                        features['emb_mean_similarity'] = sim_matrix.mean()
                        features['emb_std_similarity'] = sim_matrix.std()
                        
                        # PCA
                        pca = PCA()
                        pca.fit(emb_matrix)
                        cum_var = np.cumsum(pca.explained_variance_ratio_)
                        features['emb_effective_dim'] = np.argmax(cum_var >= 0.95) + 1
                        
                        # Variance & Density
                        features['emb_intra_variance'] = np.var(emb_matrix)
                        distances = pdist(emb_matrix, metric='euclidean')
                        features['emb_density'] = 1.0 / (distances.mean() + 1e-6)
                else:
                    features['num_papers'] = 0
            except:
                features['num_papers'] = 0
        else:
            return None  # Se non c'è embedding, skip
        
        # ====== FEATURES DA GRAFO RAW ======
        adj_file = join(author_dir, "adj_attr.txt")
        
        if exists(adj_file):
            try:
                # Leggi grafo da adj_attr.txt
                G = nx.Graph()
                
                with open(adj_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            node1 = int(parts[0])
                            node2 = int(parts[1])
                            # Attributo opzionale (peso)
                            weight = float(parts[2]) if len(parts) > 2 else 1.0
                            G.add_edge(node1, node2, weight=weight)
                
                if G.number_of_nodes() > 0:
                    features['graph_nodes'] = G.number_of_nodes()
                    features['graph_edges'] = G.number_of_edges()
                    
                    # Density
                    n_nodes = G.number_of_nodes()
                    max_edges = n_nodes * (n_nodes - 1) / 2
                    features['graph_density'] = G.number_of_edges() / max_edges if max_edges > 0 else 0
                    
                    # Degree stats
                    degrees = [d for n, d in G.degree()]
                    features['graph_avg_degree'] = np.mean(degrees) if degrees else 0
                    features['graph_max_degree'] = max(degrees) if degrees else 0
                    
                    # Clustering coefficient
                    features['graph_clustering_coef'] = nx.average_clustering(G) if n_nodes > 1 else 0
                    
                    # Connected components
                    components = list(nx.connected_components(G))
                    features['graph_num_components'] = len(components)
                    
                    if components:
                        largest_cc = max(components, key=len)
                        features['graph_largest_cc_ratio'] = len(largest_cc) / n_nodes
                    else:
                        features['graph_largest_cc_ratio'] = 0
                    
                    # Diameter (se connesso)
                    try:
                        if nx.is_connected(G):
                            features['graph_diameter'] = nx.diameter(G)
                        else:
                            largest_cc = max(nx.connected_components(G), key=len)
                            subG = G.subgraph(largest_cc)
                            features['graph_diameter'] = nx.diameter(subG)
                    except:
                        features['graph_diameter'] = 0
                else:
                    # Grafo vuoto
                    for key in ['graph_nodes', 'graph_edges', 'graph_density', 
                               'graph_avg_degree', 'graph_max_degree', 'graph_clustering_coef',
                               'graph_num_components', 'graph_largest_cc_ratio', 'graph_diameter']:
                        features[key] = 0
            except Exception as e:
                # Errore lettura grafo - usa valori default
                for key in ['graph_nodes', 'graph_edges', 'graph_density', 
                           'graph_avg_degree', 'graph_max_degree', 'graph_clustering_coef',
                           'graph_num_components', 'graph_largest_cc_ratio', 'graph_diameter']:
                    features[key] = 0
        else:
            # File grafo non trovato
            for key in ['graph_nodes', 'graph_edges', 'graph_density', 
                       'graph_avg_degree', 'graph_max_degree', 'graph_clustering_coef',
                       'graph_num_components', 'graph_largest_cc_ratio', 'graph_diameter']:
                features[key] = 0
        
        return features
    
    def _compute_f1(self, pred_clusters, gt_data):
        """Calcola F1 pairwise"""
        if not isinstance(pred_clusters, list):
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        predicted_pubs = {}
        for idx, cluster in enumerate(pred_clusters):
            for pid in cluster:
                predicted_pubs[pid] = idx
        
        # Ground truth
        true_clusters = []
        if isinstance(gt_data, dict):
            for author_id, papers in gt_data.items():
                if isinstance(papers, list) and len(papers) > 0:
                    true_clusters.append(papers)
        elif isinstance(gt_data, list):
            true_clusters = gt_data
        
        # Labels
        pubs = []
        true_labels = []
        for idx, cluster in enumerate(true_clusters):
            for pid in cluster:
                pubs.append(pid)
                true_labels.append(idx)
        
        filtered_pubs = [pid for pid in pubs if pid in predicted_pubs]
        
        if len(filtered_pubs) == 0:
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        filtered_true_labels = [true_labels[i] for i, pid in enumerate(pubs) 
                               if pid in predicted_pubs]
        predict_labels = [predicted_pubs[pid] for pid in filtered_pubs]
        
        # Pairwise
        TP = TP_FP = TP_FN = 0.0
        for i in range(len(filtered_true_labels)):
            for j in range(i + 1, len(filtered_true_labels)):
                if filtered_true_labels[i] == filtered_true_labels[j]:
                    TP_FN += 1
                if predict_labels[i] == predict_labels[j]:
                    TP_FP += 1
                if (filtered_true_labels[i] == filtered_true_labels[j]) and \
                   (predict_labels[i] == predict_labels[j]):
                    TP += 1
        
        precision = TP / TP_FP if TP_FP > 0 else 0.0
        recall = TP / TP_FN if TP_FN > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {'f1': f1, 'precision': precision, 'recall': recall}
    
    def run_pipeline(self):
        """Pipeline completo"""
        
        # 1. Carica risultati
        results_df = self.load_all_results()
        author_names = list(results_df.index)
        
        # 2. Estrai features
        features_df = self.extract_all_features(author_names)
        
        # 3. Merge
        print(f"\n[3/4] Merge features + risultati...")
        merged_df = features_df.join(results_df, how='inner')
        print(f"  Dataset combinato: {len(merged_df)} autori")
        
        # 4. Export
        print(f"\n[4/4] Salvataggio...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        output_file = join(OUTPUT_DIR, 'full_features_raw.csv')
        merged_df.to_csv(output_file)
        print(f"  ✓ {output_file}")
        
        print("\n" + "="*80)
        print("✅ ANALISI COMPLETA (RAW GRAFI) COMPLETATA!")
        print("="*80)
        print(f"\nAutori analizzati: {len(merged_df)}")
        print(f"Features estratte: {len(merged_df.columns) - 3}")
        print(f"\nF1 Score Summary:")
        print(f"  Media: {merged_df['f1'].mean():.4f}")
        print(f"  Range: [{merged_df['f1'].min():.4f}, {merged_df['f1'].max():.4f}]")
        
        # Feature correlations
        print(f"\nTop 5 features correlate con F1:")
        feature_cols = [c for c in merged_df.columns if c not in ['f1', 'precision', 'recall']]
        correlations = merged_df[feature_cols].corrwith(merged_df['f1']).abs().sort_values(ascending=False)
        for i, (feat, corr) in enumerate(correlations.head(5).items(), 1):
            print(f"  {i}. {feat:30s}: {corr:.4f}")
        
        print(f"\n💡 Prossimo step:")
        print(f"  Modifica 1_split_dataset_80_20_v2.py:")
        print(f"    INPUT_CSV = 'bond_full_features_raw/full_features_raw.csv'")
        print(f"  Poi esegui:")
        print(f"    python 1_split_dataset_80_20_v2.py")
        print("="*80)


def main():
    analyzer = RawGraphAnalyzer()
    analyzer.run_pipeline()


if __name__ == "__main__":
    main()
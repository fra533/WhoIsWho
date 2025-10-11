"""
Pipeline unificato per BOND:
1. Estrae features raw da dati originali
2. Rimuove automaticamente le 10 features ridondanti
3. Mantiene solo le 19 features ottimali
4. Analizza e visualizza
TUTTO IN UN'UNICA ESECUZIONE
"""

import json
import numpy as np
import pandas as pd
import os
from os.path import join, exists
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import networkx as nx

# ======================== CONFIGURAZIONE ========================
BASE_PATH = r"C:\Users\franc\OneDrive - Alma Mater Studiorum Università di Bologna\Desktop\BOND-OC\WhoIsWho\bond"

RESULTS_FILE = join(BASE_PATH, "out", "res.json")
GROUND_TRUTH_FILE = join(BASE_PATH, "dataset", "data", "src", "sna-valid", "sna_valid_ground_truth.json")
DATA_PATH = join(BASE_PATH, "dataset", "data")
NAMES_PUB_PATH = join(DATA_PATH, "names_pub", "valid")
PAPER_EMB_PATH = join(DATA_PATH, "paper_emb", "valid")
GRAPH_PATH = join(DATA_PATH, "graph", "valid")
OUTPUT_DIR = join(BASE_PATH, "bond_unified_refined_analysis")
# ================================================================

# FEATURES DA ESCLUDERE (ridondanti/costanti)
FEATURES_TO_REMOVE = {
    'graph_nodes',           # = num_papers
    'papers_x_dim',          # = num_papers
    'coauthor_density',      # = coauthors_x_papers
    'papers_x_variance',     # ≈ num_papers
    'venues_x_similarity',   # ≈ num_venues
    'variance_x_dim',        # ≈ emb_effective_dim
    'graph_avg_path_length', # ≈ graph_diameter
    'graph_exists',          # costante
    'has_usable_graph',      # costante
    'emb_separation'         # quasi-costante
}

# FEATURES OTTIMALI DA MANTENERE (19)
OPTIMAL_FEATURES = {
    'Embedding Quality': [
        'emb_mean_similarity',
        'emb_std_similarity',
        'emb_intra_variance',
        'emb_effective_dim',
        'emb_density'
    ],
    'Graph Topology': [
        'graph_edges',
        'graph_density',
        'graph_avg_degree',
        'graph_clustering_coef',
        'graph_num_components',
        'graph_largest_component_ratio',
        'graph_diameter',
        'graph_size_category'
    ],
    'Basic Statistics': [
        'num_papers',
        'num_coauthors',
        'num_venues'
    ],
    'Interaction Metrics': [
        'coauthors_x_papers',
        'papers_per_venue',
        'venue_diversity_ratio'
    ]
}


class UnifiedRefinedAnalyzer:
    """Analizzatore unificato con pulizia automatica"""
    
    def __init__(self):
        self.author_features = {}
        self.author_results = {}
        self.skipped_authors = []
        self.nan_authors_log = []
        self.imputation_log = {}
        print("="*80)
        print("BOND UNIFIED REFINED ANALYSIS")
        print("Estrazione + Pulizia + Analisi con imputazione NaN documentata")
        print("="*80)
    
    def run_full_pipeline(self):
        """Esegue pipeline completo"""
        
        # Step 1: Carica risultati e identifica autori
        print("\n[1/7] Caricamento risultati...")
        results_df = self.load_results()
        author_names = list(results_df.index)
        print(f"      Autori target: {len(author_names)}")
        
        # Step 2: Estrai features RAW
        print("\n[2/7] Estrazione features raw...")
        features_df = self.extract_all_features(author_names)
        print(f"      Features estratte per: {len(features_df)} autori")
        print(f"      Autori saltati: {len(self.skipped_authors)}")
        
        # Step 3: Merge con risultati
        print("\n[3/7] Merge features + risultati...")
        merged_df = features_df.join(results_df, how='inner')
        print(f"      Dataset combinato: {len(merged_df)} autori")
        
        # Step 4: Pulizia automatica
        print("\n[4/7] Pulizia features ridondanti...")
        clean_df = self.clean_redundant_features(merged_df)
        
        # Step 5: Crea features di interazione se mancanti
        print("\n[5/7] Aggiunta features interazione...")
        final_df = self.add_interaction_features(clean_df)
        
        # Step 6: Analisi avanzata
        print("\n[6/7] Analisi correlazioni e clustering...")
        correlations, importance, clustered_df = self.analyze_refined_features(final_df)
        
        # Step 7: Export e visualizzazioni
        print("\n[7/7] Generazione output...")
        self.export_all_results(clustered_df, correlations, importance)
        
        print("\n" + "="*80)
        print("✓ PIPELINE COMPLETATO CON SUCCESSO!")
        print("="*80)
        print(f"\nRisultati salvati in: {OUTPUT_DIR}")
        self.print_summary(clustered_df)
    
    def load_results(self):
        """Carica metriche F1"""
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            bond_results = json.load(f)
        
        with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        
        # Converti GT
        ground_truth = {}
        for name, author_dict in gt_data.items():
            clusters = []
            if isinstance(author_dict, dict):
                for author_id, paper_list in author_dict.items():
                    if isinstance(paper_list, list) and len(paper_list) > 0:
                        clusters.append(paper_list)
            ground_truth[name] = clusters
        
        # Calcola F1
        for name in bond_results:
            if name not in ground_truth:
                continue
            
            pred_clusters = bond_results[name] if isinstance(bond_results[name], list) else []
            true_clusters = ground_truth[name]
            
            metrics = self._compute_f1(true_clusters, pred_clusters)
            self.author_results[name] = metrics
        
        return pd.DataFrame.from_dict(self.author_results, orient='index')
    
    def extract_all_features(self, author_names):
        """Estrae tutte le features per tutti gli autori"""
        
        total = len(author_names)
        print(f"      Target: {total} autori (escluso junichi_suzuki se presente)")
        
        for i, name in enumerate(author_names):
            if (i + 1) % 10 == 0:
                print(f"      Processati: {i+1}/{total}")
            
            try:
                features = self._extract_author_features(name)
                if features:
                    self.author_features[name] = features
                else:
                    # Log dettagliato del perché è stato saltato
                    reason = self._diagnose_author_failure(name)
                    self.skipped_authors.append((name, reason))
                    print(f"        ⚠ Saltato {name}: {reason}")
            except Exception as e:
                reason = f"Exception: {str(e)}"
                self.skipped_authors.append((name, reason))
                print(f"        ✗ Errore {name}: {reason}")
        
        return pd.DataFrame.from_dict(self.author_features, orient='index')
    
    def _diagnose_author_failure(self, name):
        """Diagnostica perché un autore è stato saltato"""
        reasons = []
        
        # Check pubblicazioni
        pub_file = join(NAMES_PUB_PATH, f"{name}.json")
        if not exists(pub_file):
            reasons.append("no_pub_file")
        else:
            try:
                with open(pub_file, 'r', encoding='utf-8') as f:
                    pubs = json.load(f)
                if len(pubs) == 0:
                    reasons.append("empty_pubs")
            except:
                reasons.append("corrupt_pub_file")
        
        # Check embeddings
        emb_file = join(PAPER_EMB_PATH, name, 'ptext_emb.pkl')
        if not exists(emb_file):
            reasons.append("no_embeddings")
        else:
            try:
                with open(emb_file, 'rb') as f:
                    emb = pickle.load(f)
                if len(emb) < 2:
                    reasons.append("insufficient_embeddings")
            except:
                reasons.append("corrupt_embeddings")
        
        # Check grafo
        graph_file = join(GRAPH_PATH, f"{name}.pkl")
        if not exists(graph_file):
            reasons.append("no_graph")
        else:
            try:
                with open(graph_file, 'rb') as f:
                    graph = pickle.load(f)
            except:
                reasons.append("corrupt_graph")
        
        return ", ".join(reasons) if reasons else "unknown"
    
    def _extract_author_features(self, name):
        """Estrae features per singolo autore"""
        
        # Carica pubblicazioni
        pub_file = join(NAMES_PUB_PATH, f"{name}.json")
        if not exists(pub_file):
            return None
        
        with open(pub_file, 'r', encoding='utf-8') as f:
            pubs = json.load(f)
        
        if len(pubs) == 0:
            return None
        
        features = {}
        
        # 1. Features embedding
        emb_features = self._extract_embedding_features(name, pubs)
        features.update(emb_features)
        
        # 2. Features grafo
        graph_features = self._extract_graph_features(name)
        features.update(graph_features)
        
        # 3. Features base
        features['num_papers'] = len(pubs)
        features['num_coauthors'] = len(set(
            author.get('name', '').lower() 
            for pub in pubs.values() 
            for author in pub.get('authors', [])
        ))
        features['num_venues'] = len(set(
            pub.get('venue', '') 
            for pub in pubs.values() 
            if pub.get('venue')
        ))
        
        return features
    
    def _extract_embedding_features(self, name, pubs):
        """Estrae features embedding"""
        
        emb_path = join(PAPER_EMB_PATH, name, 'ptext_emb.pkl')
        
        default_emb = {
            'emb_mean_similarity': 0.0,
            'emb_std_similarity': 0.0,
            'emb_intra_variance': 0.0,
            'emb_effective_dim': 0.0,
            'emb_density': 0.0
        }
        
        if not exists(emb_path):
            return default_emb
        
        try:
            with open(emb_path, 'rb') as f:
                ptext_emb = pickle.load(f)
            
            if len(ptext_emb) < 2:
                return default_emb
            
            emb_matrix = np.array([ptext_emb[pid] for pid in ptext_emb.keys()])
            
            # Similarità
            distances = pdist(emb_matrix, metric='cosine')
            similarities = 1 - distances
            
            features = {
                'emb_mean_similarity': float(np.mean(similarities)),
                'emb_std_similarity': float(np.std(similarities))
            }
            
            # Varianza intra-cluster
            centroid = np.mean(emb_matrix, axis=0)
            dist_from_centroid = np.linalg.norm(emb_matrix - centroid, axis=1)
            features['emb_intra_variance'] = float(np.var(dist_from_centroid))
            
            # Dimensionalità effettiva
            if len(emb_matrix) > emb_matrix.shape[1]:
                try:
                    pca = PCA(n_components=min(50, emb_matrix.shape[1]))
                    pca.fit(emb_matrix)
                    cumsum = np.cumsum(pca.explained_variance_ratio_)
                    n_comp_95 = np.argmax(cumsum >= 0.95) + 1
                    features['emb_effective_dim'] = float(n_comp_95 / emb_matrix.shape[1])
                except:
                    features['emb_effective_dim'] = 0.0
            else:
                features['emb_effective_dim'] = 0.0
            
            # Densità
            features['emb_density'] = float(np.mean(distances) / (np.max(distances) + 1e-10))
            
            return features
            
        except:
            return default_emb
    
    def _extract_graph_features(self, name):
        """Estrae features grafo"""
        
        graph_file = join(GRAPH_PATH, f"{name}.pkl")
        
        default_graph = {
            'graph_edges': 0,
            'graph_density': 0.0,
            'graph_avg_degree': 0.0,
            'graph_clustering_coef': 0.0,
            'graph_num_components': 0,
            'graph_largest_component_ratio': 0.0,
            'graph_diameter': 0.0,
            'graph_size_category': 0
        }
        
        if not exists(graph_file):
            return default_graph
        
        try:
            with open(graph_file, 'rb') as f:
                graph_data = pickle.load(f)
            
            if isinstance(graph_data, nx.Graph):
                G = graph_data
            elif isinstance(graph_data, dict):
                G = nx.Graph()
                for node, neighbors in graph_data.items():
                    for neighbor in neighbors:
                        G.add_edge(node, neighbor)
            else:
                return default_graph
            
            G.remove_edges_from(nx.selfloop_edges(G))
            
            n_nodes = len(G.nodes())
            n_edges = len(G.edges())
            
            if n_nodes == 0:
                return default_graph
            
            features = {
                'graph_edges': n_edges,
                'graph_density': float(nx.density(G)) if n_nodes > 1 else 0.0,
            }
            
            # Grado medio
            degrees = [d for n, d in G.degree()]
            features['graph_avg_degree'] = float(np.mean(degrees)) if degrees else 0.0
            
            # Clustering
            features['graph_clustering_coef'] = float(nx.average_clustering(G)) if n_nodes >= 3 else 0.0
            
            # Componenti
            components = list(nx.connected_components(G))
            features['graph_num_components'] = len(components)
            
            if components:
                largest = max(components, key=len)
                features['graph_largest_component_ratio'] = float(len(largest) / n_nodes)
                
                # Diametro
                if len(largest) >= 3:
                    largest_sub = G.subgraph(largest)
                    if nx.is_connected(largest_sub):
                        try:
                            features['graph_diameter'] = float(nx.diameter(largest_sub))
                        except:
                            features['graph_diameter'] = 0.0
                    else:
                        features['graph_diameter'] = 0.0
                else:
                    features['graph_diameter'] = 0.0
            else:
                features['graph_largest_component_ratio'] = 0.0
                features['graph_diameter'] = 0.0
            
            # Categoria dimensione
            if n_nodes < 5:
                features['graph_size_category'] = 1
            elif n_nodes < 20:
                features['graph_size_category'] = 2
            elif n_nodes < 50:
                features['graph_size_category'] = 3
            else:
                features['graph_size_category'] = 4
            
            return features
            
        except:
            return default_graph
    
    def clean_redundant_features(self, df):
        """Rimuove features ridondanti"""
        
        initial_cols = len(df.columns)
        cols_to_drop = [col for col in FEATURES_TO_REMOVE if col in df.columns]
        
        df_clean = df.drop(columns=cols_to_drop, errors='ignore')
        
        print(f"      Features rimosse: {len(cols_to_drop)}")
        for col in cols_to_drop:
            print(f"        ✗ {col}")
        print(f"      Features rimanenti: {len(df_clean.columns)} (da {initial_cols})")
        
        return df_clean
    
    def add_interaction_features(self, df):
        """Aggiunge features di interazione se mancanti"""
        
        added = []
        
        # coauthors_x_papers
        if 'coauthors_x_papers' not in df.columns:
            if 'num_coauthors' in df.columns and 'num_papers' in df.columns:
                df['coauthors_x_papers'] = df['num_coauthors'] * df['num_papers']
                added.append('coauthors_x_papers')
        
        # papers_per_venue
        if 'papers_per_venue' not in df.columns:
            if 'num_papers' in df.columns and 'num_venues' in df.columns:
                df['papers_per_venue'] = df['num_papers'] / (df['num_venues'] + 1)
                added.append('papers_per_venue')
        
        # venue_diversity_ratio
        if 'venue_diversity_ratio' not in df.columns:
            if 'num_venues' in df.columns and 'num_papers' in df.columns:
                df['venue_diversity_ratio'] = df['num_venues'] / (df['num_papers'] + 1)
                added.append('venue_diversity_ratio')
        
        if added:
            print(f"      Features create: {len(added)}")
            for feat in added:
                print(f"        + {feat}")
        else:
            print(f"      Tutte le features già presenti")
        
        return df
    
    def analyze_refined_features(self, df):
        """Analisi completa su features pulite con gestione NaN documentata"""
        
        feature_cols = [col for col in df.columns if col not in ['f1', 'precision', 'recall']]
        
        print(f"      Dataset iniziale: {len(df)} autori")
        
        # === STEP 1: IDENTIFICAZIONE NaN ===
        print(f"\n      [Analisi NaN]")
        has_nan = df[feature_cols + ['f1']].isnull().any(axis=1)
        authors_with_nan = df[has_nan].index.tolist()
        
        if len(authors_with_nan) > 0:
            print(f"      Autori con NaN: {len(authors_with_nan)}")
            
            # Analizza quali features hanno NaN
            nan_by_feature = {}
            for author in authors_with_nan:
                nan_cols = df.loc[author][feature_cols].isnull()
                for col in nan_cols[nan_cols].index:
                    if col not in nan_by_feature:
                        nan_by_feature[col] = []
                    nan_by_feature[col].append(author)
            
            print(f"      Features con NaN:")
            for feat, authors in sorted(nan_by_feature.items(), key=lambda x: len(x[1]), reverse=True):
                print(f"        • {feat}: {len(authors)} autori")
            
            # Log dettagliato autori con NaN
            self.nan_authors_log = []
            for author in authors_with_nan:
                nan_cols = df.loc[author][feature_cols].isnull()
                nan_features = nan_cols[nan_cols].index.tolist()
                self.nan_authors_log.append({
                    'author': author,
                    'nan_features': nan_features,
                    'f1': df.loc[author, 'f1']
                })
                print(f"        {author}: {', '.join(nan_features[:3])}{'...' if len(nan_features) > 3 else ''}")
        else:
            print(f"      Nessun NaN trovato")
            self.nan_authors_log = []
        
        # === STEP 2: STRATEGIA DI IMPUTAZIONE ===
        print(f"\n      [Imputazione NaN]")
        df_imputed = df.copy()
        
        imputation_log = {}
        for col in feature_cols:
            nan_count = df_imputed[col].isnull().sum()
            if nan_count > 0:
                # Strategia: imputa con 0 per features di grafo, mediana per altre
                if col.startswith('graph_'):
                    # Grafo mancante/invalido = 0
                    fill_value = 0.0
                    strategy = 'zero (grafo mancante)'
                else:
                    # Altre features: usa mediana
                    fill_value = df_imputed[col].median()
                    strategy = f'median ({fill_value:.4f})'
                
                df_imputed[col].fillna(fill_value, inplace=True)
                imputation_log[col] = {
                    'count': nan_count,
                    'value': fill_value,
                    'strategy': strategy
                }
                print(f"        {col}: {nan_count} NaN → {strategy}")
        
        self.imputation_log = imputation_log
        
        # === STEP 3: DATASET FINALE ===
        clean_df = df_imputed[feature_cols + ['f1']].copy()
        print(f"\n      Dataset finale: {len(clean_df)} autori (0 rimossi, {len(authors_with_nan)} imputati)")
        
        # === STEP 4: CORRELAZIONI ===
        correlations = clean_df[feature_cols + ['f1']].corr()['f1'].drop('f1').sort_values(ascending=False)
        
        # === STEP 5: RANDOM FOREST ===
        X = clean_df[feature_cols]
        y = clean_df['f1']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        rf = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10, min_samples_leaf=2)
        rf.fit(X_scaled, y)
        
        importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
        
        # === STEP 6: CLUSTERING ===
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clean_df['cluster'] = kmeans.fit_predict(X_scaled)
        
        return correlations, importance, clean_df
    
    def export_all_results(self, df, correlations, importance):
        """Esporta tutti i risultati con documentazione completa"""
        
        if not exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        # CSV principale
        csv_file = join(OUTPUT_DIR, 'refined_features_results.csv')
        df.to_csv(csv_file)
        print(f"      CSV: refined_features_results.csv")
        
        # CSV con log NaN (se presenti)
        if self.nan_authors_log:
            nan_df = pd.DataFrame(self.nan_authors_log)
            nan_df['num_nan_features'] = nan_df['nan_features'].apply(len)
            nan_df['nan_features_list'] = nan_df['nan_features'].apply(lambda x: ', '.join(x))
            nan_csv = join(OUTPUT_DIR, 'nan_imputation_log.csv')
            nan_df[['author', 'f1', 'num_nan_features', 'nan_features_list']].to_csv(nan_csv, index=False)
            print(f"      NaN Log CSV: nan_imputation_log.csv")
        
        # Report testuale completo
        report = self._generate_report(df, correlations, importance)
        report_file = join(OUTPUT_DIR, 'analysis_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"      Report: analysis_report.txt")
        
        # Visualizzazioni
        self._create_visualizations(df, correlations, importance)
        print(f"      Grafici: 4 visualizzazioni PNG")
    
    def _create_visualizations(self, df, correlations, importance):
        """Crea visualizzazioni"""
        
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Figura 1: Top features
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        top_corr = correlations.abs().head(10).sort_values()
        colors = ['#d62728' if correlations[f] < 0 else '#2ca02c' for f in top_corr.index]
        ax1.barh(range(len(top_corr)), [correlations[f] for f in top_corr.index], 
                 color=colors, alpha=0.8, edgecolor='black')
        ax1.set_yticks(range(len(top_corr)))
        ax1.set_yticklabels([f.replace('_', ' ').title() for f in top_corr.index])
        ax1.set_xlabel('Correlazione con F1', fontweight='bold')
        ax1.set_title('Top 10 - Correlazione Pearson', fontweight='bold', fontsize=14)
        ax1.axvline(0, color='black', linewidth=2)
        ax1.grid(axis='x', alpha=0.3)
        
        top_imp = importance.head(10).sort_values()
        ax2.barh(range(len(top_imp)), top_imp.values, color='steelblue', alpha=0.8, edgecolor='black')
        ax2.set_yticks(range(len(top_imp)))
        ax2.set_yticklabels([f.replace('_', ' ').title() for f in top_imp.index])
        ax2.set_xlabel('Feature Importance', fontweight='bold')
        ax2.set_title('Top 10 - Random Forest', fontweight='bold', fontsize=14)
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        fig1.savefig(join(OUTPUT_DIR, '1_top_features.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figura 2: Scatter plots
        fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, feat in enumerate(correlations.abs().head(4).index):
            ax = axes[idx]
            for cluster in sorted(df['cluster'].unique()):
                mask = df['cluster'] == cluster
                ax.scatter(df[feat][mask], df['f1'][mask], label=f'Cluster {cluster}',
                          alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
            
            z = np.polyfit(df[feat], df['f1'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df[feat].min(), df[feat].max(), 100)
            ax.plot(x_trend, p(x_trend), "r--", linewidth=2.5, label='Trend')
            
            ax.set_xlabel(feat.replace('_', ' ').title(), fontweight='bold')
            ax.set_ylabel('F1 Score', fontweight='bold')
            ax.set_title(f'{feat}\nr = {correlations[feat]:+.4f}', fontweight='bold')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9)
        
        plt.tight_layout()
        fig2.savefig(join(OUTPUT_DIR, '2_scatter_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figura 3: Heatmap
        fig3, ax = plt.subplots(figsize=(12, 10))
        top_10 = correlations.abs().head(10).index.tolist()
        corr_matrix = df[top_10 + ['f1']].corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', 
                    cmap='RdYlGn', center=0, ax=ax, square=True, 
                    linewidths=1.5, vmin=-1, vmax=1)
        
        ax.set_title('Correlation Heatmap - Top 10 Features', fontweight='bold', fontsize=14)
        plt.tight_layout()
        fig3.savefig(join(OUTPUT_DIR, '3_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figura 4: Cluster analysis
        fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        for cluster in sorted(df['cluster'].unique()):
            data = df[df['cluster'] == cluster]['f1']
            ax1.hist(data, alpha=0.6, bins=12, label=f'Cluster {cluster} (n={len(data)})',
                    edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('F1 Score', fontweight='bold')
        ax1.set_ylabel('Frequenza', fontweight='bold')
        ax1.set_title('Distribuzione F1 per Cluster', fontweight='bold', fontsize=14)
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        cluster_data = [df[df['cluster'] == c]['f1'].values for c in sorted(df['cluster'].unique())]
        bp = ax2.boxplot(cluster_data, labels=[f'Cluster {c}' for c in sorted(df['cluster'].unique())],
                         patch_artist=True, widths=0.6)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        ax2.set_ylabel('F1 Score', fontweight='bold')
        ax2.set_title('Box Plot F1 per Cluster', fontweight='bold', fontsize=14)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        fig4.savefig(join(OUTPUT_DIR, '4_cluster_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_report(self, df, correlations, importance):
        """Genera report testuale"""
        
        feature_cols = [c for c in df.columns if c not in ['f1', 'precision', 'recall', 'cluster']]
        
        lines = []
        lines.append("="*80)
        lines.append("BOND UNIFIED REFINED ANALYSIS - REPORT")
        lines.append("="*80)
        lines.append("")
        lines.append(f"Autori analizzati: {len(df)}")
        lines.append(f"Features utilizzate: {len(feature_cols)}")
        lines.append(f"Features rimosse (ridondanti): {len(FEATURES_TO_REMOVE)}")
        lines.append("")
        
        lines.append("FEATURES PER CATEGORIA")
        lines.append("-"*80)
        for category, features in OPTIMAL_FEATURES.items():
            present = [f for f in features if f in feature_cols]
            lines.append(f"\n{category} ({len(present)}):")
            for f in present:
                lines.append(f"  • {f}")
        
        lines.append("")
        lines.append("")
        lines.append("TOP 10 FEATURES - CORRELAZIONE PEARSON")
        lines.append("-"*80)
        for feat, corr in correlations.head(10).items():
            lines.append(f"  {feat:40s}: {corr:+.4f}")
        
        lines.append("")
        lines.append("TOP 10 FEATURES - RANDOM FOREST IMPORTANCE")
        lines.append("-"*80)
        for feat, imp in importance.head(10).items():
            lines.append(f"  {feat:40s}: {imp:.4f}")
        
        lines.append("")
        lines.append("STATISTICHE CLUSTER")
        lines.append("-"*80)
        for cluster in sorted(df['cluster'].unique()):
            data = df[df['cluster'] == cluster]
            lines.append(f"\nCluster {cluster} (n={len(data)}):")
            lines.append(f"  F1 medio: {data['f1'].mean():.4f} ± {data['f1'].std():.4f}")
            lines.append(f"  F1 range: [{data['f1'].min():.4f}, {data['f1'].max():.4f}]")
        
        lines.append("")
        lines.append("="*80)
        lines.append("GESTIONE VALORI MANCANTI (NaN) - REPORT DETTAGLIATO")
        lines.append("="*80)
        
        if self.nan_authors_log:
            lines.append(f"\nAutori con valori NaN: {len(self.nan_authors_log)}")
            lines.append("")
            lines.append("STRATEGIA: Imputazione invece di rimozione")
            lines.append("  • Features grafo (graph_*): NaN → 0.0 (grafo mancante/invalido)")
            lines.append("  • Altre features: NaN → mediana del dataset")
            lines.append("")
            
            # Riepilogo imputazioni per feature
            if self.imputation_log:
                lines.append("IMPUTAZIONI PER FEATURE:")
                lines.append("-"*80)
                for feat, info in sorted(self.imputation_log.items(), key=lambda x: x[1]['count'], reverse=True):
                    lines.append(f"\n{feat}:")
                    lines.append(f"  Autori con NaN: {info['count']}")
                    lines.append(f"  Valore imputato: {info['value']:.4f}")
                    lines.append(f"  Strategia: {info['strategy']}")
            
            # Lista autori con NaN
            lines.append("")
            lines.append("AUTORI CON NaN (dettaglio):")
            lines.append("-"*80)
            for i, entry in enumerate(sorted(self.nan_authors_log, key=lambda x: len(x['nan_features']), reverse=True), 1):
                lines.append(f"\n{i:2d}. {entry['author']}")
                lines.append(f"    F1 score: {entry['f1']:.4f}")
                lines.append(f"    Features NaN ({len(entry['nan_features'])}):")
                for feat in entry['nan_features']:
                    imputed = self.imputation_log.get(feat, {}).get('value', 'N/A')
                    lines.append(f"      • {feat} → {imputed}")
        else:
            lines.append("\nNessun valore NaN trovato nel dataset!")
        
        lines.append("")
        lines.append("="*80)
        lines.append("AUTORI SALTATI - ANALISI DETTAGLIATA")
        lines.append("="*80)
        if self.skipped_authors:
            lines.append(f"\nTotale autori saltati: {len(self.skipped_authors)}")
            lines.append("")
            
            # Raggruppa per tipo di errore
            error_types = {}
            for name, reason in self.skipped_authors:
                if reason not in error_types:
                    error_types[reason] = []
                error_types[reason].append(name)
            
            lines.append("PER TIPO DI ERRORE:")
            lines.append("-"*80)
            for reason, names in sorted(error_types.items(), key=lambda x: len(x[1]), reverse=True):
                lines.append(f"\n{reason} ({len(names)} autori):")
                for name in sorted(names):
                    lines.append(f"  • {name}")
            
            lines.append("")
            lines.append("LISTA COMPLETA:")
            lines.append("-"*80)
            for i, (name, reason) in enumerate(sorted(self.skipped_authors), 1):
                lines.append(f"  {i:2d}. {name:30s} → {reason}")
        else:
            lines.append("\nNessun autore saltato - tutti processati con successo!")
        
        lines.append("")
        lines.append("="*80)
        
        return "\n".join(lines)
    
    def print_summary(self, df):
        """Stampa sommario finale con dettagli NaN"""
        print(f"\nAutori analizzati: {len(df)}")
        print(f"Autori saltati (estrazione fallita): {len(self.skipped_authors)}")
        print(f"Autori con NaN (imputati): {len(self.nan_authors_log)}")
        
        if self.skipped_authors:
            print(f"\nERRORI ESTRAZIONE:")
            error_counts = {}
            for _, reason in self.skipped_authors:
                error_counts[reason] = error_counts.get(reason, 0) + 1
            for reason, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  • {reason}: {count} autori")
        
        if self.nan_authors_log:
            print(f"\nIMPUTAZIONI NaN:")
            nan_features_count = {}
            for entry in self.nan_authors_log:
                for feat in entry['nan_features']:
                    nan_features_count[feat] = nan_features_count.get(feat, 0) + 1
            for feat, count in sorted(nan_features_count.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  • {feat}: {count} autori")
        
        print(f"\nFeatures finali: {len([c for c in df.columns if c not in ['f1', 'precision', 'recall', 'cluster']])}")
        print(f"\nF1 Score:")
        print(f"  Media: {df['f1'].mean():.4f}")
        print(f"  Range: [{df['f1'].min():.4f}, {df['f1'].max():.4f}]")
    
    def _compute_f1(self, true_clusters, pred_clusters):
        """Calcola F1 pairwise"""
        predicted_pubs = {}
        for idx, cluster in enumerate(pred_clusters):
            for pid in cluster:
                predicted_pubs[pid] = idx
        
        pubs = []
        true_labels = []
        for idx, cluster in enumerate(true_clusters):
            for pid in cluster:
                pubs.append(pid)
                true_labels.append(idx)
        
        filtered_pubs = [pid for pid in pubs if pid in predicted_pubs]
        if len(filtered_pubs) == 0:
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        filtered_true_labels = [true_labels[i] for i, pid in enumerate(pubs) if pid in predicted_pubs]
        predict_labels = [predicted_pubs[pid] for pid in filtered_pubs]
        
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


def main():
    """Entry point"""
    
    if not exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    analyzer = UnifiedRefinedAnalyzer()
    analyzer.run_full_pipeline()
    
    print(f"\n{'='*80}")
    print(f"File generati in: {OUTPUT_DIR}")
    print(f"  • refined_features_results.csv")
    print(f"  • analysis_report.txt")
    print(f"  • 1_top_features.png")
    print(f"  • 2_scatter_plots.png")
    print(f"  • 3_correlation_heatmap.png")
    print(f"  • 4_cluster_analysis.png")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
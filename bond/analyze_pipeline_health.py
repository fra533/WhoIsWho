"""
Analizza la salute di tutta la pipeline BOND
Controlla embeddings, grafi, e dà raccomandazioni
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import pickle

from params import set_params

args = set_params()


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def analyze_pipeline_health(mode='valid'):
    """
    Analizza ogni componente della pipeline
    """
    print("="*70)
    print("🏥 BOND PIPELINE HEALTH CHECK")
    print("="*70)
    print(f"Mode: {mode}\n")
    
    base_path = Path(args.save_path)
    
    # ===== 1. CHECK FILES =====
    print("\n" + "="*70)
    print("📁 FILE EXISTENCE CHECK")
    print("="*70)
    
    checks = {
        'Ground Truth': base_path / 'src' / f'sna-{mode}' / f'sna_{mode}_ground_truth.json',
        'Publications': base_path / 'src' / f'sna-{mode}' / f'sna_{mode}_pub.json',
        'Raw Pubs': base_path / 'src' / f'sna-{mode}' / f'sna_{mode}_raw.json',
        'W2V Model': base_path / 'w2v_model' / 'w2v_256.model',
        'Names Pub Dir': base_path / 'names_pub' / mode,
        'Relations Dir': base_path / 'relations' / mode,
        'Graph Dir': base_path / 'graph' / mode,
        'Paper Emb Dir': base_path / 'paper_emb' / mode
    }
    
    missing_critical = []
    
    for name, path in checks.items():
        exists = path.exists()
        symbol = "✅" if exists else "❌"
        print(f"  {symbol} {name:20} {path}")
        
        if not exists and name in ['Ground Truth', 'W2V Model', 'Paper Emb Dir', 'Graph Dir']:
            missing_critical.append(name)
    
    if missing_critical:
        print(f"\n  ⚠️  Missing critical components: {', '.join(missing_critical)}")
        print(f"     Pipeline cannot run without these!")
        return False
    
    # ===== 2. GROUND TRUTH STATS =====
    print("\n" + "="*70)
    print("📊 GROUND TRUTH STATISTICS")
    print("="*70)
    
    gt_file = base_path / 'src' / f'sna-{mode}' / f'sna_{mode}_ground_truth.json'
    if gt_file.exists():
        with open(gt_file, encoding='utf-8') as f:
            gt = json.load(f)
        
        total_authors = len(gt)
        
        papers_per_cluster = []
        clusters_per_author = []
        
        for author_name, author_clusters in gt.items():
            n_clusters = len(author_clusters)
            clusters_per_author.append(n_clusters)
            
            for cluster_papers in author_clusters.values():
                papers_per_cluster.append(len(cluster_papers))
        
        total_papers = sum(papers_per_cluster)
        total_clusters = len(papers_per_cluster)
        
        print(f"  Authors: {total_authors}")
        print(f"  Total papers: {total_papers}")
        print(f"  Total clusters: {total_clusters}")
        print(f"  Avg papers per cluster: {np.mean(papers_per_cluster):.1f}")
        print(f"  Avg clusters per author: {np.mean(clusters_per_author):.1f}")
        print(f"  Papers per author: min={min([sum(len(p) for p in a.values()) for a in gt.values()])}, "
              f"max={max([sum(len(p) for p in a.values()) for a in gt.values()])}")
    
    # ===== 3. WORD2VEC EMBEDDINGS =====
    print("\n" + "="*70)
    print("📖 WORD2VEC EMBEDDINGS HEALTH")
    print("="*70)
    
    paper_emb_dir = base_path / 'paper_emb' / mode
    if paper_emb_dir.exists():
        authors = list(paper_emb_dir.iterdir())[:5]  # Sample 5
        
        empty_counts = []
        similarity_means = []
        
        for author_dir in authors:
            emb_file = author_dir / 'ptext_emb.pkl'
            tcp_file = author_dir / 'tcp.pkl'
            
            if emb_file.exists() and tcp_file.exists():
                ptext_emb = load_pkl(emb_file)
                tcp = load_pkl(tcp_file)
                
                if len(ptext_emb) > 1:
                    emb_matrix = np.array(list(ptext_emb.values()))
                    sims = cosine_similarity(emb_matrix)
                    np.fill_diagonal(sims, 0)
                    
                    empty_counts.append(len(tcp))
                    similarity_means.append(sims.mean())
        
        if empty_counts:
            print(f"  Avg empty embeddings per author: {np.mean(empty_counts):.1f}")
            print(f"  Avg embedding similarity: {np.mean(similarity_means):.4f}")
            print(f"  Similarity std: {np.std(similarity_means):.4f}")
            
            if np.mean(similarity_means) < 0.2:
                print(f"  ⚠️  WARNING: Very low similarity ({np.mean(similarity_means):.4f})")
                print(f"     Embeddings may be too sparse!")
            elif np.mean(similarity_means) > 0.7:
                print(f"  ⚠️  WARNING: Very high similarity ({np.mean(similarity_means):.4f})")
                print(f"     Embeddings may not discriminate well!")
        
        print(f"\n  For detailed W2V analysis, run:")
        print(f"    python analyze_w2v_embeddings.py --mode {mode} --full")
    
    # ===== 4. GRAPH STATS =====
    print("\n" + "="*70)
    print("🕸️  GRAPH STATISTICS")
    print("="*70)

    graph_dir = base_path / 'graph' / mode
    if graph_dir.exists():
        all_authors = list(graph_dir.iterdir())
        
        # ===== FILTRA SOLO AUTORI IN GROUND TRUTH =====
        authors_to_analyze = all_authors  # Default: tutti
        
        if mode in ['valid', 'test']:
            gt_file = base_path / 'src' / f'sna-{mode}' / f'sna_{mode}_ground_truth.json'
            if gt_file.exists():
                with open(gt_file, encoding='utf-8') as f:
                    gt = json.load(f)
                gt_names = set(gt.keys())
                
                # Filtra solo autori in GT
                authors_to_analyze = [a for a in all_authors if a.name in gt_names]
                
                print(f"  Total authors in graph dir: {len(all_authors)}")
                print(f"  Authors in ground truth: {len(gt_names)}")
                print(f"  Analyzing (filtered to GT): {len(authors_to_analyze)}")
                
                if len(authors_to_analyze) < len(gt_names):
                    missing = len(gt_names) - len(authors_to_analyze)
                    print(f"  ⚠️  {missing} GT authors missing from graphs!")
            else:
                print(f"  No ground truth found - analyzing all {len(all_authors)} authors")
        else:
            print(f"  Analyzing all {len(authors_to_analyze)} authors (train mode)")
        # ==============================================
        
        node_counts = []
        edge_counts = []
        empty_graphs = 0
        citation_coverage = {'out': 0, 'in': 0, 'total': 0}
        
        print(f"  Processing {len(authors_to_analyze)} authors...")
        
        # ===== LOOP OTTIMIZZATO =====
        for author_dir in authors_to_analyze:
            # Conta nodi da pids.txt (veloce)
            pids_file = author_dir / 'pids.txt'
            if pids_file.exists():
                with open(pids_file, 'r', encoding='utf-8') as f:
                    n_nodes = sum(1 for _ in f)
                    node_counts.append(n_nodes)
            
            # Conta edge e analizza citations in un passaggio
            adj_file = author_dir / 'adj_attr.txt'
            if adj_file.exists():
                n_edges = 0
                sample_count = 0
                max_sample = 100
                
                with open(adj_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        n_edges += 1
                        
                        # Sample solo prime 100 per citations
                        if sample_count < max_sample:
                            parts = line.strip().split('\t')
                            if len(parts) >= 11:
                                citation_coverage['total'] += 1
                                
                                # Check veloce senza float()
                                if parts[8] not in ('0', '0.0', '0.0000'):
                                    citation_coverage['out'] += 1
                                if parts[10] not in ('0', '0.0', '0.0000'):
                                    citation_coverage['in'] += 1
                            sample_count += 1
                
                edge_counts.append(n_edges)
                
                if n_edges == 0:
                    empty_graphs += 1
        # ============================
        
        # ===== STATISTICHE =====
        if len(authors_to_analyze) > 0:
            print(f"\n  Empty graphs: {empty_graphs}/{len(authors_to_analyze)} ({empty_graphs/len(authors_to_analyze)*100:.1f}%)")
        
        if node_counts:
            print(f"\n  Node statistics:")
            print(f"    Min: {min(node_counts)}, Max: {max(node_counts)}")
            print(f"    Mean: {np.mean(node_counts):.1f}, Median: {np.median(node_counts):.1f}")
        
        if edge_counts:
            print(f"\n  Edge statistics:")
            print(f"    Min: {min(edge_counts)}, Max: {max(edge_counts)}")
            print(f"    Mean: {np.mean(edge_counts):.1f}, Median: {np.median(edge_counts):.1f}")
        
        if citation_coverage['total'] > 0:
            print(f"\n  Citation coverage (sample of {citation_coverage['total']} edges):")
            print(f"    Edges with cite_out > 0: {citation_coverage['out']/citation_coverage['total']*100:.1f}%")
            print(f"    Edges with cite_in > 0: {citation_coverage['in']/citation_coverage['total']*100:.1f}%")
        else:
            print(f"\n  ⚠️  No citation data found in graphs!")
    
    # ===== 5. RECOMMENDATIONS =====
    print("\n" + "="*70)
    print("💡 RECOMMENDATIONS")
    print("="*70)
    
    issues_found = False
    
    if empty_graphs / len(authors) > 0.5:
        print("  ⚠️  >50% empty graphs")
        print("     → Many authors have no relational connections")
        print("     → Consider lowering graph construction thresholds")
        issues_found = True
    
    if similarity_means and np.mean(similarity_means) < 0.3:
        print("  ⚠️  Very low W2V similarity (<0.3)")
        print("     → Embeddings are too dissimilar")
        print("     → DBSCAN will struggle to form clusters")
        print("     → Use higher db_eps (0.3-0.5) or review text preprocessing")
        issues_found = True
    
    if edge_counts and np.mean(edge_counts) < 10:
        print("  ⚠️  Very few edges (mean < 10)")
        print("     → Papers have few connections")
        print("     → GNN may not learn effectively")
        issues_found = True
    
    if not issues_found:
        print("  ✓ No major issues detected!")
        print("  ✓ Pipeline should run normally")
    
    print("\n" + "="*70)
    print("✓ Health check complete!")
    print("="*70)
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Check BOND pipeline health')
    parser.add_argument('--mode', type=str, default='valid', choices=['train', 'valid', 'test'],
                       help='Dataset mode to check')
    
    args_script = parser.parse_args()
    
    analyze_pipeline_health(mode=args_script.mode)
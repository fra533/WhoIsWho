"""
CLUSTER ACCEPTANCE EVALUATION - Basato su Kim (2019)

Questo script valuta i cluster predetti per decidere quali accettare.
Implementa le metriche di Kim (2019) con focus su:
1. Identificare cluster "sicuri" da accettare
2. Identificare cluster "dubbi" da rivedere
3. Fornire confidence score per ogni autore

RIFERIMENTO:
Kim, J. (2019). A fast and integrative algorithm for clustering performance 
evaluation in author name disambiguation. Scientometrics, 120, 661-681.
"""

import json
import numpy as np
from collections import defaultdict

# ======================== PATHS ========================
PRED_FILE = r"C:\Users\franc\OneDrive - Alma Mater Studiorum Università di Bologna\Desktop\BOND-OC\WhoIsWho\bond\out\res.json"
GT_FILE = r"C:\Users\franc\OneDrive - Alma Mater Studiorum Università di Bologna\Desktop\BOND-OC\WhoIsWho\bond\dataset\data\src\sna-valid\sna_valid_ground_truth.json"
    

# Thresholds per accettazione cluster
CONFIDENCE_THRESHOLDS = {
    'HIGH': 0.90,      # Cluster da accettare automaticamente
    'MEDIUM': 0.70,    # Cluster da rivedere
    'LOW': 0.50        # Cluster probabilmente da rigettare
}
# =======================================================


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_clusters(data):
    """Estrae cluster in formato uniforme (lista di set)"""
    clusters = []
    
    if isinstance(data, dict):
        for papers in data.values():
            if isinstance(papers, list):
                clusters.append(set(papers))
            else:
                clusters.append({papers})
    elif isinstance(data, list):
        for cluster in data:
            if isinstance(cluster, list):
                clusters.append(set(cluster))
            else:
                clusters.append({cluster})
    
    return clusters


def get_all_papers(clusters):
    papers = set()
    for cluster in clusters:
        papers.update(cluster)
    return papers


def compute_per_author_metrics(true_clusters, pred_clusters, author_name):
    """
    Calcola metriche per singolo autore seguendo Kim (2019).
    
    Returns:
        dict con metriche + confidence score + recommendation
    """
    metrics = {}
    
    # 1. B³ Precision/Recall (equivalente a K-metric AAP/ACP)
    all_papers = get_all_papers(true_clusters)
    N = len(all_papers)
    
    if N == 0:
        return None
    
    # B³ Recall (AAP nel paper)
    b3_recall_sum = 0
    for Tj in true_clusters:
        for Pi in pred_clusters:
            intersection = Pi & Tj
            if len(intersection) > 0:
                b3_recall_sum += len(intersection)**2 / len(Tj)
    
    metrics['b3_recall'] = b3_recall_sum / N if N > 0 else 0
    
    # B³ Precision (ACP nel paper)
    b3_precision_sum = 0
    for Tj in true_clusters:
        for Pi in pred_clusters:
            intersection = Pi & Tj
            if len(intersection) > 0:
                b3_precision_sum += len(intersection)**2 / len(Pi)
    
    metrics['b3_precision'] = b3_precision_sum / N if N > 0 else 0
    
    # B³ F1
    if metrics['b3_recall'] + metrics['b3_precision'] > 0:
        metrics['b3_f1'] = 2 * metrics['b3_recall'] * metrics['b3_precision'] / \
                          (metrics['b3_recall'] + metrics['b3_precision'])
    else:
        metrics['b3_f1'] = 0.0
    
    # 2. Cluster Purity & Inverse Purity
    # Cluster Purity: quanto sono "puri" i cluster predetti
    purity_sum = 0
    total_pred_papers = 0
    for Pi in pred_clusters:
        max_overlap = max([len(Pi & Tj) for Tj in true_clusters], default=0)
        purity_sum += max_overlap
        total_pred_papers += len(Pi)
    
    metrics['cluster_purity'] = purity_sum / total_pred_papers if total_pred_papers > 0 else 0
    
    # Inverse Purity: quanto sono "completi" i cluster predetti
    inv_purity_sum = 0
    total_true_papers = 0
    for Tj in true_clusters:
        max_overlap = max([len(Tj & Pi) for Pi in pred_clusters], default=0)
        inv_purity_sum += max_overlap
        total_true_papers += len(Tj)
    
    metrics['inverse_purity'] = inv_purity_sum / total_true_papers if total_true_papers > 0 else 0
    
    # 3. Splitting/Lumping Analysis
    num_true_clusters = len(true_clusters)
    num_pred_clusters = len(pred_clusters)
    
    metrics['num_true_clusters'] = num_true_clusters
    metrics['num_pred_clusters'] = num_pred_clusters
    metrics['is_over_clustered'] = num_pred_clusters > num_true_clusters
    metrics['is_under_clustered'] = num_pred_clusters < num_true_clusters
    metrics['is_perfect_match'] = num_pred_clusters == num_true_clusters
    
    # 4. Splitting Error Rate (per questo autore)
    if N > 1:
        # Per autori con più paper, calcola splitting
        splitting_pairs = 0
        total_true_pairs = 0
        
        # Crea mapping paper -> predicted cluster
        pred_map = {}
        for idx, Pi in enumerate(pred_clusters):
            for paper in Pi:
                pred_map[paper] = idx
        
        # Conta splitting errors
        for Tj in true_clusters:
            papers_list = list(Tj)
            for i in range(len(papers_list)):
                for j in range(i+1, len(papers_list)):
                    p1, p2 = papers_list[i], papers_list[j]
                    total_true_pairs += 1
                    
                    if p1 in pred_map and p2 in pred_map:
                        if pred_map[p1] != pred_map[p2]:
                            splitting_pairs += 1
        
        metrics['splitting_rate'] = splitting_pairs / total_true_pairs if total_true_pairs > 0 else 0
    else:
        metrics['splitting_rate'] = 0.0  # Un solo paper, non può essere splittato
    
    # 5. CONFIDENCE SCORE (pesato sulle metriche più robuste)
    # Basato su Kim (2019): B³ è più robusto di Pairwise-F per cluster piccoli
    confidence = (
        0.40 * metrics['b3_f1'] +           # Metrica principale
        0.30 * metrics['cluster_purity'] +   # Evita lumping
        0.20 * metrics['inverse_purity'] +   # Evita splitting
        0.10 * (1 - metrics['splitting_rate'])  # Penalizza splitting
    )
    
    metrics['confidence_score'] = confidence
    
    # 6. RECOMMENDATION
    if confidence >= CONFIDENCE_THRESHOLDS['HIGH']:
        recommendation = 'ACCEPT'
        reason = 'High confidence clustering'
    elif confidence >= CONFIDENCE_THRESHOLDS['MEDIUM']:
        if metrics['cluster_purity'] >= 0.95:
            recommendation = 'ACCEPT_WITH_CAUTION'
            reason = 'Good purity but some splitting'
        else:
            recommendation = 'REVIEW'
            reason = 'Medium confidence - manual check recommended'
    else:
        if metrics['splitting_rate'] > 0.5:
            recommendation = 'REJECT'
            reason = 'Excessive splitting detected'
        else:
            recommendation = 'REVIEW'
            reason = 'Low confidence clustering'
    
    metrics['recommendation'] = recommendation
    metrics['reason'] = reason
    
    return metrics


def evaluate_for_cluster_acceptance(pred_file, gt_file):
    """
    Valutazione completa per decisione di accettazione cluster.
    """
    print("="*80)
    print("CLUSTER ACCEPTANCE EVALUATION - Kim (2019) Framework")
    print("="*80)
    
    # Carica dati
    print(f"\n📂 Caricamento dati...")
    pred_data = load_json(pred_file)
    gt_data = load_json(gt_file)
    
    common_authors = set(pred_data.keys()) & set(gt_data.keys())
    print(f"   ✓ Autori da valutare: {len(common_authors)}")
    
    # Valuta ogni autore
    results = []
    recommendations_count = defaultdict(int)
    
    print(f"\n🔍 Valutazione autori in corso...")
    
    for author in common_authors:
        true_clusters = extract_clusters(gt_data[author])
        pred_clusters = extract_clusters(pred_data[author])
        
        metrics = compute_per_author_metrics(true_clusters, pred_clusters, author)
        
        if metrics:
            metrics['author'] = author
            metrics['num_papers'] = len(get_all_papers(true_clusters))
            results.append(metrics)
            recommendations_count[metrics['recommendation']] += 1
    
    # Ordina per confidence score
    results.sort(key=lambda x: x['confidence_score'], reverse=True)
    
    # ============ REPORT ============
    print(f"\n{'='*80}")
    print("📊 RISULTATI VALUTAZIONE")
    print(f"{'='*80}")
    
    print(f"\n🎯 Distribuzione raccomandazioni:")
    total = len(results)
    for rec, count in sorted(recommendations_count.items()):
        pct = (count/total*100) if total > 0 else 0
        icon = {'ACCEPT': '✅', 'ACCEPT_WITH_CAUTION': '⚠️', 'REVIEW': '🔍', 'REJECT': '❌'}.get(rec, '•')
        print(f"   {icon} {rec}: {count}/{total} ({pct:.1f}%)")
    
    # Statistiche metriche
    print(f"\n📈 Statistiche metriche (media):")
    avg_b3_f1 = np.mean([r['b3_f1'] for r in results])
    avg_purity = np.mean([r['cluster_purity'] for r in results])
    avg_inv_purity = np.mean([r['inverse_purity'] for r in results])
    avg_splitting = np.mean([r['splitting_rate'] for r in results])
    avg_confidence = np.mean([r['confidence_score'] for r in results])
    
    print(f"   B³ F1:            {avg_b3_f1:.4f}")
    print(f"   Cluster Purity:   {avg_purity:.4f}")
    print(f"   Inverse Purity:   {avg_inv_purity:.4f}")
    print(f"   Splitting Rate:   {avg_splitting:.4f}")
    print(f"   Confidence Score: {avg_confidence:.4f}")
    
    # Over/under clustering
    over = sum(1 for r in results if r['is_over_clustered'])
    under = sum(1 for r in results if r['is_under_clustered'])
    perfect = sum(1 for r in results if r['is_perfect_match'])
    
    print(f"\n🔢 Analisi clustering:")
    print(f"   Over-clustered:  {over}/{total} ({over/total*100:.1f}%)")
    print(f"   Under-clustered: {under}/{total} ({under/total*100:.1f}%)")
    print(f"   Perfect match:   {perfect}/{total} ({perfect/total*100:.1f}%)")
    
    # Top 10 ACCEPT
    print(f"\n{'='*80}")
    print("✅ TOP 10 AUTORI DA ACCETTARE (confidence più alta)")
    print(f"{'='*80}")
    
    accepts = [r for r in results if r['recommendation'] == 'ACCEPT'][:10]
    for i, r in enumerate(accepts, 1):
        print(f"\n[{i}] {r['author']}")
        print(f"    Confidence: {r['confidence_score']:.4f}")
        print(f"    B³ F1: {r['b3_f1']:.4f} | Purity: {r['cluster_purity']:.4f} | Splitting: {r['splitting_rate']:.4f}")
        print(f"    Clusters: True={r['num_true_clusters']}, Pred={r['num_pred_clusters']}, Papers={r['num_papers']}")
    
    # Top 10 PROBLEMATICI
    print(f"\n{'='*80}")
    print("❌ TOP 10 AUTORI PROBLEMATICI (confidence più bassa)")
    print(f"{'='*80}")
    
    problematic = [r for r in results if r['recommendation'] in ['REJECT', 'REVIEW']][-10:]
    for i, r in enumerate(problematic, 1):
        print(f"\n[{i}] {r['author']}")
        print(f"    Recommendation: {r['recommendation']} - {r['reason']}")
        print(f"    Confidence: {r['confidence_score']:.4f}")
        print(f"    B³ F1: {r['b3_f1']:.4f} | Purity: {r['cluster_purity']:.4f} | Splitting: {r['splitting_rate']:.4f}")
        print(f"    Clusters: True={r['num_true_clusters']}, Pred={r['num_pred_clusters']}, Papers={r['num_papers']}")
    
    # Salva risultati dettagliati
    output_file = PRED_FILE.replace('.json', '_evaluation_report.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"💾 Report dettagliato salvato in:")
    print(f"   {output_file}")
    print(f"{'='*80}")
    
    return results, recommendations_count


if __name__ == "__main__":
    try:
        results, recommendations = evaluate_for_cluster_acceptance(PRED_FILE, GT_FILE)
        
        print("\n💡 INTERPRETAZIONE:")
        print("   • ACCEPT: Cluster sicuri, confidence ≥90%")
        print("   • ACCEPT_WITH_CAUTION: Buona purity ma possibile splitting")
        print("   • REVIEW: Richiedono verifica manuale")
        print("   • REJECT: Troppo splitting, meglio rifare")
        print("\n   Basato su Kim (2019): B³ metrics più robusto di Pairwise-F")
        print("   per dataset con pochi paper per autore.")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERRORE: File non trovato - {e}")
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()
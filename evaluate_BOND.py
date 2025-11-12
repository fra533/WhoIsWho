"""
Script di valutazione multi-metrica per Author Name Disambiguation
Basato su Kim (2019) "A fast and integrative algorithm for clustering performance 
evaluation in author name disambiguation"

CALCOLA:
1. Pairwise-F (precision, recall, F1)
2. K-metric (AAP, ACP, K)
3. B³ (equivalente a K-metric, ma espresso come B³)
4. Cluster-F (precision, recall, F1)
5. Splitting & Lumping Error (SE, LE)

ANALIZZA:
- Casi triviali (1 paper) vs difficili (multipli papers)
- Distribuzione errori per-name
- Identificazione nomi problematici
"""
import numpy as np
import os
import json
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

def load_json(path):
    """Carica file JSON"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    """Salva file JSON"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


class MultiMetricEvaluator:
    """
    Valutatore multi-metrica completo secondo Kim (2019)
    
    Usa il framework integrato (Algorithm 6) per calcolare
    tutte le metriche in un singolo passaggio.
    """
    
    def __init__(self, predictions_file, ground_truth_file, output_dir=None):
        self.predictions_file = predictions_file
        self.ground_truth_file = ground_truth_file
        self.output_dir = Path(output_dir) if output_dir else None
        
        # Risultati
        self.results = {}
        self.name_level_results = {}
        
        # Statistiche
        self.stats = {
            'total_names': 0,
            'trivial_names': 0,
            'difficult_names': 0,
            'perfect_matches': 0,
            'total_instances': 0
        }
    
    def evaluate_all_metrics(self):
        """
        Calcola tutte le metriche usando il framework integrato
        """
        try:
            # Carica dati
            print(f"\nLoading data...")
            print(f"  Predictions: {self.predictions_file}")
            print(f"  Ground truth: {self.ground_truth_file}")
            
            predict_result = load_json(self.predictions_file)
            ground_truth = load_json(self.ground_truth_file)
            
            print(f"\nData loaded:")
            print(f"  Names in predictions: {len(predict_result)}")
            print(f"  Names in ground truth: {len(ground_truth)}")
            
            # Filtra solo nomi comuni
            filtered_predict = {n: p for n, p in predict_result.items() 
                               if n in ground_truth}
            
            print(f"  Common names: {len(filtered_predict)}")
            
            if not filtered_predict:
                print("\n❌ ERROR: No common names between predictions and ground truth!")
                return self._empty_results()
            
            # Inizializza accumulatori globali
            global_metrics = {
                'pairwise': {'tp': 0, 'fp': 0, 'fn': 0},
                'k_metric': {'aap_sum': 0, 'acp_sum': 0, 'n_instances': 0},
                'cluster': {'matches': 0, 'n_truth': 0, 'n_pred': 0},
                'split_lump': {
                    'split_sum': 0, 'split_total': 0,
                    'lump_sum': 0, 'lump_total': 0
                }
            }
            
            # Valuta nome per nome
            print(f"\nEvaluating {len(filtered_predict)} names...")
            
            for name in filtered_predict:
                name_metrics = self._evaluate_single_name(
                    name, 
                    filtered_predict[name], 
                    ground_truth[name]
                )
                
                # Accumula
                self._accumulate_metrics(global_metrics, name_metrics)
                
                # Salva per analisi
                self.name_level_results[name] = name_metrics
                
                # Aggiorna statistiche
                self.stats['total_instances'] += name_metrics['n_instances']
                if name_metrics['is_trivial']:
                    self.stats['trivial_names'] += 1
                else:
                    self.stats['difficult_names'] += 1
                if name_metrics['cluster']['match'] == 1:
                    self.stats['perfect_matches'] += 1
            
            self.stats['total_names'] = len(filtered_predict)
            
            # Calcola metriche finali
            self.results = self._compute_final_metrics(global_metrics)
            
            # Aggiungi statistiche name-level
            self.results['name_level_stats'] = self._compute_name_level_stats()
            
            # Calcola composite score
            self.results['composite_score'] = self._compute_composite_score()
            
            return self.results
            
        except Exception as e:
            print(f"\n❌ ERROR in evaluation: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_results()
    
    def _evaluate_single_name(self, name, predicted_clusters, truth_data):
        """
        Valuta un singolo nome con tutte le metriche
        Implementa Algorithm 6 del paper Kim (2019)
        """
        # Prepara truth clusters
        truth_clusters = self._prepare_truth_clusters(truth_data)
        
        if not truth_clusters or not predicted_clusters:
            return self._empty_name_metrics()
        
        # pIndex: mapping instance -> predicted_cluster_id
        pIndex = {}
        for pred_idx, pred_cluster in enumerate(predicted_clusters):
            for instance in pred_cluster:
                pIndex[instance] = pred_idx
        
        # Statistiche cluster
        n_truth_clusters = len(truth_clusters)
        n_pred_clusters = len(predicted_clusters)
        n_instances = sum(len(tc) for tc in truth_clusters)
        
        # Dimensioni cluster predetti
        pred_sizes = [len(pc) for pc in predicted_clusters]
        
        # Inizializza metriche
        metrics = {
            'n_instances': n_instances,
            'n_truth_clusters': n_truth_clusters,
            'n_pred_clusters': n_pred_clusters,
            'is_trivial': (n_truth_clusters == 1 and n_instances == 1),
            'pairwise': {'tp': 0, 'fp': 0, 'fn': 0},
            'k_metric': {'aap_sum': 0, 'acp_sum': 0},
            'cluster': {'match': 0},
            'split_lump': {'split': 0, 'lump': 0}
        }
        
        # ====== LOOP SUI TRUTH CLUSTERS ======
        for truth_cluster in truth_clusters:
            # tMap: predicted_cluster_id -> overlap_size
            tMap = {}
            
            for instance in truth_cluster:
                if instance in pIndex:
                    pred_idx = pIndex[instance]
                    tMap[pred_idx] = tMap.get(pred_idx, 0) + 1
            
            truth_size = len(truth_cluster)
            
            # Trova predicted cluster con massimo overlap
            max_overlap = 0
            max_pred_idx = -1
            for pred_idx, overlap_size in tMap.items():
                if overlap_size > max_overlap:
                    max_overlap = overlap_size
                    max_pred_idx = pred_idx
            
            # ====== CLUSTER-F ======
            if max_pred_idx >= 0:
                if (max_overlap == truth_size and 
                    pred_sizes[max_pred_idx] == truth_size):
                    metrics['cluster']['match'] = 1
            
            # ====== K-METRIC ======
            for pred_idx, overlap_size in tMap.items():
                # AAP: |Pi ∩ Tj|² / |Tj|
                metrics['k_metric']['aap_sum'] += (overlap_size ** 2) / truth_size
                # ACP: |Pi ∩ Tj|² / |Pi|
                metrics['k_metric']['acp_sum'] += (overlap_size ** 2) / pred_sizes[pred_idx]
            
            # ====== PAIRWISE-F ======
            # Truth pairs in questo cluster
            truth_pairs = truth_size * (truth_size - 1) // 2
            
            # Intersection pairs
            for pred_idx, overlap_size in tMap.items():
                intersection_pairs = overlap_size * (overlap_size - 1) // 2
                metrics['pairwise']['tp'] += intersection_pairs
            
            # False negatives
            metrics['pairwise']['fn'] += truth_pairs - sum(
                overlap * (overlap - 1) // 2 for overlap in tMap.values()
            )
            
            # ====== SPLITTING ERROR ======
            if max_pred_idx >= 0:
                metrics['split_lump']['split'] += (truth_size - max_overlap)
        
        # ====== FALSE POSITIVES (Pairwise-F) ======
        for pred_idx, pred_cluster in enumerate(predicted_clusters):
            pred_pairs = len(pred_cluster) * (len(pred_cluster) - 1) // 2
            
            # Conta coppie corrette
            correct_pairs = 0
            for truth_cluster in truth_clusters:
                overlap = sum(1 for inst in truth_cluster if pIndex.get(inst) == pred_idx)
                correct_pairs += overlap * (overlap - 1) // 2
            
            metrics['pairwise']['fp'] += (pred_pairs - correct_pairs)
        
        # ====== LUMPING ERROR ======
        for truth_cluster in truth_clusters:
            tMap = {}
            for instance in truth_cluster:
                if instance in pIndex:
                    pred_idx = pIndex[instance]
                    tMap[pred_idx] = tMap.get(pred_idx, 0) + 1
            
            if tMap:
                max_pred_idx = max(tMap.items(), key=lambda x: x[1])[0]
                pred_cluster_size = pred_sizes[max_pred_idx]
                truth_instances_in_pred = tMap[max_pred_idx]
                
                metrics['split_lump']['lump'] += (pred_cluster_size - truth_instances_in_pred)
        
        return metrics
    
    def _prepare_truth_clusters(self, truth_data):
        """Converte truth_data in lista di cluster"""
        clusters = []
        
        if isinstance(truth_data, dict):
            # Formato: {"author_id": ["paper1", "paper2"]}
            for author_id, instances in truth_data.items():
                if instances:
                    clusters.append(instances)
        elif isinstance(truth_data, list):
            # Formato: [["paper1", "paper2"], ["paper3", "paper4"]]
            for cluster in truth_data:
                if isinstance(cluster, list) and cluster:
                    clusters.append(cluster)
        
        return clusters
    
    def _accumulate_metrics(self, global_metrics, name_metrics):
        """Accumula metriche da un singolo nome"""
        # Pairwise
        global_metrics['pairwise']['tp'] += name_metrics['pairwise']['tp']
        global_metrics['pairwise']['fp'] += name_metrics['pairwise']['fp']
        global_metrics['pairwise']['fn'] += name_metrics['pairwise']['fn']
        
        # K-metric
        global_metrics['k_metric']['aap_sum'] += name_metrics['k_metric']['aap_sum']
        global_metrics['k_metric']['acp_sum'] += name_metrics['k_metric']['acp_sum']
        global_metrics['k_metric']['n_instances'] += name_metrics['n_instances']
        
        # Cluster-F
        global_metrics['cluster']['matches'] += name_metrics['cluster']['match']
        global_metrics['cluster']['n_truth'] += name_metrics['n_truth_clusters']
        global_metrics['cluster']['n_pred'] += name_metrics['n_pred_clusters']
        
        # Split & Lump
        global_metrics['split_lump']['split_sum'] += name_metrics['split_lump']['split']
        global_metrics['split_lump']['split_total'] += name_metrics['n_instances']
        global_metrics['split_lump']['lump_sum'] += name_metrics['split_lump']['lump']
        global_metrics['split_lump']['lump_total'] += name_metrics['n_instances']
    
    def _compute_final_metrics(self, global_metrics):
        """Calcola metriche finali da accumulatori"""
        results = {}
        
        # ====== PAIRWISE-F ======
        pw = global_metrics['pairwise']
        pw_recall = pw['tp'] / (pw['tp'] + pw['fn']) if (pw['tp'] + pw['fn']) > 0 else 0.0
        pw_precision = pw['tp'] / (pw['tp'] + pw['fp']) if (pw['tp'] + pw['fp']) > 0 else 0.0
        pw_f1 = (2 * pw_recall * pw_precision / (pw_recall + pw_precision)) if (pw_recall + pw_precision) > 0 else 0.0
        
        results['pairwise'] = {
            'recall': pw_recall,
            'precision': pw_precision,
            'f1': pw_f1
        }
        
        # ====== K-METRIC ======
        km = global_metrics['k_metric']
        n = km['n_instances']
        
        aap = km['aap_sum'] / n if n > 0 else 0.0
        acp = km['acp_sum'] / n if n > 0 else 0.0
        k = np.sqrt(aap * acp) if (aap * acp) > 0 else 0.0
        
        results['k_metric'] = {
            'aap': aap,
            'acp': acp,
            'k': k
        }
        
        # ====== B³ (uguale a K-metric) ======
        results['b3'] = {
            'recall': aap,
            'precision': acp,
            'f1': (2 * aap * acp / (aap + acp)) if (aap + acp) > 0 else 0.0
        }
        
        # ====== CLUSTER-F ======
        cl = global_metrics['cluster']
        cluster_recall = cl['matches'] / cl['n_truth'] if cl['n_truth'] > 0 else 0.0
        cluster_precision = cl['matches'] / cl['n_pred'] if cl['n_pred'] > 0 else 0.0
        cluster_f1 = (2 * cluster_recall * cluster_precision / (cluster_recall + cluster_precision)) if (cluster_recall + cluster_precision) > 0 else 0.0
        
        results['cluster'] = {
            'recall': cluster_recall,
            'precision': cluster_precision,
            'f1': cluster_f1
        }
        
        # ====== SPLITTING & LUMPING ======
        sl = global_metrics['split_lump']
        
        splitting_error = sl['split_sum'] / sl['split_total'] if sl['split_total'] > 0 else 0.0
        splitting_recall = 1 - splitting_error
        
        lumping_error = sl['lump_sum'] / sl['lump_total'] if sl['lump_total'] > 0 else 0.0
        lumping_precision = 1 - lumping_error
        
        sl_f1 = (2 * splitting_recall * lumping_precision / (splitting_recall + lumping_precision)) if (splitting_recall + lumping_precision) > 0 else 0.0
        
        results['split_lump'] = {
            'splitting_error': splitting_error,
            'lumping_error': lumping_error,
            'recall': splitting_recall,
            'precision': lumping_precision,
            'f1': sl_f1
        }
        
        return results
    
    def _compute_name_level_stats(self):
        """Calcola statistiche aggregate per-name"""
        if not self.name_level_results:
            return {}
        
        trivial_names = [n for n, m in self.name_level_results.items() if m['is_trivial']]
        difficult_names = [n for n, m in self.name_level_results.items() if not m['is_trivial']]
        
        stats = {
            'total_names': len(self.name_level_results),
            'trivial_names': len(trivial_names),
            'difficult_names': len(difficult_names),
            'trivial_ratio': len(trivial_names) / len(self.name_level_results) if self.name_level_results else 0
        }
        
        if difficult_names:
            difficult_metrics = [self.name_level_results[n] for n in difficult_names]
            
            stats['difficult'] = {
                'avg_truth_clusters': np.mean([m['n_truth_clusters'] for m in difficult_metrics]),
                'avg_pred_clusters': np.mean([m['n_pred_clusters'] for m in difficult_metrics]),
                'avg_instances': np.mean([m['n_instances'] for m in difficult_metrics]),
                'max_instances': max([m['n_instances'] for m in difficult_metrics])
            }
        
        return stats
    
    def _compute_composite_score(self):
        """
        Calcola composite score usando pesi standard
        """
        METRIC_WEIGHTS = {
            'pairwise_f1': 0.35,
            'k_metric': 0.30,
            'cluster_f1': 0.15,
            'splitting_error': 0.10,
            'lumping_error': 0.10
        }
        
        pairwise_f1 = self.results.get('pairwise', {}).get('f1', 0.0)
        k_metric = self.results.get('k_metric', {}).get('k', 0.0)
        cluster_f1 = self.results.get('cluster', {}).get('f1', 0.0)
        splitting_score = 1 - self.results.get('split_lump', {}).get('splitting_error', 1.0)
        lumping_score = 1 - self.results.get('split_lump', {}).get('lumping_error', 1.0)
        
        composite = (
            METRIC_WEIGHTS['pairwise_f1'] * pairwise_f1 +
            METRIC_WEIGHTS['k_metric'] * k_metric +
            METRIC_WEIGHTS['cluster_f1'] * cluster_f1 +
            METRIC_WEIGHTS['splitting_error'] * splitting_score +
            METRIC_WEIGHTS['lumping_error'] * lumping_score
        )
        
        return composite
    
    def print_results(self, detailed=True):
        """Stampa risultati in formato leggibile"""
        if not self.results:
            print("\n❌ No results available")
            return
        
        print("\n" + "="*70)
        print("MULTI-METRIC EVALUATION RESULTS")
        print("Based on Kim (2019) - All metrics computed in single framework")
        print("="*70)
        
        # Dataset statistics
        print("\n[DATASET STATISTICS]")
        print(f"  Total names evaluated:    {self.stats['total_names']}")
        print(f"  Total instances:          {self.stats['total_instances']}")
        print(f"  Trivial cases (1 paper):  {self.stats['trivial_names']} ({self.stats['trivial_names']/self.stats['total_names']:.1%})")
        print(f"  Difficult cases:          {self.stats['difficult_names']} ({self.stats['difficult_names']/self.stats['total_names']:.1%})")
        print(f"  Perfect matches:          {self.stats['perfect_matches']}")
        
        # Pairwise-F
        print("\n[PAIRWISE-F] (Most commonly used metric)")
        pw = self.results['pairwise']
        print(f"  Precision: {pw['precision']:.4f}")
        print(f"  Recall:    {pw['recall']:.4f}")
        print(f"  F1:        {pw['f1']:.4f}")
        
        # K-metric
        print("\n[K-METRIC] (Geometric mean of AAP and ACP)")
        km = self.results['k_metric']
        print(f"  ACP (precision): {km['acp']:.4f}")
        print(f"  AAP (recall):    {km['aap']:.4f}")
        print(f"  K (geometric):   {km['k']:.4f}")
        
        # B³
        print("\n[B³ (B-CUBED)] (Equivalent to K-metric)")
        b3 = self.results['b3']
        print(f"  Precision: {b3['precision']:.4f}")
        print(f"  Recall:    {b3['recall']:.4f}")
        print(f"  F1:        {b3['f1']:.4f}")
        
        # Cluster-F
        print("\n[CLUSTER-F] (Strict - requires perfect cluster match)")
        cl = self.results['cluster']
        print(f"  Precision: {cl['precision']:.4f}")
        print(f"  Recall:    {cl['recall']:.4f}")
        print(f"  F1:        {cl['f1']:.4f}")
        
        # Split & Lump
        print("\n[SPLITTING & LUMPING ERRORS]")
        sl = self.results['split_lump']
        print(f"  Splitting Error: {sl['splitting_error']:.4f}  (lower is better)")
        print(f"  Lumping Error:   {sl['lumping_error']:.4f}  (lower is better)")
        print(f"  Recall (1-SE):   {sl['recall']:.4f}")
        print(f"  Precision (1-LE):{sl['precision']:.4f}")
        print(f"  F1:              {sl['f1']:.4f}")
        
        # Composite score
        print("\n[COMPOSITE SCORE] (Weighted combination of all metrics)")
        print(f"  Score: {self.results['composite_score']:.4f}")
        
        # Name-level stats
        if detailed:
            stats = self.results.get('name_level_stats', {})
            if 'difficult' in stats:
                print("\n[DIFFICULT CASES STATISTICS]")
                diff = stats['difficult']
                print(f"  Avg truth clusters per name: {diff['avg_truth_clusters']:.2f}")
                print(f"  Avg pred clusters per name:  {diff['avg_pred_clusters']:.2f}")
                print(f"  Avg instances per name:      {diff['avg_instances']:.2f}")
                print(f"  Max instances in a name:     {diff['max_instances']}")
        
        print("\n" + "="*70)
        
        # Interpretazione
        self._print_interpretation()
    
    def _print_interpretation(self):
        """Stampa interpretazione dei risultati"""
        print("\n[INTERPRETATION]")
        
        pw_f1 = self.results['pairwise']['f1']
        cluster_f1 = self.results['cluster']['f1']
        k = self.results['k_metric']['k']
        
        # Gap tra Pairwise e Cluster-F
        gap = pw_f1 - cluster_f1
        
        if gap > 0.2:
            print("  ⚠️  Large gap between Pairwise-F1 and Cluster-F1!")
            print("      → Pairwise-F may be overoptimistic")
            print("      → Many small errors across different clusters")
            print("      → Consider using Cluster-F or K-metric for optimization")
        elif gap > 0.1:
            print("  ⚡ Moderate gap between Pairwise-F1 and Cluster-F1")
            print("      → Some clustering imperfections")
            print("      → Overall good performance")
        else:
            print("  ✅ Small gap between metrics - consistent performance")
        
        # Splitting vs Lumping
        split_err = self.results['split_lump']['splitting_error']
        lump_err = self.results['split_lump']['lumping_error']
        
        if split_err > lump_err + 0.1:
            print("\n  📊 More splitting than lumping errors")
            print("      → Tendency to split authors into multiple clusters")
            print("      → Consider lowering clustering thresholds")
        elif lump_err > split_err + 0.1:
            print("\n  📊 More lumping than splitting errors")
            print("      → Tendency to merge different authors")
            print("      → Consider raising clustering thresholds")
        else:
            print("\n  📊 Balanced splitting and lumping")
        
        # Overall assessment
        print("\n[OVERALL ASSESSMENT]")
        composite = self.results['composite_score']
        
        if composite >= 0.90:
            print("  🌟 Excellent performance!")
        elif composite >= 0.80:
            print("  ✅ Good performance")
        elif composite >= 0.70:
            print("  ⚡ Moderate performance - room for improvement")
        else:
            print("  ⚠️  Low performance - significant improvements needed")
    
    def find_problematic_names(self, top_k=10):
        """Trova i nomi più problematici"""
        if not self.name_level_results:
            return []
        
        # Calcola score per ogni nome (media di recall e precision)
        name_scores = []
        
        for name, metrics in self.name_level_results.items():
            if metrics['is_trivial']:
                continue  # Skip casi triviali
            
            # Calcola pairwise F1 per questo nome
            tp = metrics['pairwise']['tp']
            fp = metrics['pairwise']['fp']
            fn = metrics['pairwise']['fn']
            
            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0.0
            
            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0.0
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            
            name_scores.append({
                'name': name,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'n_instances': metrics['n_instances'],
                'n_truth_clusters': metrics['n_truth_clusters'],
                'n_pred_clusters': metrics['n_pred_clusters']
            })
        
        # Ordina per F1 crescente (peggiori primi)
        name_scores.sort(key=lambda x: x['f1'])
        
        return name_scores[:top_k]
    
    def save_results(self, output_path=None):
        """Salva risultati in JSON"""
        if output_path is None:
            if self.output_dir:
                output_path = self.output_dir / "evaluation_results.json"
            else:
                output_path = Path("evaluation_results.json")
        
        output_data = {
            'summary': self.results,
            'statistics': self.stats,
            'timestamp': str(np.datetime64('now')),
            'files': {
                'predictions': str(self.predictions_file),
                'ground_truth': str(self.ground_truth_file)
            }
        }
        
        save_json(output_data, output_path)
        print(f"\n✅ Results saved to: {output_path}")
    
    def _empty_results(self):
        """Risultati vuoti"""
        return {
            'pairwise': {'recall': 0.0, 'precision': 0.0, 'f1': 0.0},
            'k_metric': {'aap': 0.0, 'acp': 0.0, 'k': 0.0},
            'b3': {'recall': 0.0, 'precision': 0.0, 'f1': 0.0},
            'cluster': {'recall': 0.0, 'precision': 0.0, 'f1': 0.0},
            'split_lump': {
                'splitting_error': 1.0, 'lumping_error': 1.0,
                'recall': 0.0, 'precision': 0.0, 'f1': 0.0
            },
            'name_level_stats': {},
            'composite_score': 0.0
        }
    
    def _empty_name_metrics(self):
        """Metriche vuote per un nome"""
        return {
            'n_instances': 0,
            'n_truth_clusters': 0,
            'n_pred_clusters': 0,
            'is_trivial': True,
            'pairwise': {'tp': 0, 'fp': 0, 'fn': 0},
            'k_metric': {'aap_sum': 0, 'acp_sum': 0},
            'cluster': {'match': 0},
            'split_lump': {'split': 0, 'lump': 0}
        }


def debug_data_formats(predictions_file, ground_truth_file, max_names=5):
    """
    Debug function per capire i formati dei dati
    """
    predict_result = load_json(predictions_file)
    ground_truth = load_json(ground_truth_file)
    
    print("\n" + "="*70)
    print("DEBUG: DATA FORMATS")
    print("="*70)
    
    # Controlla nomi comuni
    common_names = set(predict_result.keys()) & set(ground_truth.keys())
    print(f"\nNames in predictions: {len(predict_result)}")
    print(f"Names in ground truth: {len(ground_truth)}")
    print(f"Common names: {len(common_names)}")
    
    if len(common_names) == 0:
        print("\n❌ ERROR: No common names!")
        print(f"\nPrediction names sample: {list(predict_result.keys())[:5]}")
        print(f"Ground truth names sample: {list(ground_truth.keys())[:5]}")
        return
    
    # Analizza alcuni nomi
    print(f"\n[Analyzing {min(max_names, len(common_names))} sample names]\n")
    
    for i, name in enumerate(list(common_names)[:max_names]):
        print(f"--- {name} ---")
        print(f"  Predictions: {len(predict_result[name])} clusters")
        
        # Mostra sample dei cluster predetti
        for j, cluster in enumerate(predict_result[name][:2]):
            print(f"    Cluster {j}: {len(cluster)} papers - {cluster[:3]}...")
        
        print(f"  Ground truth type: {type(ground_truth[name])}")
        
        if isinstance(ground_truth[name], dict):
            print(f"    Dict with {len(ground_truth[name])} authors")
            for j, (aid, papers) in enumerate(list(ground_truth[name].items())[:2]):
                print(f"      Author {aid}: {len(papers)} papers")
        elif isinstance(ground_truth[name], list):
            print(f"    List with {len(ground_truth[name])} clusters")
            for j, cluster in enumerate(ground_truth[name][:2]):
                print(f"      Cluster {j}: {len(cluster) if isinstance(cluster, list) else 1} papers")
        
        print()


def main():
    """Main function con esempi di uso"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Multi-metric evaluation for Author Name Disambiguation (Kim 2019)'
    )
    parser.add_argument('--predictions', '-p', required=True,
                       help='Path to predictions JSON file')
    parser.add_argument('--ground_truth', '-g', required=True,
                       help='Path to ground truth JSON file')
    parser.add_argument('--output', '-o', default=None,
                       help='Output directory for results')
    parser.add_argument('--debug', action='store_true',
                       help='Run debug mode to check data formats')
    parser.add_argument('--find_problems', '-f', type=int, default=0,
                       help='Find top K problematic names')
    
    args = parser.parse_args()
    
    # Debug mode
    if args.debug:
        debug_data_formats(args.predictions, args.ground_truth)
        return
    
    # Evaluation
    print("\n" + "="*70)
    print("MULTI-METRIC EVALUATION")
    print("="*70)
    
    evaluator = MultiMetricEvaluator(
        args.predictions,
        args.ground_truth,
        output_dir=args.output
    )
    
    # Calcola metriche
    results = evaluator.evaluate_all_metrics()
    
    # Stampa risultati
    evaluator.print_results(detailed=True)
    
    # Trova nomi problematici
    if args.find_problems > 0:
        print("\n" + "="*70)
        print(f"TOP {args.find_problems} MOST PROBLEMATIC NAMES")
        print("="*70)
        
        problematic = evaluator.find_problematic_names(top_k=args.find_problems)
        
        for i, name_data in enumerate(problematic, 1):
            print(f"\n{i}. {name_data['name']}")
            print(f"   F1: {name_data['f1']:.4f}  |  Precision: {name_data['precision']:.4f}  |  Recall: {name_data['recall']:.4f}")
            print(f"   Instances: {name_data['n_instances']}  |  Truth clusters: {name_data['n_truth_clusters']}  |  Pred clusters: {name_data['n_pred_clusters']}")
    
    # Salva risultati
    if args.output:
        evaluator.save_results()


if __name__ == '__main__':
    # Se chiamato direttamente senza argomenti, usa percorsi di default
    import sys
    
    if len(sys.argv) == 1:
        # Modalità interattiva
        print("\n" + "="*70)
        print("MULTI-METRIC EVALUATION - Interactive Mode")
        print("="*70)
        
        # Chiedi i percorsi
        predictions = input("\nPath to predictions file: ").strip()
        ground_truth = input("Path to ground truth file: ").strip()
        
        if not predictions or not ground_truth:
            print("\n❌ Error: Both files are required!")
            sys.exit(1)
        
        # Debug first?
        debug_choice = input("\nRun debug first? (y/n): ").strip().lower()
        
        if debug_choice == 'y':
            debug_data_formats(predictions, ground_truth)
            print("\n" + "="*70)
            proceed = input("\nProceed with evaluation? (y/n): ").strip().lower()
            if proceed != 'y':
                sys.exit(0)
        
        # Evaluate
        evaluator = MultiMetricEvaluator(predictions, ground_truth)
        results = evaluator.evaluate_all_metrics()
        evaluator.print_results(detailed=True)
        
        # Find problems?
        find_choice = input("\nFind problematic names? (enter number or 0 to skip): ").strip()
        
        try:
            n_problems = int(find_choice)
            if n_problems > 0:
                print("\n" + "="*70)
                print(f"TOP {n_problems} MOST PROBLEMATIC NAMES")
                print("="*70)
                
                problematic = evaluator.find_problematic_names(top_k=n_problems)
                
                for i, name_data in enumerate(problematic, 1):
                    print(f"\n{i}. {name_data['name']}")
                    print(f"   F1: {name_data['f1']:.4f}  |  P: {name_data['precision']:.4f}  |  R: {name_data['recall']:.4f}")
                    print(f"   Instances: {name_data['n_instances']}  |  Truth: {name_data['n_truth_clusters']}  |  Pred: {name_data['n_pred_clusters']}")
        except:
            pass
        
        # Save?
        save_choice = input("\nSave results to JSON? (y/n): ").strip().lower()
        if save_choice == 'y':
            output_path = input("Output path (default: evaluation_results.json): ").strip()
            if not output_path:
                output_path = "evaluation_results.json"
            evaluator.save_results(output_path)
    else:
        # Command line mode
        main()





#    predict = r'C:\Users\franc\OneDrive - Alma Mater Studiorum Università di Bologna\Desktop\BOND-OC\WhoIsWho\bond\out\res.json'
#    ground_truth = r"C:\Users\franc\OneDrive - Alma Mater Studiorum Università di Bologna\Desktop\BOND-OC\WhoIsWho\bond\dataset\data\src\sna-valid\sna_valid_ground_truth.json"
    
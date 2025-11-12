"""
Script di ottimizzazione iperparametri per BOND con valutazione multi-metrica
Basato su Kim (2019) "A fast and integrative algorithm for clustering performance 
evaluation in author name disambiguation"

MIGLIORAMENTI RISPETTO ALLA VERSIONE PRECEDENTE:
1. Valutazione con multiple metriche (non solo Pairwise-F1)
2. Funzione obiettivo composita che bilancia precision/recall
3. Early stopping basato su convergenza effettiva
4. Distinzione tra casi triviali (1 paper) e difficili (multipli papers)
5. Analisi per-name per identificare punti deboli
"""
import optuna
import json
import subprocess
import os
import sys
import time
import shutil
import numpy as np
from datetime import datetime, timedelta
from optuna.samplers import TPESampler
from pathlib import Path
from collections import defaultdict

# ======================== CONFIGURAZIONE ========================
BASE_PATH = Path(__file__).parent
DEMO_SCRIPT = BASE_PATH / "demo.py"
DATA_PATH = BASE_PATH / "dataset" / "data"

RESULTS_DIR = BASE_PATH / "hyperopt_multimetric_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Citazioni - Range da esplorare
CITATION_CONFIG = {
    'use_citations': True,
    'cite_out_weight': (0.5, 1.5),
    'cite_in_weight': (0.5, 1.5),
    'cite_out_th': (0.0, 0.2),
    'cite_in_th': (0.0, 0.2)
}

# ==================== CONFIGURAZIONE EARLY STOPPING ====================
EARLY_STOPPING_CONFIG = {
    'patience': 15,  # trials senza miglioramento prima di fermarsi
    'min_delta': 0.001,  # miglioramento minimo considerato significativo
    'min_trials': 20,  # minimo numero di trial prima di considerare early stopping
    'convergence_window': 10,  # finestra per calcolare convergenza
    'convergence_threshold': 0.0005  # varianza massima per considerare convergenza
}

# ==================== PESI METRICHE ====================
# Questi pesi definiscono quanto ogni metrica contribuisce allo score finale
METRIC_WEIGHTS = {
    'pairwise_f1': 0.35,      # Standard, ma può essere ottimistico
    'k_metric': 0.30,          # Geometrica di AAP/ACP, buona per bilanciare
    'cluster_f1': 0.15,        # Strict, penalizza anche singoli errori
    'splitting_error': 0.10,   # Quanto split ci sono (recall)
    'lumping_error': 0.10      # Quanto lump ci sono (precision)
}

# Verifica che i pesi sommino a 1
assert abs(sum(METRIC_WEIGHTS.values()) - 1.0) < 1e-6, "I pesi devono sommare a 1!"

# ================================================================


class MultiMetricEvaluator:
    """
    Valutatore multi-metrica basato su Kim (2019)
    
    Implementa il framework integrato che calcola tutte le metriche
    in un singolo passaggio usando hash tables.
    """
    
    def __init__(self, predictions_file, ground_truth_file):
        self.predictions_file = predictions_file
        self.ground_truth_file = ground_truth_file
        self.results = {}
        self.name_level_results = {}
        
    def evaluate_all_metrics(self):
        """
        Calcola tutte le metriche in un singolo framework integrato
        seguendo l'Algorithm 6 di Kim (2019)
        """
        try:
            # Carica dati
            with open(self.predictions_file, 'r', encoding='utf-8') as f:
                predict_result = json.load(f)
            with open(self.ground_truth_file, 'r', encoding='utf-8') as f:
                ground_truth = json.load(f)
            
            # Filtra solo nomi comuni
            filtered_predict = {n: p for n, p in predict_result.items() 
                               if n in ground_truth}
            
            if not filtered_predict:
                print("  WARNING: No common names between predictions and ground truth")
                return self._empty_results()
            
            # Inizializza accumulatori per metriche globali
            global_metrics = {
                'pairwise': {'tp': 0, 'fp': 0, 'fn': 0},
                'k_metric': {'aap_sum': 0, 'acp_sum': 0, 'n_instances': 0},
                'cluster': {'matches': 0, 'n_truth': 0, 'n_pred': 0},
                'split_lump': {
                    'split_sum': 0, 'split_total': 0,
                    'lump_sum': 0, 'lump_total': 0
                }
            }
            
            # Risultati per-name per analisi dettagliata
            self.name_level_results = {}
            
            # ============ FRAMEWORK INTEGRATO (Algorithm 6 del paper) ============
            for name in filtered_predict:
                name_metrics = self._evaluate_single_name(
                    name, 
                    filtered_predict[name], 
                    ground_truth[name]
                )
                
                # Accumula metriche globali
                self._accumulate_metrics(global_metrics, name_metrics)
                
                # Salva risultati per-name
                self.name_level_results[name] = name_metrics
            
            # Calcola metriche finali
            self.results = self._compute_final_metrics(global_metrics)
            
            # Aggiungi statistiche aggregate
            self.results['name_level_stats'] = self._compute_name_level_stats()
            
            # Calcola score composito
            self.results['composite_score'] = self._compute_composite_score()
            
            return self.results
            
        except Exception as e:
            print(f"  ERROR in multi-metric evaluation: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_results()
    
    def _evaluate_single_name(self, name, predicted_clusters, truth_data):
        """
        Valuta un singolo nome con tutte le metriche
        Implementa il core dell'Algorithm 6
        """
        # Preparazione dati truth
        truth_clusters = self._prepare_truth_clusters(truth_data)
        
        if not truth_clusters or not predicted_clusters:
            return self._empty_name_metrics()
        
        # Crea pIndex: mapping instance -> predicted_cluster_id
        pIndex = {}
        for pred_idx, pred_cluster in enumerate(predicted_clusters):
            for instance in pred_cluster:
                pIndex[instance] = pred_idx
        
        # Statistiche sui cluster
        n_truth_clusters = len(truth_clusters)
        n_pred_clusters = len(predicted_clusters)
        n_instances = sum(len(tc) for tc in truth_clusters)
        
        # Dimensioni cluster predetti
        pred_sizes = [len(pc) for pc in predicted_clusters]
        
        # Inizializza metriche per questo nome
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
        
        # ====== Loop principale sui truth clusters ======
        for truth_cluster in truth_clusters:
            # tMap: predicted_cluster_id -> frequency
            tMap = {}
            
            for instance in truth_cluster:
                if instance in pIndex:
                    pred_idx = pIndex[instance]
                    tMap[pred_idx] = tMap.get(pred_idx, 0) + 1
            
            truth_size = len(truth_cluster)
            
            # Trova il predicted cluster con più istanze di questo truth cluster
            max_overlap = 0
            max_pred_idx = -1
            for pred_idx, overlap_size in tMap.items():
                if overlap_size > max_overlap:
                    max_overlap = overlap_size
                    max_pred_idx = pred_idx
            
            # ====== CLUSTER-F ======
            # Match perfetto: tutte le istanze del truth cluster sono in un 
            # predicted cluster che contiene SOLO quelle istanze
            if max_pred_idx >= 0:
                if (max_overlap == truth_size and 
                    pred_sizes[max_pred_idx] == truth_size):
                    metrics['cluster']['match'] = 1
            
            # ====== K-METRIC e B³ (producono stesso risultato) ======
            for pred_idx, overlap_size in tMap.items():
                # AAP (recall): |Pi ∩ Tj|² / |Tj|
                metrics['k_metric']['aap_sum'] += (overlap_size ** 2) / truth_size
                
                # ACP (precision): |Pi ∩ Tj|² / |Pi|
                metrics['k_metric']['acp_sum'] += (overlap_size ** 2) / pred_sizes[pred_idx]
            
            # ====== PAIRWISE-F ======
            # Usa euristica: #pairs = n*(n-1)/2
            # Truth pairs in questo cluster
            truth_pairs = truth_size * (truth_size - 1) // 2
            
            # Intersection pairs (coppie che appaiono insieme sia in truth che in predicted)
            for pred_idx, overlap_size in tMap.items():
                intersection_pairs = overlap_size * (overlap_size - 1) // 2
                metrics['pairwise']['tp'] += intersection_pairs
            
            # False negatives: truth pairs NON trovati insieme in predicted
            metrics['pairwise']['fn'] += truth_pairs - sum(
                overlap * (overlap - 1) // 2 for overlap in tMap.values()
            )
            
            # ====== SPLITTING ERROR ======
            # Istanze del truth cluster NON nel predicted cluster principale
            if max_pred_idx >= 0:
                metrics['split_lump']['split'] += (truth_size - max_overlap)
        
        # ====== FALSE POSITIVES per Pairwise-F ======
        # Calculated from predicted clusters
        for pred_idx, pred_cluster in enumerate(predicted_clusters):
            pred_pairs = len(pred_cluster) * (len(pred_cluster) - 1) // 2
            
            # Conta quante coppie sono corrette
            correct_pairs = 0
            # Usa tMap per ogni truth cluster che interseca questo predicted
            for truth_cluster in truth_clusters:
                overlap = sum(1 for inst in truth_cluster if pIndex.get(inst) == pred_idx)
                correct_pairs += overlap * (overlap - 1) // 2
            
            metrics['pairwise']['fp'] += (pred_pairs - correct_pairs)
        
        # ====== LUMPING ERROR ======
        # Per ogni truth cluster, conta istanze erroneamente aggiunte al suo predicted principale
        for truth_idx, truth_cluster in enumerate(truth_clusters):
            # tMap per questo truth cluster
            tMap = {}
            for instance in truth_cluster:
                if instance in pIndex:
                    pred_idx = pIndex[instance]
                    tMap[pred_idx] = tMap.get(pred_idx, 0) + 1
            
            # Trova predicted principale
            if tMap:
                max_pred_idx = max(tMap.items(), key=lambda x: x[1])[0]
                pred_cluster_size = pred_sizes[max_pred_idx]
                truth_instances_in_pred = tMap[max_pred_idx]
                
                # Lumping: istanze nel predicted che NON appartengono al truth
                metrics['split_lump']['lump'] += (pred_cluster_size - truth_instances_in_pred)
        
        return metrics
    
    def _prepare_truth_clusters(self, truth_data):
        """Converte truth_data in lista di cluster (liste di instance IDs)"""
        clusters = []
        
        if isinstance(truth_data, dict):
            for author_id, instances in truth_data.items():
                if instances:
                    clusters.append(instances)
        elif isinstance(truth_data, list):
            for cluster in truth_data:
                if isinstance(cluster, list) and cluster:
                    clusters.append(cluster)
        
        return clusters
    
    def _accumulate_metrics(self, global_metrics, name_metrics):
        """Accumula metriche da un singolo nome nelle metriche globali"""
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
        # Per lumping, il totale è la somma delle dimensioni dei predicted cluster principali
        # Approssimazione: usiamo n_instances (sarà corretto in media)
        global_metrics['split_lump']['lump_total'] += name_metrics['n_instances']
    
    def _compute_final_metrics(self, global_metrics):
        """Calcola metriche finali da accumulatori globali"""
        results = {}
        
        # ====== PAIRWISE-F ======
        pw = global_metrics['pairwise']
        if pw['tp'] + pw['fn'] > 0:
            pw_recall = pw['tp'] / (pw['tp'] + pw['fn'])
        else:
            pw_recall = 0.0
        
        if pw['tp'] + pw['fp'] > 0:
            pw_precision = pw['tp'] / (pw['tp'] + pw['fp'])
        else:
            pw_precision = 0.0
        
        if pw_recall + pw_precision > 0:
            pw_f1 = 2 * pw_recall * pw_precision / (pw_recall + pw_precision)
        else:
            pw_f1 = 0.0
        
        results['pairwise'] = {
            'recall': pw_recall,
            'precision': pw_precision,
            'f1': pw_f1
        }
        
        # ====== K-METRIC (AAP, ACP, K) ======
        km = global_metrics['k_metric']
        n = km['n_instances']
        
        if n > 0:
            aap = km['aap_sum'] / n  # recall
            acp = km['acp_sum'] / n  # precision
            k = np.sqrt(aap * acp) if (aap * acp) > 0 else 0.0
        else:
            aap = acp = k = 0.0
        
        results['k_metric'] = {
            'aap': aap,  # recall
            'acp': acp,  # precision
            'k': k       # geometric mean
        }
        
        # ====== CLUSTER-F ======
        cl = global_metrics['cluster']
        if cl['n_truth'] > 0:
            cluster_recall = cl['matches'] / cl['n_truth']
        else:
            cluster_recall = 0.0
        
        if cl['n_pred'] > 0:
            cluster_precision = cl['matches'] / cl['n_pred']
        else:
            cluster_precision = 0.0
        
        if cluster_recall + cluster_precision > 0:
            cluster_f1 = 2 * cluster_recall * cluster_precision / (cluster_recall + cluster_precision)
        else:
            cluster_f1 = 0.0
        
        results['cluster'] = {
            'recall': cluster_recall,
            'precision': cluster_precision,
            'f1': cluster_f1
        }
        
        # ====== SPLITTING & LUMPING ERROR ======
        sl = global_metrics['split_lump']
        
        if sl['split_total'] > 0:
            splitting_error = sl['split_sum'] / sl['split_total']
            splitting_recall = 1 - splitting_error  # eR = 1 - SE
        else:
            splitting_error = 0.0
            splitting_recall = 1.0
        
        if sl['lump_total'] > 0:
            lumping_error = sl['lump_sum'] / sl['lump_total']
            lumping_precision = 1 - lumping_error  # eP = 1 - LE
        else:
            lumping_error = 0.0
            lumping_precision = 1.0
        
        if splitting_recall + lumping_precision > 0:
            sl_f1 = 2 * splitting_recall * lumping_precision / (splitting_recall + lumping_precision)
        else:
            sl_f1 = 0.0
        
        results['split_lump'] = {
            'splitting_error': splitting_error,
            'lumping_error': lumping_error,
            'recall': splitting_recall,
            'precision': lumping_precision,
            'f1': sl_f1
        }
        
        return results
    
    def _compute_name_level_stats(self):
        """Calcola statistiche aggregate sui risultati per-name"""
        if not self.name_level_results:
            return {}
        
        # Separa nomi triviali (1 paper, 1 author) da quelli difficili
        trivial_names = [n for n, m in self.name_level_results.items() if m['is_trivial']]
        difficult_names = [n for n, m in self.name_level_results.items() if not m['is_trivial']]
        
        stats = {
            'total_names': len(self.name_level_results),
            'trivial_names': len(trivial_names),
            'difficult_names': len(difficult_names),
            'trivial_ratio': len(trivial_names) / len(self.name_level_results) if self.name_level_results else 0
        }
        
        # Statistiche sui casi difficili
        if difficult_names:
            difficult_metrics = [self.name_level_results[n] for n in difficult_names]
            
            # Distribuzioni
            stats['difficult'] = {
                'avg_truth_clusters': np.mean([m['n_truth_clusters'] for m in difficult_metrics]),
                'avg_pred_clusters': np.mean([m['n_pred_clusters'] for m in difficult_metrics]),
                'avg_instances': np.mean([m['n_instances'] for m in difficult_metrics]),
                'max_instances': max([m['n_instances'] for m in difficult_metrics])
            }
        
        return stats
    
    def _compute_composite_score(self):
        """
        Calcola score composito usando i pesi definiti
        
        Questo è lo score che Optuna cercherà di massimizzare.
        """
        if not self.results:
            return 0.0
        
        # Estrai gli score dalle diverse metriche
        pairwise_f1 = self.results.get('pairwise', {}).get('f1', 0.0)
        k_metric = self.results.get('k_metric', {}).get('k', 0.0)
        cluster_f1 = self.results.get('cluster', {}).get('f1', 0.0)
        
        # Per split/lump, usiamo 1 - error (così più alto = meglio)
        splitting_score = 1 - self.results.get('split_lump', {}).get('splitting_error', 1.0)
        lumping_score = 1 - self.results.get('split_lump', {}).get('lumping_error', 1.0)
        
        # Score composito pesato
        composite = (
            METRIC_WEIGHTS['pairwise_f1'] * pairwise_f1 +
            METRIC_WEIGHTS['k_metric'] * k_metric +
            METRIC_WEIGHTS['cluster_f1'] * cluster_f1 +
            METRIC_WEIGHTS['splitting_error'] * splitting_score +
            METRIC_WEIGHTS['lumping_error'] * lumping_score
        )
        
        return composite
    
    def _empty_results(self):
        """Ritorna struttura results vuota"""
        return {
            'pairwise': {'recall': 0.0, 'precision': 0.0, 'f1': 0.0},
            'k_metric': {'aap': 0.0, 'acp': 0.0, 'k': 0.0},
            'cluster': {'recall': 0.0, 'precision': 0.0, 'f1': 0.0},
            'split_lump': {
                'splitting_error': 1.0, 'lumping_error': 1.0,
                'recall': 0.0, 'precision': 0.0, 'f1': 0.0
            },
            'name_level_stats': {},
            'composite_score': 0.0
        }
    
    def _empty_name_metrics(self):
        """Ritorna struttura name_metrics vuota"""
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
    
    def print_detailed_results(self):
        """Stampa risultati dettagliati per debugging"""
        if not self.results:
            print("No results available")
            return
        
        print("\n" + "="*70)
        print("DETAILED MULTI-METRIC EVALUATION RESULTS")
        print("="*70)
        
        # Pairwise-F
        print("\n[PAIRWISE-F]")
        pw = self.results['pairwise']
        print(f"  Precision: {pw['precision']:.4f}")
        print(f"  Recall:    {pw['recall']:.4f}")
        print(f"  F1:        {pw['f1']:.4f}")
        
        # K-metric
        print("\n[K-METRIC]")
        km = self.results['k_metric']
        print(f"  ACP (precision): {km['acp']:.4f}")
        print(f"  AAP (recall):    {km['aap']:.4f}")
        print(f"  K (geometric):   {km['k']:.4f}")
        
        # Cluster-F
        print("\n[CLUSTER-F]")
        cl = self.results['cluster']
        print(f"  Precision: {cl['precision']:.4f}")
        print(f"  Recall:    {cl['recall']:.4f}")
        print(f"  F1:        {cl['f1']:.4f}")
        
        # Split & Lump
        print("\n[SPLITTING & LUMPING]")
        sl = self.results['split_lump']
        print(f"  Splitting Error: {sl['splitting_error']:.4f}")
        print(f"  Lumping Error:   {sl['lumping_error']:.4f}")
        print(f"  Recall (1-SE):   {sl['recall']:.4f}")
        print(f"  Precision (1-LE):{sl['precision']:.4f}")
        print(f"  F1:              {sl['f1']:.4f}")
        
        # Name-level stats
        print("\n[NAME-LEVEL STATISTICS]")
        stats = self.results.get('name_level_stats', {})
        if stats:
            print(f"  Total names:      {stats.get('total_names', 0)}")
            print(f"  Trivial (1 paper):{stats.get('trivial_names', 0)} ({stats.get('trivial_ratio', 0):.1%})")
            print(f"  Difficult:        {stats.get('difficult_names', 0)}")
            
            if 'difficult' in stats:
                diff = stats['difficult']
                print(f"\n  [Difficult names stats]")
                print(f"    Avg truth clusters: {diff.get('avg_truth_clusters', 0):.1f}")
                print(f"    Avg pred clusters:  {diff.get('avg_pred_clusters', 0):.1f}")
                print(f"    Avg instances:      {diff.get('avg_instances', 0):.1f}")
                print(f"    Max instances:      {diff.get('max_instances', 0)}")
        
        # Composite score
        print("\n[COMPOSITE SCORE]")
        print(f"  Weighted score: {self.results['composite_score']:.4f}")
        print(f"\n  Weights used:")
        for metric, weight in METRIC_WEIGHTS.items():
            print(f"    {metric:20s}: {weight:.2f}")
        
        print("="*70 + "\n")


class EarlyStoppingCallback:
    """
    Callback per early stopping intelligente
    
    Si ferma quando:
    1. Nessun miglioramento significativo per `patience` trials
    2. Le performance convergono (varianza bassa nella finestra)
    """
    
    def __init__(self, patience=15, min_delta=0.001, min_trials=20, 
                 convergence_window=10, convergence_threshold=0.0005):
        self.patience = patience
        self.min_delta = min_delta
        self.min_trials = min_trials
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold
        
        self.best_value = -float('inf')
        self.best_trial = None
        self.trials_without_improvement = 0
        self.trial_values = []
        
    def __call__(self, study, trial):
        """
        Chiamato dopo ogni trial
        """
        # Salta se non abbiamo completato trials minimi
        if len(study.trials) < self.min_trials:
            return
        
        current_value = trial.value
        if current_value is None:
            return
        
        self.trial_values.append(current_value)
        
        # Check 1: Miglioramento significativo?
        improvement = current_value - self.best_value
        
        if improvement > self.min_delta:
            # Abbiamo un miglioramento!
            self.best_value = current_value
            self.best_trial = trial.number
            self.trials_without_improvement = 0
            print(f"  ✓ New best score: {current_value:.4f} (+{improvement:.4f})")
        else:
            self.trials_without_improvement += 1
            print(f"  ○ No improvement for {self.trials_without_improvement} trials " 
                  f"(best: {self.best_value:.4f})")
        
        # Check 2: Convergenza?
        if len(self.trial_values) >= self.convergence_window:
            recent_values = self.trial_values[-self.convergence_window:]
            variance = np.var(recent_values)
            
            if variance < self.convergence_threshold:
                print(f"\n  ⚠ Convergence detected! Variance={variance:.6f} < {self.convergence_threshold}")
                print(f"  Stopping optimization at trial {trial.number}")
                study.stop()
                return
        
        # Check 3: Patience esaurita?
        if self.trials_without_improvement >= self.patience:
            print(f"\n  ⚠ Early stopping! No improvement for {self.patience} trials")
            print(f"  Best trial was #{self.best_trial} with score {self.best_value:.4f}")
            study.stop()


class ProgressTracker:
    """Tracker per mostrare il progresso dell'ottimizzazione"""
    def __init__(self, n_trials):
        self.n_trials = n_trials
        self.start_time = time.time()
        self.best_value = 0.0
        self.current_trial = 0
        
    def __call__(self, study, trial):
        self.current_trial += 1
        elapsed = time.time() - self.start_time
        
        if self.current_trial > 1:
            avg_time = elapsed / self.current_trial
            remaining = (self.n_trials - self.current_trial) * avg_time
            eta = str(timedelta(seconds=int(remaining)))
        else:
            eta = "Calculating..."
        
        if trial.value and trial.value > self.best_value:
            self.best_value = trial.value
        
        progress = (self.current_trial / self.n_trials) * 100
        print(f"\n{'='*70}")
        print(f"Progress: Trial {self.current_trial}/{self.n_trials} ({progress:.1f}%)")
        print(f"Best Composite Score so far: {self.best_value:.4f}")
        print(f"ETA: {eta}")
        print(f"{'='*70}\n")


def objective(trial):
    """
    Funzione obiettivo per Optuna con valutazione multi-metrica
    
    OTTIMIZZA: Composite score (combinazione pesata di tutte le metriche)
    invece di solo Pairwise-F1
    """
    # Parametri da ottimizzare
    params = {
        # Clustering
        'db_eps': trial.suggest_float('db_eps', 0.05, 0.3, step=0.01),
        'db_min': trial.suggest_int('db_min', 3, 10),
        'cluster_w': trial.suggest_float('cluster_w', 0.1, 0.9, step=0.1),
        
        # Training
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'l2_coef': trial.suggest_float('l2_coef', 1e-5, 1e-3, log=True),
        
        # Architecture
        'hidden_dim_0': trial.suggest_categorical('hidden_dim_0', [128, 256, 512]),
        'hidden_dim_1': trial.suggest_categorical('hidden_dim_1', [256, 512, 1024]),
        'compress_ratio': trial.suggest_float('compress_ratio', 0.5, 1.0, step=0.1),
        
        # Thresholds
        'th_a_0': trial.suggest_float('th_a_0', 0.0, 0.5),
        'th_a_1': trial.suggest_float('th_a_1', 0.5, 1.5),
        'th_o_0': trial.suggest_float('th_o_0', 0.3, 0.8),
        'th_o_1': trial.suggest_float('th_o_1', 0.3, 0.8),
        'th_v_0': trial.suggest_float('th_v_0', 0.5, 2.0),
        'th_v_1': trial.suggest_float('th_v_1', 1.0, 3.0),
        
        # Citazioni
        'use_citations': True,
        'cite_out_weight': trial.suggest_float('cite_out_weight', 
                                               CITATION_CONFIG['cite_out_weight'][0],
                                               CITATION_CONFIG['cite_out_weight'][1],
                                               step=0.1),
        'cite_in_weight': trial.suggest_float('cite_in_weight',
                                              CITATION_CONFIG['cite_in_weight'][0],
                                              CITATION_CONFIG['cite_in_weight'][1],
                                              step=0.1),
        'cite_out_th': trial.suggest_float('cite_out_th',
                                           CITATION_CONFIG['cite_out_th'][0],
                                           CITATION_CONFIG['cite_out_th'][1],
                                           step=0.05),
        'cite_in_th': trial.suggest_float('cite_in_th',
                                          CITATION_CONFIG['cite_in_th'][0],
                                          CITATION_CONFIG['cite_in_th'][1],
                                          step=0.05),
        
        'epochs': 50
    }

    # Environment
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env["SKIP_PREPROCESS"] = "1"

    # Directory trial
    trial_dir = RESULTS_DIR / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Comando BOND
    cmd = [
        'python', str(DEMO_SCRIPT),
        '--mode', 'train',
        '--save_path', str(DATA_PATH),
        '--db_eps', str(params['db_eps']),
        '--db_min', str(params['db_min']),
        '--cluster_w', str(params['cluster_w']),
        '--lr', str(params['lr']),
        '--l2_coef', str(params['l2_coef']),
        '--epochs', str(params['epochs']),
        '--hidden_dim', str(params['hidden_dim_0']), str(params['hidden_dim_1']),
        '--compress_ratio', str(params['compress_ratio']),
        '--th_a', str(params['th_a_0']), str(params['th_a_1']),
        '--th_o', str(params['th_o_0']), str(params['th_o_1']),
        '--th_v', str(params['th_v_0']), str(params['th_v_1']),
        '--cite_out_weight', str(params['cite_out_weight']),
        '--cite_in_weight', str(params['cite_in_weight']),
        '--cite_out_th', str(params['cite_out_th']),
        '--cite_in_th', str(params['cite_in_th'])
    ]

    print(f"\nTrial {trial.number} - Starting BOND training...")
    print(f"  Clustering: eps={params['db_eps']}, min={params['db_min']}, w={params['cluster_w']}")
    print(f"  Training: lr={params['lr']:.2e}, l2={params['l2_coef']:.2e}")
    print(f"  Citations: out_w={params['cite_out_weight']:.2f}, in_w={params['cite_in_weight']:.2f}")
    
    try:
        # Esegui BOND
        result = subprocess.run(
            cmd, 
            env=env,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=7200
        )
        
        if result.returncode != 0:
            print(f"  WARNING: BOND exited with code {result.returncode}")
            with open(trial_dir / "error.txt", 'w') as f:
                f.write(result.stderr)
        
        # File predizioni
        predictions_file = BASE_PATH / "out" / "res.json"
        ground_truth_file = DATA_PATH / "src" / "train" / "train_author.json"
        
        if not predictions_file.exists():
            print(f"  ERROR: Predictions file not found: {predictions_file}")
            return 0.0
        
        # Copia predizioni
        trial_predictions = trial_dir / "predictions.json"
        shutil.copy2(predictions_file, trial_predictions)
        
        # ============ VALUTAZIONE MULTI-METRICA ============
        print(f"\n  Evaluating with multiple metrics...")
        evaluator = MultiMetricEvaluator(trial_predictions, ground_truth_file)
        results = evaluator.evaluate_all_metrics()
        
        # Stampa risultati dettagliati
        evaluator.print_detailed_results()
        
        # Lo score da ottimizzare è il composite score
        composite_score = results['composite_score']
        
        # Salva risultati completi
        result_data = {
            "trial": trial.number,
            "composite_score": composite_score,
            "all_metrics": results,
            "params": params,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(trial_dir / "results.json", 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"\n  Trial {trial.number} completed:")
        print(f"    Composite Score: {composite_score:.4f}")
        print(f"    Pairwise F1:     {results['pairwise']['f1']:.4f}")
        print(f"    K-metric:        {results['k_metric']['k']:.4f}")
        print(f"    Cluster F1:      {results['cluster']['f1']:.4f}")
        
        return composite_score

    except subprocess.TimeoutExpired:
        print(f"  ERROR: Trial {trial.number} timed out")
        return 0.0
    except Exception as e:
        print(f"  ERROR: Trial {trial.number} failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def run_bond_validation(params):
    """Esegue validation con i parametri migliori"""
    print("\n" + "="*70)
    print("RUNNING VALIDATION WITH BEST PARAMETERS")
    print("="*70)
    
    # [Il codice rimane identico, usa solo MultiMetricEvaluator invece della vecchia evaluate_predictions]
    # ... ometto per brevità, ma il concetto è lo stesso del training
    pass


def optimize_hyperparameters():
    """Ottimizzazione iperparametri con valutazione multi-metrica"""
    print("\n" + "="*70)
    print("BOND HYPERPARAMETER OPTIMIZATION")
    print("MULTI-METRIC EVALUATION (Kim 2019)")
    print("="*70)
    
    print("\n[CONFIGURATION]")
    print(f"  Optimizing: Composite score (weighted combination)")
    print(f"  Metric weights:")
    for metric, weight in METRIC_WEIGHTS.items():
        print(f"    {metric:20s}: {weight:.2f}")
    
    print(f"\n  Early stopping:")
    print(f"    Patience:            {EARLY_STOPPING_CONFIG['patience']} trials")
    print(f"    Min improvement:     {EARLY_STOPPING_CONFIG['min_delta']:.4f}")
    print(f"    Convergence window:  {EARLY_STOPPING_CONFIG['convergence_window']} trials")
    print(f"    Convergence threshold: {EARLY_STOPPING_CONFIG['convergence_threshold']:.6f}")
    
    # Numero trial (può essere interrotto prima)
    try:
        n_trials = int(input("\nMax number of trials (default 100): ") or "100")
    except ValueError:
        n_trials = 100
    
    print(f"\nStarting optimization:")
    print(f"  - Max trials: {n_trials}")
    print(f"  - Will stop early if convergence detected")
    print(f"  - Results: {RESULTS_DIR}")
    
    input("\nPress ENTER to start...")
    
    # Crea study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name='bond_multimetric_optimization'
    )
    
    # Callbacks
    progress = ProgressTracker(n_trials)
    early_stopping = EarlyStoppingCallback(**EARLY_STOPPING_CONFIG)
    
    # Ottimizza
    study.optimize(
        objective, 
        n_trials=n_trials, 
        callbacks=[progress, early_stopping],
        show_progress_bar=True
    )
    
    # Risultati
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    
    best_trial = study.best_trial
    print(f"\nBest trial: #{best_trial.number}")
    print(f"Best Composite Score: {best_trial.value:.4f}")
    
    # Carica risultati dettagliati del best trial
    best_trial_dir = RESULTS_DIR / f"trial_{best_trial.number:03d}"
    best_results_file = best_trial_dir / "results.json"
    
    if best_results_file.exists():
        with open(best_results_file, 'r') as f:
            best_results = json.load(f)
        
        print(f"\n[Best trial detailed metrics]")
        metrics = best_results['all_metrics']
        print(f"  Pairwise-F1:  {metrics['pairwise']['f1']:.4f}")
        print(f"  K-metric:     {metrics['k_metric']['k']:.4f}")
        print(f"  Cluster-F1:   {metrics['cluster']['f1']:.4f}")
        print(f"  Split Error:  {metrics['split_lump']['splitting_error']:.4f}")
        print(f"  Lump Error:   {metrics['split_lump']['lumping_error']:.4f}")
    
    print(f"\n[Best parameters]")
    for key, value in best_trial.params.items():
        if 'cite' in key:
            print(f"  ⭐ {key}: {value}")
        else:
            print(f"  {key}: {value}")
    
    # Salva best params
    best_params_file = RESULTS_DIR / "best_parameters.json"
    with open(best_params_file, 'w') as f:
        json.dump({
            'trial_number': best_trial.number,
            'composite_score': best_trial.value,
            'params': best_trial.params,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n✓ Best parameters saved: {best_params_file}")
    
    # Analisi convergenza
    print(f"\n[Convergence analysis]")
    trial_values = [t.value for t in study.trials if t.value is not None]
    if len(trial_values) >= 10:
        recent_10 = trial_values[-10:]
        print(f"  Last 10 trials mean: {np.mean(recent_10):.4f}")
        print(f"  Last 10 trials std:  {np.std(recent_10):.4f}")
        print(f"  Total trials run:    {len(study.trials)}/{n_trials}")
    
    print(f"\n{'='*70}")
    print(f"All results saved in: {RESULTS_DIR}")
    print(f"{'='*70}\n")


def main():
    """Entry point"""
    try:
        optimize_hyperparameters()
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
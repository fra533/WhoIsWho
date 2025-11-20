"""
Cluster Confidence Scoring & Filtering
=======================================

Questo script calcola uno score di confidenza per ogni cluster predetto
e ti permette di filtrare/accettare solo i cluster affidabili.

IDEA: Non tutti i cluster sono uguali!
- Alcuni sono molto affidabili (alta coesione, features forti)
- Altri sono dubbi (bassa coesione, features deboli)

SOLUZIONE: Calcola uno score di confidenza e filtra.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


class ClusterConfidenceScorer:
    """
    Calcola confidence scores per i cluster predetti
    
    Lo score si basa su:
    1. Coesione interna (quanto sono simili i paper nel cluster?)
    2. Separazione esterna (quanto è distinto dagli altri cluster?)
    3. Feature strength (citazioni, coautori, venue, etc.)
    4. Cluster size (singoli paper sono meno affidabili)
    """
    
    def __init__(self, predictions_file, pubs_file=None):
        """
        Args:
            predictions_file: res.json con i cluster predetti
            pubs_file: (opzionale) dati dei paper per analisi avanzata
        """
        self.predictions_file = predictions_file
        self.pubs_file = pubs_file
        
        # Carica predizioni
        with open(predictions_file, 'r', encoding='utf-8') as f:
            self.predictions = json.load(f)
        
        # Carica paper data se disponibile
        self.pubs_data = None
        if pubs_file and Path(pubs_file).exists():
            with open(pubs_file, 'r', encoding='utf-8') as f:
                self.pubs_data = json.load(f)
        
        # Risultati
        self.cluster_scores = {}
    
    def compute_all_scores(self):
        """Calcola scores per tutti i cluster"""
        print("\nComputing cluster confidence scores...")
        
        for name, clusters in self.predictions.items():
            name_scores = []
            
            for cluster_idx, cluster in enumerate(clusters):
                score = self._compute_cluster_score(name, cluster, cluster_idx)
                name_scores.append(score)
            
            self.cluster_scores[name] = name_scores
        
        return self.cluster_scores
    
    def _compute_cluster_score(self, name, cluster, cluster_idx):
        """
        Calcola confidence score per un singolo cluster
        
        Returns:
            dict con:
                - overall_score: [0, 1]
                - size_score: basato su dimensione
                - feature_score: basato su features disponibili
                - consistency_score: coerenza interna
        """
        cluster_size = len(cluster)
        
        # ====== SIZE SCORE ======
        # Cluster molto piccoli (1 paper) sono meno affidabili
        # Cluster molto grandi potrebbero essere lumping
        size_score = self._compute_size_score(cluster_size)
        
        # ====== FEATURE SCORE ======
        # Quanto sono forti le features per questo cluster?
        feature_score = self._compute_feature_score(name, cluster)
        
        # ====== CONSISTENCY SCORE ======
        # I paper nel cluster sono coerenti tra loro?
        consistency_score = self._compute_consistency_score(cluster)
        
        # ====== OVERALL SCORE ======
        # Combinazione pesata
        overall_score = (
            0.3 * size_score +
            0.4 * feature_score +
            0.3 * consistency_score
        )
        
        return {
            'cluster_idx': cluster_idx,
            'cluster_size': cluster_size,
            'papers': cluster,
            'scores': {
                'overall': overall_score,
                'size': size_score,
                'feature': feature_score,
                'consistency': consistency_score
            }
        }
    
    def _compute_size_score(self, size):
        """
        Score basato su dimensione cluster
        
        Logica:
        - size = 1:     0.3 (molto incerto)
        - size = 2-3:   0.5 (incerto)
        - size = 4-10:  0.8 (buono)
        - size = 11-20: 0.9 (ottimo)
        - size > 20:    0.7 (possibile lumping)
        """
        if size == 1:
            return 0.3
        elif size <= 3:
            return 0.5
        elif size <= 10:
            return 0.8
        elif size <= 20:
            return 0.9
        else:
            # Penalizza cluster molto grandi (possibile lumping)
            return max(0.5, 0.9 - 0.02 * (size - 20))
    
    def _compute_feature_score(self, name, cluster):
        """
        Score basato su features disponibili
        
        Se hai accesso ai paper data, puoi calcolare:
        - Citation overlap
        - Coauthor overlap
        - Venue consistency
        - Temporal consistency
        
        Senza paper data, usa euristica basata su size
        """
        if self.pubs_data is None:
            # Fallback: assume correlazione con size
            return min(1.0, 0.5 + 0.1 * len(cluster))
        
        # TODO: Implementa con paper data reali
        # Esempio:
        # - Calcola quante citazioni condivise
        # - Calcola overlap coautori
        # - Controlla venue consistency
        
        return 0.7  # Placeholder
    
    def _compute_consistency_score(self, cluster):
        """
        Score basato su consistenza interna
        
        Senza paper data, usa euristica:
        - Cluster piccoli sono più consistenti
        - Cluster grandi richiedono validazione
        """
        size = len(cluster)
        
        if size == 1:
            return 1.0  # Trivialmente consistente
        elif size <= 5:
            return 0.9
        elif size <= 10:
            return 0.8
        else:
            return 0.7
    
    def filter_clusters(self, min_confidence=0.6):
        """
        Filtra cluster sotto una soglia di confidenza
        
        Args:
            min_confidence: soglia minima (0-1)
        
        Returns:
            filtered_predictions: solo cluster affidabili
            rejected_clusters: cluster scartati
        """
        if not self.cluster_scores:
            self.compute_all_scores()
        
        filtered_predictions = {}
        rejected_clusters = {}
        stats = {
            'total_clusters': 0,
            'accepted_clusters': 0,
            'rejected_clusters': 0,
            'total_papers': 0,
            'accepted_papers': 0,
            'rejected_papers': 0
        }
        
        for name, clusters_scores in self.cluster_scores.items():
            accepted = []
            rejected = []
            
            for score_data in clusters_scores:
                overall_score = score_data['scores']['overall']
                cluster_size = score_data['cluster_size']
                
                stats['total_clusters'] += 1
                stats['total_papers'] += cluster_size
                
                if overall_score >= min_confidence:
                    accepted.append(score_data['papers'])
                    stats['accepted_clusters'] += 1
                    stats['accepted_papers'] += cluster_size
                else:
                    rejected.append(score_data)
                    stats['rejected_clusters'] += 1
                    stats['rejected_papers'] += cluster_size
            
            if accepted:
                filtered_predictions[name] = accepted
            if rejected:
                rejected_clusters[name] = rejected
        
        return filtered_predictions, rejected_clusters, stats
    
    def analyze_confidence_distribution(self):
        """Analizza la distribuzione degli score di confidenza"""
        if not self.cluster_scores:
            self.compute_all_scores()
        
        all_scores = []
        for clusters_scores in self.cluster_scores.values():
            for score_data in clusters_scores:
                all_scores.append(score_data['scores']['overall'])
        
        all_scores = np.array(all_scores)
        
        print("\n" + "="*70)
        print("CONFIDENCE SCORE DISTRIBUTION")
        print("="*70)
        print(f"Total clusters: {len(all_scores)}")
        print(f"Mean confidence: {all_scores.mean():.3f}")
        print(f"Std confidence:  {all_scores.std():.3f}")
        print(f"Min confidence:  {all_scores.min():.3f}")
        print(f"Max confidence:  {all_scores.max():.3f}")
        
        # Percentili
        print("\nPercentiles:")
        for p in [10, 25, 50, 75, 90]:
            print(f"  {p}th: {np.percentile(all_scores, p):.3f}")
        
        # Distribuzione per bin
        print("\nDistribution:")
        bins = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(all_scores, bins=bins)
        
        for i in range(len(bins)-1):
            count = hist[i]
            pct = 100 * count / len(all_scores)
            print(f"  [{bins[i]:.1f} - {bins[i+1]:.1f}): {count:4d} ({pct:5.1f}%)")
        
        return {
            'mean': all_scores.mean(),
            'std': all_scores.std(),
            'min': all_scores.min(),
            'max': all_scores.max(),
            'percentiles': {p: np.percentile(all_scores, p) for p in [10, 25, 50, 75, 90]}
        }
    
    def print_cluster_report(self, name, top_k=10):
        """Stampa report dettagliato per un nome specifico"""
        if name not in self.cluster_scores:
            print(f"Name '{name}' not found")
            return
        
        clusters_scores = self.cluster_scores[name]
        
        # Ordina per overall score (decrescente)
        sorted_clusters = sorted(
            clusters_scores,
            key=lambda x: x['scores']['overall'],
            reverse=True
        )
        
        print("\n" + "="*70)
        print(f"CLUSTER REPORT: {name}")
        print("="*70)
        print(f"Total clusters: {len(sorted_clusters)}")
        
        print(f"\nTop {min(top_k, len(sorted_clusters))} clusters by confidence:")
        
        for i, cluster_data in enumerate(sorted_clusters[:top_k], 1):
            scores = cluster_data['scores']
            size = cluster_data['cluster_size']
            
            print(f"\n{i}. Cluster {cluster_data['cluster_idx']}")
            print(f"   Size: {size} papers")
            print(f"   Overall:     {scores['overall']:.3f}")
            print(f"   Size score:  {scores['size']:.3f}")
            print(f"   Feature:     {scores['feature']:.3f}")
            print(f"   Consistency: {scores['consistency']:.3f}")
            
            # Interpretazione
            if scores['overall'] >= 0.8:
                print(f"   →  HIGH CONFIDENCE - Likely correct")
            elif scores['overall'] >= 0.6:
                print(f"   →  MEDIUM CONFIDENCE - Review recommended")
            else:
                print(f"   →  LOW CONFIDENCE - Likely incorrect")
    
    def recommend_threshold(self):
        """
        Raccomanda una soglia di confidenza ottimale
        
        Logica:
        - Guarda la distribuzione degli score
        - Trova il "gap" naturale
        - Suggerisci soglia che massimizza precision vs recall
        """
        if not self.cluster_scores:
            self.compute_all_scores()
        
        all_scores = []
        for clusters_scores in self.cluster_scores.values():
            for score_data in clusters_scores:
                all_scores.append(score_data['scores']['overall'])
        
        all_scores = np.array(sorted(all_scores))
        
        # Cerca il gap più grande
        gaps = np.diff(all_scores)
        max_gap_idx = np.argmax(gaps)
        
        # Soglia = metà del gap più grande
        recommended_threshold = (all_scores[max_gap_idx] + all_scores[max_gap_idx + 1]) / 2
        
        print("\n" + "="*70)
        print("RECOMMENDED CONFIDENCE THRESHOLD")
        print("="*70)
        print(f"Recommended: {recommended_threshold:.3f}")
        print(f"\nRationale:")
        print(f"  - Largest gap in distribution at {all_scores[max_gap_idx]:.3f}")
        print(f"  - Natural split between high/low confidence clusters")
        
        # Simula filtro
        n_accepted = np.sum(all_scores >= recommended_threshold)
        n_rejected = len(all_scores) - n_accepted
        
        print(f"\nImpact:")
        print(f"  - Clusters accepted: {n_accepted} ({100*n_accepted/len(all_scores):.1f}%)")
        print(f"  - Clusters rejected: {n_rejected} ({100*n_rejected/len(all_scores):.1f}%)")
        
        return recommended_threshold
    
    def save_filtered_results(self, output_file, min_confidence=0.6):
        """Salva predizioni filtrate"""
        filtered, rejected, stats = self.filter_clusters(min_confidence)
        
        # Salva filtrate
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered, f, indent=2, ensure_ascii=False)
        
        # Salva stats
        stats_file = Path(output_file).parent / (Path(output_file).stem + "_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n Filtered results saved to: {output_file}")
        print(f" Statistics saved to: {stats_file}")
        
        print("\n" + "="*70)
        print("FILTERING STATISTICS")
        print("="*70)
        print(f"Min confidence threshold: {min_confidence:.2f}")
        print(f"\nClusters:")
        print(f"  Total:    {stats['total_clusters']}")
        print(f"  Accepted: {stats['accepted_clusters']} ({100*stats['accepted_clusters']/stats['total_clusters']:.1f}%)")
        print(f"  Rejected: {stats['rejected_clusters']} ({100*stats['rejected_clusters']/stats['total_clusters']:.1f}%)")
        print(f"\nPapers:")
        print(f"  Total:    {stats['total_papers']}")
        print(f"  Accepted: {stats['accepted_papers']} ({100*stats['accepted_papers']/stats['total_papers']:.1f}%)")
        print(f"  Rejected: {stats['rejected_papers']} ({100*stats['rejected_papers']/stats['total_papers']:.1f}%)")
        
        return filtered, rejected, stats


def main():
    """Esempio d'uso"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compute confidence scores for predicted clusters'
    )
    parser.add_argument('--predictions', '-p', required=True,
                       help='Path to predictions JSON (res.json)')
    parser.add_argument('--pubs', default=None,
                       help='Path to publications data (optional)')
    parser.add_argument('--output', '-o', default='filtered_predictions.json',
                       help='Output file for filtered predictions')
    parser.add_argument('--threshold', '-t', type=float, default=None,
                       help='Confidence threshold (default: auto-recommend)')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze confidence distribution')
    parser.add_argument('--report', default=None,
                       help='Generate detailed report for specific name')
    
    args = parser.parse_args()
    
    # Crea scorer
    scorer = ClusterConfidenceScorer(args.predictions, args.pubs)
    
    # Calcola scores
    print("Computing confidence scores...")
    scorer.compute_all_scores()
    print("Done!")
    
    # Analizza distribuzione
    if args.analyze:
        scorer.analyze_confidence_distribution()
    
    # Report per nome specifico
    if args.report:
        scorer.print_cluster_report(args.report)
    
    # Raccomanda soglia
    if args.threshold is None:
        recommended = scorer.recommend_threshold()
        
        choice = input("\nUse recommended threshold? (y/n): ").strip().lower()
        if choice == 'y':
            args.threshold = recommended
        else:
            threshold_input = input("Enter threshold [0.0-1.0]: ").strip()
            args.threshold = float(threshold_input) if threshold_input else 0.6
    
    # Filtra e salva
    scorer.save_filtered_results(args.output, args.threshold)


if __name__ == '__main__':
    import sys
    
    # ============================================================
    # Se NON ci sono argomenti command-line, usa path fissi
    # ============================================================
    if len(sys.argv) == 1:
        # 📁 MODIFICA QUESTI PATH
        predictions_file = r"C:\Users\franc\OneDrive - Alma Mater Studiorum Università di Bologna\Desktop\BOND-OC\WhoIsWho\bond\out\res.json"
        output_file = r"C:\Users\franc\OneDrive - Alma Mater Studiorum Università di Bologna\Desktop\BOND-OC\WhoIsWho\bond\out\filtered_predictions.json"
        pubs_file = r"C:\Users\franc\OneDrive - Alma Mater Studiorum Università di Bologna\Desktop\BOND-OC\WhoIsWho\bond\dataset\data\src\sna-valid\sna_valid_pub.json"
        
        # ⚙️ CONFIGURAZIONE
        analyze_distribution = True
        auto_threshold = True
        manual_threshold = 0.6
        
        # ESECUZIONE
        print("\n" + "="*70)
        print(" CLUSTER CONFIDENCE SCORING (Automatic Mode)")
        print("="*70)
        print(f"\nInput:  {predictions_file}")
        print(f"Output: {output_file}")
        
        scorer = ClusterConfidenceScorer(predictions_file, pubs_file)
        
        print("\n📊 Computing confidence scores...")
        scorer.compute_all_scores()
        print(" Scores computed!")
        
        if analyze_distribution:
            scorer.analyze_confidence_distribution()
        
        if auto_threshold:
            print("\n Computing recommended threshold...")
            threshold = scorer.recommend_threshold()
        else:
            threshold = manual_threshold
            print(f"\n  Using manual threshold: {threshold:.2f}")
        
        print(f"\n Filtering clusters with threshold: {threshold:.2f}")
        scorer.save_filtered_results(output_file, threshold)
        
        print("\n" + "="*70)
        print(" COMPLETED!")
        print("="*70)
    else:
        # Se ci sono argomenti, usa la funzione main() originale
        main()
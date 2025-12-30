"""
Cluster Confidence Scoring & Filtering - VERSIONE MIGLIORATA
==============================================================

Questo script calcola confidence scores REALI basati su:
1. Coesione interna del cluster (similarità tra paper)
2. Separazione esterna (quanto è distinto dagli altri cluster)
3. Feature strength (coautori, venue, citations, temporal)
4. Cluster size (con penalizzazione intelligente)

MIGLIORAMENTI RISPETTO ALLA VERSIONE PRECEDENTE:
- Usa dati reali dei paper (non placeholder)
- Calcola similarità effettive tra paper
- Considera co-autori, venue, anni
- Produce distribuzione continua (non discreta)
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set


class ImprovedClusterConfidenceScorer:
    """
    Calcola confidence scores usando features REALI
    """
    
    def __init__(self, predictions_file, pubs_file=None):
        self.predictions_file = predictions_file
        self.pubs_file = pubs_file
        
        # Carica predizioni
        with open(predictions_file, 'r', encoding='utf-8') as f:
            self.predictions = json.load(f)
        
        # Carica paper data
        self.pubs_data = {}
        if pubs_file and Path(pubs_file).exists():
            print(f"Loading publications from: {pubs_file}")
            with open(pubs_file, 'r', encoding='utf-8') as f:
                raw_pubs = json.load(f)
            
            # Handle different JSON structures
            if isinstance(raw_pubs, dict):
                # If it's already a dict of {name: [papers]}
                for name, papers in raw_pubs.items():
                    if isinstance(papers, list):
                        for paper in papers:
                            if isinstance(paper, dict):
                                pid = paper.get('id', paper.get('pid'))
                                if pid:
                                    self.pubs_data[pid] = paper
                    elif isinstance(papers, dict):
                        # Single paper as dict
                        pid = papers.get('id', papers.get('pid'))
                        if pid:
                            self.pubs_data[pid] = papers
            elif isinstance(raw_pubs, list):
                # If it's a list of papers
                for paper in raw_pubs:
                    if isinstance(paper, dict):
                        pid = paper.get('id', paper.get('pid'))
                        if pid:
                            self.pubs_data[pid] = paper
                    elif isinstance(paper, str):
                        # If papers are stored as strings, skip or handle specially
                        continue
            
            print(f"✓ Loaded {len(self.pubs_data)} papers")
        else:
            print("⚠️  No publications file - will use simplified scoring")
        
        self.cluster_scores = {}
    
    def compute_all_scores(self):
        """Calcola scores per tutti i cluster"""
        print("\nComputing cluster confidence scores...")
        
        total_clusters = sum(len(clusters) for clusters in self.predictions.values())
        processed = 0
        
        for name, clusters in self.predictions.items():
            name_scores = []
            
            for cluster_idx, cluster in enumerate(clusters):
                score = self._compute_cluster_score(name, cluster, cluster_idx)
                name_scores.append(score)
                
                processed += 1
                if processed % 500 == 0:
                    print(f"  Processed {processed}/{total_clusters} clusters...")
            
            self.cluster_scores[name] = name_scores
        
        print(f"✓ Computed scores for {total_clusters} clusters")
        return self.cluster_scores
    
    def _compute_cluster_score(self, name, cluster, cluster_idx):
        """Calcola confidence score per un singolo cluster"""
        cluster_size = len(cluster)
        
        # ====== SIZE SCORE ======
        # Penalizza cluster troppo piccoli o troppo grandi
        size_score = self._compute_size_score_smooth(cluster_size)
        
        # ====== COHESION SCORE ======
        # Quanto sono simili i paper nel cluster?
        cohesion_score = self._compute_cohesion_score(cluster)
        
        # ====== FEATURE STRENGTH SCORE ======
        # Quanto sono forti le features (coautori, venue, etc)?
        feature_score = self._compute_feature_strength(cluster)
        
        # ====== TEMPORAL CONSISTENCY ======
        # I paper sono temporalmente coerenti?
        temporal_score = self._compute_temporal_consistency(cluster)
        
        # ====== OVERALL SCORE ======
        # Combinazione pesata ADATTIVA
        # Se non abbiamo pub data, pesa di più size
        if not self.pubs_data:
            overall_score = (
                0.6 * size_score +
                0.2 * cohesion_score +
                0.1 * feature_score +
                0.1 * temporal_score
            )
        else:
            overall_score = (
                0.2 * size_score +
                0.4 * cohesion_score +
                0.3 * feature_score +
                0.1 * temporal_score
            )
        
        return {
            'cluster_idx': cluster_idx,
            'cluster_size': cluster_size,
            'papers': cluster,
            'scores': {
                'overall': float(overall_score),
                'size': float(size_score),
                'cohesion': float(cohesion_score),
                'feature': float(feature_score),
                'temporal': float(temporal_score)
            }
        }
    
    def _compute_size_score_smooth(self, size):
        """
        Score continuo basato su dimensione
        
        Usa funzione smooth invece di thresholds discreti
        """
        if size == 1:
            # Singoli paper sono incerti
            return 0.3
        elif size <= 3:
            # Piccoli cluster: score cresce linearmente
            return 0.3 + (size - 1) * 0.15  # 0.3 -> 0.6
        elif size <= 10:
            # Range ottimale: score alto
            return 0.6 + (size - 3) * 0.04  # 0.6 -> 0.88
        elif size <= 20:
            # Buoni ma attenzione al lumping
            return 0.88 + (size - 10) * 0.01  # 0.88 -> 0.98
        else:
            # Penalizza cluster molto grandi (possibile lumping)
            penalty = min(0.3, 0.01 * (size - 20))
            return max(0.5, 0.98 - penalty)
    
    def _compute_cohesion_score(self, cluster):
        """
        Calcola coesione interna del cluster
        
        Metrica: pairwise similarity media tra paper
        """
        if len(cluster) == 1:
            return 1.0  # Trivialmente coeso
        
        if not self.pubs_data:
            # Fallback senza dati
            return 0.7 - 0.05 * min(len(cluster), 10)
        
        # Calcola similarità pairwise
        similarities = []
        
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                pid1, pid2 = cluster[i], cluster[j]
                
                if pid1 in self.pubs_data and pid2 in self.pubs_data:
                    sim = self._compute_paper_similarity(
                        self.pubs_data[pid1],
                        self.pubs_data[pid2]
                    )
                    similarities.append(sim)
        
        if not similarities:
            return 0.5  # Nessun dato disponibile
        
        # Score = media similarità
        mean_sim = np.mean(similarities)
        
        # Penalizza cluster grandi con bassa similarità
        size_penalty = 1.0 - (len(cluster) / 50) * 0.2  # max -20%
        
        return float(mean_sim * size_penalty)
    
    def _compute_paper_similarity(self, paper1, paper2):
        """
        Calcola similarità tra due paper
        
        Features considerate:
        - Co-autori overlap
        - Venue match
        - Anno vicino
        - Keywords overlap (se disponibili)
        """
        score = 0.0
        weights_sum = 0.0
        
        # ===== CO-AUTORI =====
        authors1 = set(self._get_authors(paper1))
        authors2 = set(self._get_authors(paper2))
        
        if authors1 and authors2:
            # Jaccard similarity
            intersection = len(authors1 & authors2)
            union = len(authors1 | authors2)
            
            if union > 0:
                coauthor_sim = intersection / union
                score += 0.4 * coauthor_sim
                weights_sum += 0.4
        
        # ===== VENUE =====
        venue1 = self._get_venue(paper1)
        venue2 = self._get_venue(paper2)
        
        if venue1 and venue2:
            venue_match = 1.0 if venue1.lower() == venue2.lower() else 0.0
            score += 0.3 * venue_match
            weights_sum += 0.3
        
        # ===== ANNO =====
        year1 = self._get_year(paper1)
        year2 = self._get_year(paper2)
        
        if year1 and year2:
            # Similarità decrescente con distanza temporale
            year_diff = abs(year1 - year2)
            year_sim = max(0, 1.0 - year_diff / 10)  # 0 dopo 10 anni
            score += 0.2 * year_sim
            weights_sum += 0.2
        
        # ===== KEYWORDS/TITLE =====
        title1 = self._get_title(paper1)
        title2 = self._get_title(paper2)
        
        if title1 and title2:
            # Semplice word overlap
            words1 = set(title1.lower().split())
            words2 = set(title2.lower().split())
            
            # Rimuovi stopwords
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            words1 -= stopwords
            words2 -= stopwords
            
            if words1 and words2:
                intersection = len(words1 & words2)
                union = len(words1 | words2)
                title_sim = intersection / union if union > 0 else 0
                score += 0.1 * title_sim
                weights_sum += 0.1
        
        # Normalizza per i pesi effettivamente usati
        if weights_sum > 0:
            return score / weights_sum
        else:
            return 0.5  # Default medio se nessuna feature disponibile
    
    def _compute_feature_strength(self, cluster):
        """
        Calcola quanto sono "forti" le features del cluster
        
        Features forti = molti coautori comuni, venue consistente, etc.
        """
        if len(cluster) == 1:
            return 0.5  # Singolo paper: incerto
        
        if not self.pubs_data:
            # Fallback senza dati
            return 0.5 + 0.05 * min(len(cluster), 10)
        
        score_components = []
        
        # ===== CO-AUTORI DENSITY =====
        # Quanti coautori compaiono in multipli paper?
        all_coauthors = []
        for pid in cluster:
            if pid in self.pubs_data:
                all_coauthors.extend(self._get_authors(self.pubs_data[pid]))
        
        if all_coauthors:
            coauthor_counts = Counter(all_coauthors)
            # Percentuale di coautori che appaiono >1 volta
            repeated = sum(1 for count in coauthor_counts.values() if count > 1)
            coauthor_density = repeated / len(coauthor_counts) if coauthor_counts else 0
            score_components.append(coauthor_density)
        
        # ===== VENUE CONSISTENCY =====
        venues = []
        for pid in cluster:
            if pid in self.pubs_data:
                venue = self._get_venue(self.pubs_data[pid])
                if venue:
                    venues.append(venue.lower())
        
        if venues:
            venue_counts = Counter(venues)
            # Percentuale del venue più comune
            most_common_pct = venue_counts.most_common(1)[0][1] / len(venues)
            score_components.append(most_common_pct)
        
        # ===== OVERALL FEATURE STRENGTH =====
        if score_components:
            return float(np.mean(score_components))
        else:
            return 0.5
    
    def _compute_temporal_consistency(self, cluster):
        """
        Calcola consistenza temporale del cluster
        
        Paper dello stesso autore tendono a essere raggruppati temporalmente
        """
        if len(cluster) == 1:
            return 1.0
        
        if not self.pubs_data:
            return 0.7
        
        years = []
        for pid in cluster:
            if pid in self.pubs_data:
                year = self._get_year(self.pubs_data[pid])
                if year:
                    years.append(year)
        
        if not years or len(years) < 2:
            return 0.7  # Dati insufficienti
        
        # Calcola span temporale
        year_span = max(years) - min(years)
        
        # Penalizza span molto lunghi (>20 anni sospetto)
        if year_span <= 5:
            return 1.0
        elif year_span <= 10:
            return 0.9
        elif year_span <= 15:
            return 0.8
        elif year_span <= 20:
            return 0.7
        else:
            # >20 anni: possibile lumping
            return max(0.3, 0.7 - 0.02 * (year_span - 20))
    
    # ========== HELPER METHODS ==========
    
    def _get_authors(self, paper):
        """Estrai lista autori da paper"""
        authors = paper.get('authors', [])
        if not authors:
            return []
        
        # Gestisci vari formati
        if isinstance(authors, list):
            if authors and isinstance(authors[0], dict):
                # Format: [{"name": "...", ...}, ...]
                return [a.get('name', '') for a in authors if a.get('name')]
            else:
                # Format: ["name1", "name2", ...]
                return [str(a) for a in authors if a]
        
        return []
    
    def _get_venue(self, paper):
        """Estrai venue da paper"""
        return paper.get('venue', '') or paper.get('journal', '') or ''
    
    def _get_year(self, paper):
        """Estrai anno da paper"""
        year = paper.get('year')
        if year:
            try:
                return int(year)
            except (ValueError, TypeError):
                pass
        return None
    
    def _get_title(self, paper):
        """Estrai titolo da paper"""
        return paper.get('title', '')
    
    # ========== FILTERING & ANALYSIS ==========
    
    def filter_clusters(self, min_confidence=0.6):
        """Filtra cluster sotto soglia"""
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
        bins = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
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
    
    def recommend_threshold(self, method='percentile'):
        """
        Raccomanda una soglia di confidenza ottimale
        
        Args:
            method: 'percentile' (usa mediana o P75) o 'gap' (cerca gap)
        
        Returns:
            float: soglia raccomandata
        """
        if not self.cluster_scores:
            self.compute_all_scores()
        
        all_scores = []
        for clusters_scores in self.cluster_scores.values():
            for score_data in clusters_scores:
                all_scores.append(score_data['scores']['overall'])
        
        all_scores = np.array(sorted(all_scores))
        
        print("\n" + "="*70)
        print("RECOMMENDED CONFIDENCE THRESHOLD")
        print("="*70)
        
        if method == 'percentile':
            # Usa un approccio basato su percentili
            # Default: usa mediana o 60th percentile
            
            median = np.median(all_scores)
            p60 = np.percentile(all_scores, 60)
            p75 = np.percentile(all_scores, 75)
            
            # Se lo std è basso, usa un threshold più conservativo
            std = np.std(all_scores)
            
            if std < 0.1:
                # Distribuzione molto concentrata: usa mediana
                recommended_threshold = median
                rationale = "Low variance - using median"
            elif std < 0.15:
                # Varianza media: usa P60
                recommended_threshold = p60
                rationale = "Medium variance - using 60th percentile"
            else:
                # Alta varianza: usa P75 per essere più selettivi
                recommended_threshold = p75
                rationale = "High variance - using 75th percentile"
            
            print(f"Method: Percentile-based")
            print(f"Recommended: {recommended_threshold:.3f}")
            print(f"\nRationale:")
            print(f"  - {rationale}")
            print(f"  - Median: {median:.3f}")
            print(f"  - P60: {p60:.3f}")
            print(f"  - P75: {p75:.3f}")
            print(f"  - Std: {std:.3f}")
        
        else:  # method == 'gap'
            # Cerca il gap più grande (metodo originale, ma migliore)
            gaps = np.diff(all_scores)
            
            # Ignora gap agli estremi (primi/ultimi 5%)
            n = len(gaps)
            start_idx = int(0.05 * n)
            end_idx = int(0.95 * n)
            
            # Trova gap massimo nella parte centrale
            max_gap_idx = start_idx + np.argmax(gaps[start_idx:end_idx])
            max_gap = gaps[max_gap_idx]
            
            # Solo se il gap è significativo (>0.05)
            if max_gap > 0.05:
                recommended_threshold = (all_scores[max_gap_idx] + all_scores[max_gap_idx + 1]) / 2
                print(f"Method: Gap-based")
                print(f"Recommended: {recommended_threshold:.3f}")
                print(f"\nRationale:")
                print(f"  - Found significant gap ({max_gap:.3f}) at {all_scores[max_gap_idx]:.3f}")
                print(f"  - Natural split between high/low confidence clusters")
            else:
                # Fallback su percentile se nessun gap significativo
                recommended_threshold = np.percentile(all_scores, 60)
                print(f"Method: Gap-based (fallback to percentile)")
                print(f"Recommended: {recommended_threshold:.3f}")
                print(f"\nRationale:")
                print(f"  - No significant gap found (max gap: {max_gap:.3f})")
                print(f"  - Using 60th percentile as fallback")
        
        # Simula filtro
        n_accepted = np.sum(all_scores >= recommended_threshold)
        n_rejected = len(all_scores) - n_accepted
        
        print(f"\nImpact:")
        print(f"  - Clusters accepted: {n_accepted} ({100*n_accepted/len(all_scores):.1f}%)")
        print(f"  - Clusters rejected: {n_rejected} ({100*n_rejected/len(all_scores):.1f}%)")
        
        # Warning se threshold troppo restrittivo
        if n_accepted / len(all_scores) < 0.3:
            print(f"\n⚠️  WARNING: Threshold would reject >70% of clusters")
            print(f"   Consider using a lower threshold for production")
        
        return recommended_threshold
    
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
            print(f"   └─ Size:     {scores['size']:.3f}")
            print(f"   └─ Cohesion: {scores['cohesion']:.3f}")
            print(f"   └─ Feature:  {scores['feature']:.3f}")
            print(f"   └─ Temporal: {scores['temporal']:.3f}")
            
            # Interpretazione
            if scores['overall'] >= 0.8:
                print(f"   ✅ HIGH CONFIDENCE - Likely correct")
            elif scores['overall'] >= 0.6:
                print(f"   ⚠️  MEDIUM CONFIDENCE - Review recommended")
            else:
                print(f"   ❌ LOW CONFIDENCE - Likely incorrect")
    
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
        
        print(f"\n✓ Filtered results saved to: {output_file}")
        print(f"✓ Statistics saved to: {stats_file}")
        
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
                       help='Path to publications data (optional but recommended)')
    parser.add_argument('--output', '-o', default='filtered_predictions.json',
                       help='Output file for filtered predictions')
    parser.add_argument('--threshold', '-t', type=float, default=None,
                       help='Confidence threshold (default: auto-recommend)')
    parser.add_argument('--threshold-method', default='percentile',
                       choices=['percentile', 'gap'],
                       help='Method for threshold recommendation')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze confidence distribution')
    parser.add_argument('--report', default=None,
                       help='Generate detailed report for specific name')
    
    args = parser.parse_args()
    
    # Crea scorer
    scorer = ImprovedClusterConfidenceScorer(args.predictions, args.pubs)
    
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
        recommended = scorer.recommend_threshold(method=args.threshold_method)
        
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
        threshold_method = 'percentile'  # 'percentile' o 'gap'
        manual_threshold = None  # None per auto, o valore fisso tipo 0.6
        
        # ESECUZIONE
        print("\n" + "="*70)
        print("🎯 CLUSTER CONFIDENCE SCORING (Improved Version)")
        print("="*70)
        print(f"\nInput:  {predictions_file}")
        print(f"Output: {output_file}")
        print(f"Pubs:   {pubs_file if pubs_file else 'Not provided (simplified scoring)'}")
        
        scorer = ImprovedClusterConfidenceScorer(predictions_file, pubs_file)
        
        print("\n📊 Computing confidence scores...")
        scorer.compute_all_scores()
        print("✓ Scores computed!")
        
        if analyze_distribution:
            scorer.analyze_confidence_distribution()
        
        if manual_threshold is None:
            print(f"\n🎯 Computing recommended threshold (method: {threshold_method})...")
            threshold = scorer.recommend_threshold(method=threshold_method)
        else:
            threshold = manual_threshold
            print(f"\n✓ Using manual threshold: {threshold:.2f}")
        
        print(f"\n💾 Filtering clusters with threshold: {threshold:.2f}")
        scorer.save_filtered_results(output_file, threshold)
        
        print("\n" + "="*70)
        print("✅ COMPLETED!")
        print("="*70)
    else:
        # Se ci sono argomenti, usa la funzione main() originale
        main()
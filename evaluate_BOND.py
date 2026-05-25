import json
from pathlib import Path
from datetime import datetime
import pandas as pd


# =========================================================
# IO
# =========================================================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =========================================================
# EVALUATOR
# =========================================================
class MultiMetricEvaluator:

    def __init__(self, predictions_file, ground_truth_file, verbose=True):
        self.predictions_file = predictions_file
        self.ground_truth_file = ground_truth_file
        self.verbose = verbose
        self.results = {}

    # =====================================================
    # MAIN
    # =====================================================
    def evaluate(self):
        """
        Esegue la valutazione multi-metrica allineando dinamicamente predizioni e GT.
        Nel caso di dataset OC, valuta solo i paper effettivamente recuperati,
        evitando di penalizzare la Recall per dati mancanti nel database.
        """
        pred = load_json(self.predictions_file)
        truth = load_json(self.ground_truth_file)

        global_stats = {
            "tp": 0, "fp": 0, "fn": 0,
            "b3p": 0.0, "b3r": 0.0,
            # FIX: LE e SE ora accumulano valori già normalizzati [0,1] per nome,
            # non somme di istanze. Vengono divisi per num_names in _finalize,
            # coerentemente con la definizione cluster-level del paper.
            "le": 0.0, "se": 0.0,
            "n": 0,
            "num_names": 0  # FIX: contatore nomi per normalizzare LE/SE
        }

        # Trova i nomi comuni tra predizioni e GT
        common_names = [k for k in pred if k in truth]

        if not common_names:
            self.results = self._empty()
            self.stats = {"total_names": 0, "total_instances": 0, "perfect_matches": 0}
            return self.results

        perfect_matches = 0

        for name in common_names:
            # 1. Isola i paper presenti nelle predizioni (set di paper presenti in OC)
            all_pred_papers = set(x for cluster in pred[name] for x in cluster)

            # 2. Filtra la Ground Truth originale per includere SOLO i paper presenti in OC.
            # Questo garantisce che la valutazione sia equa e basata sui dati disponibili.
            raw_truth_clusters = self._prepare_truth(truth[name])
            filtered_truth = []
            for t_cluster in raw_truth_clusters:
                new_cluster = [p for p in t_cluster if p in all_pred_papers]
                if new_cluster:
                    filtered_truth.append(new_cluster)

            # 3. Esegue la valutazione sul subset allineato
            stats = self._evaluate_one(pred[name], filtered_truth)

            # 4. Accumulo globale
            self._accumulate(global_stats, stats)

            # Conteggio "Perfect Matches" (nessun lumping, nessun splitting)
            if stats["n"] > 0 and stats["le"] == 0.0 and stats["se"] == 0.0:
                perfect_matches += 1

        self.stats = {
            "total_names": len(common_names),
            "total_instances": global_stats["n"],
            "perfect_matches": perfect_matches,
            "accuracy_names": perfect_matches / len(common_names) if common_names else 0
        }

        self.results = self._finalize(global_stats)
        self.results["composite_score"] = self._composite(self.results)

        if self.verbose:
            self.print_results(detailed=False)

        return self.results

    # =====================================================
    # SINGLE SAMPLE
    # =====================================================
    def _evaluate_one(self, pred_clusters, truth_data):

        truth_clusters = self._prepare_truth(truth_data)

        if not pred_clusters or not truth_clusters:
            return self._empty_one()

        # -------------------------
        # INDEXING
        # -------------------------
        p_index = {}  # paper -> predicted cluster index
        t_index = {}  # paper -> ground-truth cluster index

        for i, c in enumerate(pred_clusters):
            for x in c:
                p_index[x] = i

        for i, c in enumerate(truth_clusters):
            for x in c:
                t_index[x] = i

        # =====================================================
        # PAIRWISE
        # =====================================================
        tp = fp = fn = 0

        for pc in pred_clusters:
            n = len(pc)
            tp_local = 0
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = pc[i], pc[j]
                    if a in t_index and b in t_index and t_index[a] == t_index[b]:
                        tp_local += 1
            total = n * (n - 1) // 2
            tp += tp_local
            fp += total - tp_local

        for tc in truth_clusters:
            n = len(tc)
            tp_local = 0
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = tc[i], tc[j]
                    if a in p_index and b in p_index and p_index[a] == p_index[b]:
                        tp_local += 1
            total = n * (n - 1) // 2
            fn += total - tp_local

        # =====================================================
        # B³ (instance-level, invariata)
        # =====================================================
        b3p_sum = 0.0
        b3r_sum = 0.0

        all_instances = set(p_index.keys()) | set(t_index.keys())

        for x in all_instances:
            if x not in p_index or x not in t_index:
                continue
            pc = pred_clusters[p_index[x]]
            tc = truth_clusters[t_index[x]]
            inter = len(set(pc) & set(tc))
            b3p_sum += inter / len(pc)
            b3r_sum += inter / len(tc)

        n_instances = len(all_instances)

        # =====================================================
        # FIX — LE e SE: formulazione cluster-level
        #
        # LE = (1/|C|) * sum_{C in C} max(0, g(C)-1) / g(C)
        #   dove g(C) = numero di ground-truth cluster distinti dentro C.
        #   LE=0 iff ogni cluster predetto è puro; LE->1 iff ogni cluster
        #   predetto mescola molti autori distinti.
        #
        # SE = (1/|T|) * sum_{T in T} max(0, p(T)-1) / p(T)
        #   dove p(T) = numero di cluster predetti che intersecano T.
        #   SE=0 iff ogni autore GT è interamente in un solo cluster;
        #   SE->1 iff ogni autore GT è frammentato su molti cluster.
        #
        # Entrambe le quantità sono già in [0,1] per costruzione.
        # Non vengono sommate su istanze ma su cluster, quindi NON
        # vanno divise per n_instances in _finalize.
        # =====================================================

        # Lumping Error (cluster-level)
        le_sum = 0.0
        for pc in pred_clusters:
            gt_ids = set(t_index[x] for x in pc if x in t_index)
            g = len(gt_ids)
            if g > 0:
                le_sum += max(0, g - 1) / g
        le = le_sum / len(pred_clusters) if pred_clusters else 0.0

        # Splitting Error (cluster-level)
        se_sum = 0.0
        for tc in truth_clusters:
            pred_ids = set(p_index[x] for x in tc if x in p_index)
            p = len(pred_ids)
            if p > 0:
                se_sum += max(0, p - 1) / p
        se = se_sum / len(truth_clusters) if truth_clusters else 0.0

        return {
            "tp": tp, "fp": fp, "fn": fn,
            "b3p": b3p_sum,
            "b3r": b3r_sum,
            "le": le,       # già normalizzato in [0,1]
            "se": se,       # già normalizzato in [0,1]
            "n": n_instances,
            "num_names": 1  # ogni chiamata corrisponde a un nome
        }

    # =====================================================
    # FINAL METRICS
    # =====================================================
    def _finalize(self, g):
        tp, fp, fn = g["tp"], g["fp"], g["fn"]

        # 1. PAIRWISE
        pw_p = tp / (tp + fp) if (tp + fp) else 0
        pw_r = tp / (tp + fn) if (tp + fn) else 0
        pw_f1 = (2 * pw_p * pw_r / (pw_p + pw_r)) if (pw_p + pw_r) else 0

        n = g["n"] if g["n"] else 1

        # 2. B³ (instance-level: divide per n_instances, invariato)
        b3_p = g["b3p"] / n
        b3_r = g["b3r"] / n
        b3_f1 = (2 * b3_p * b3_r / (b3_p + b3_r)) if (b3_p + b3_r) else 0

        # 3. K-METRIC
        k_aap = b3_r
        k_acp = b3_p
        k_metric = (k_aap + k_acp) / 2

        # 4. FIX — LE e SE: già normalizzati per nome in _evaluate_one.
        # In _accumulate vengono sommati su tutti i nomi, quindi qui
        # dividiamo per num_names (media tra nomi), non per n_instances.
        num_names = g["num_names"] if g["num_names"] else 1
        le = g["le"] / num_names
        se = g["se"] / num_names

        # S_struct = 1 - (LE + SE) / 2, in [0,1]
        struct_score = 1.0 - (le + se) / 2.0

        return {
            "pairwise": {
                "precision": pw_p, "recall": pw_r, "f1": pw_f1
            },
            "b3": {
                "precision": b3_p, "recall": b3_r, "f1": b3_f1
            },
            "k_metric": {
                "aap": k_aap, "acp": k_acp, "k": k_metric
            },
            "structural": {
                "lumping_error": le,
                "splitting_error": se,
                "score": struct_score
            }
        }

    def _composite(self, r):
        # Formula basata sui pesi del paper: 0.3 PW + 0.5 B³ + 0.2 Struct
        return (
            0.3 * r["pairwise"]["f1"] +
            0.5 * r["b3"]["f1"] +
            0.2 * r["structural"]["score"]
        )

    def print_results(self, detailed=True):
        """Stampa completa di statistiche, metriche Kim (2019) e interpretazione."""
        if not hasattr(self, 'results') or not self.results:
            print("\n❌ Nessun risultato disponibile. Eseguire prima .evaluate()")
            return

        print("\n" + "=" * 70)
        print("📊 MULTI-METRIC EVALUATION RESULTS (Kim 2019)")
        print("=" * 70)

        if hasattr(self, 'stats'):
            print("\n[DATASET STATISTICS]")
            print(f"   Total names:       {self.stats.get('total_names', 'N/A')}")
            print(f"   Total instances:   {self.stats.get('total_instances', 'N/A')}")
            print(f"   Perfect matches:   {self.stats.get('perfect_matches', 'N/A')}")

        pw = self.results.get('pairwise', {})
        print("\n[PAIRWISE-F]")
        print(f"   Precision: {pw.get('precision', 0):.4f}")
        print(f"   Recall:    {pw.get('recall', 0):.4f}")
        print(f"   F1:        {pw.get('f1', 0):.4f}")

        b3 = self.results.get('b3', {})
        print("\n[B³ (B-CUBED)]")
        print(f"   Precision: {b3.get('precision', 0):.4f}")
        print(f"   Recall:    {b3.get('recall', 0):.4f}")
        print(f"   F1:        {b3.get('f1', 0):.4f}")

        sl = self.results.get('structural', {})
        print("\n[STRUCTURAL ERRORS]")
        print(f"   Lumping Error:    {sl.get('lumping_error', 0.0):.4f}")
        print(f"   Splitting Error:  {sl.get('splitting_error', 0.0):.4f}")
        print(f"   Structural Score: {sl.get('score', 0.0):.4f}")

        print("\n[COMPOSITE SCORE]")
        print(f"   Final Score:     {self.results.get('composite_score', 0):.4f}")

        if detailed and 'name_level_stats' in self.results:
            diff = self.results['name_level_stats'].get('difficult', {})
            if diff:
                print("\n[DIFFICULT CASES ANALYSIS]")
                print(f"   Avg clusters/name (Truth): {diff.get('avg_truth_clusters', 0):.2f}")
                print(f"   Avg clusters/name (Pred):  {diff.get('avg_pred_clusters', 0):.2f}")

        print("\n" + "=" * 70)

    # =====================================================
    # HELPERS
    # =====================================================
    def _prepare_truth(self, truth):
        if isinstance(truth, dict):
            return [v for v in truth.values() if v]
        elif isinstance(truth, list):
            return [v for v in truth if isinstance(v, list) and v]
        return []

    def _accumulate(self, g, s):
        # FIX: LE e SE sono già valori scalari [0,1] per nome,
        # vengono sommati direttamente (non pesati per istanze).
        # Tutti gli altri campi si accumulano come prima.
        for k in g:
            g[k] += s[k]

    def _empty_one(self):
        return {
            "tp": 0, "fp": 0, "fn": 0,
            "b3p": 0.0, "b3r": 0.0,
            "le": 0.0, "se": 0.0,
            "n": 0,
            "num_names": 1
        }

    def _empty(self):
        return {
            "pairwise": {"precision": 0, "recall": 0, "f1": 0},
            "b3": {"precision": 0, "recall": 0, "f1": 0},
            "structural": {
                "lumping_error": 1.0,
                "splitting_error": 1.0,
                "score": 0.0
            },
            "composite_score": 0.0
        }

    def save_to_excel(self, base_output_path="reports", experiment_name="Exp", args=None):
        """
        Salva il report Excel nominando il file in base al dataset e all'embedding.
        """
        r = self.results

        emb_type = getattr(args, 'emb_type', 'unknown_emb').upper() if args else "EMB"

        dataset_name = "Dataset"
        if args and hasattr(args, 'save_path'):
            path_str = str(args.save_path).lower()
            if "whoiswho" in path_str:
                dataset_name = "WhoIsWho"
            elif "oc" in path_str or "opencitations" in path_str:
                dataset_name = "OC"

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        filename = f"{emb_type}_{dataset_name}_{experiment_name}_{timestamp}.xlsx"
        full_path = Path(base_output_path) / filename
        full_path.parent.mkdir(parents=True, exist_ok=True)

        pw = r.get("pairwise", {})
        b3 = r.get("b3", {})
        kmet = r.get("k_metric", {})
        struc = r.get("structural", {})
        stats = getattr(self, "stats", {})

        data = {
            "Dataset": dataset_name,
            "Emb_Type": emb_type,
            "Mode": experiment_name,
            "Composite_Score": r.get("composite_score", 0),

            "Pairwise_F1": pw.get("f1", 0),
            "Pairwise_Prec": pw.get("precision", 0),
            "Pairwise_Rec": pw.get("recall", 0),

            "B3_F1": b3.get("f1", 0),
            "B3_Prec": b3.get("precision", 0),
            "B3_Rec": b3.get("recall", 0),

            "K_Metric_K": kmet.get("k", 0),
            "K_Metric_AAP": kmet.get("aap", 0),
            "K_Metric_ACP": kmet.get("acp", 0),

            "Lumping_Error": struc.get("lumping_error", 0),
            "Splitting_Error": struc.get("splitting_error", 0),
            "Structural_Score": struc.get("score", 0),

            "Perfect_Matches": stats.get("perfect_matches", 0),
            "Accuracy_Names": stats.get("accuracy_names", 0),
            "Total_Instances": stats.get("total_instances", 0),
            "Date": now.strftime("%Y-%m-%d %H:%M:%S")
        }

        if args:
            data.update({
                "Db_Eps": getattr(args, 'db_eps', 'N/A'),
                "Db_Min": getattr(args, 'db_min', 'N/A'),
                "Use_Citations": getattr(args, 'use_citations', 'N/A'),
                "Rel_On": getattr(args, 'rel_on', 'N/A')
            })

        try:
            df = pd.DataFrame([data])
            df.to_excel(full_path, index=False)
            print(f"📊 Report Excel generato con successo: {full_path}")
        except Exception as e:
            print(f"⚠️ Errore durante la generazione dell'Excel: {e}")

        return str(full_path)

    def save_results(self, path):
        save_json(self.results, path)


# =========================================================
# USAGE
# =========================================================
if __name__ == "__main__":

    evaluator = MultiMetricEvaluator(
        "predictions.json",
        "ground_truth.json",
        verbose=True
    )

    results = evaluator.evaluate()

    save_json(results, "results.json")

    evaluator.save_to_excel(base_output_path="reports")
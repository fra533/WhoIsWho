import numpy as np
import os
from os.path import join
from tqdm import tqdm
from datetime import datetime
from whoiswho.dataset.data_process import read_pubs, read_raw_pubs
from whoiswho.utils import load_json, save_json

def detailed_debug_single_name(predict_result, ground_truth, name):
    """Debug dettagliato per un singolo nome"""
    print(f"\n{'='*60}")
    print(f"DETAILED DEBUG per {name}")
    print(f"{'='*60}")
    
    # 1. Analisi predizioni
    pred_clusters = predict_result[name]
    print(f"🔮 PREDICTIONS:")
    print(f"  Numero cluster: {len(pred_clusters)}")
    
    pred_papers = set()
    for i, cluster in enumerate(pred_clusters):
        print(f"  Cluster {i}: {len(cluster)} paper")
        pred_papers.update(cluster)
        if i < 3:  # Mostra primi 3 cluster
            print(f"    Paper: {cluster[:5]}{'...' if len(cluster) > 5 else ''}")
    
    print(f"  Totale paper predetti: {len(pred_papers)}")
    
    # 2. Analisi ground truth
    gt_clusters = ground_truth[name]
    print(f"\n🎯 GROUND TRUTH:")
    print(f"  Numero cluster: {len(gt_clusters)}")
    
    gt_papers = set()
    for i, cluster in enumerate(gt_clusters):
        if isinstance(cluster, list):
            print(f"  Cluster {i}: {len(cluster)} paper")
            gt_papers.update(cluster)
            if i < 3:  # Mostra primi 3 cluster
                print(f"    Paper: {cluster[:5]}{'...' if len(cluster) > 5 else ''}")
    
    print(f"  Totale paper GT: {len(gt_papers)}")
    
    # 3. Analisi overlap
    common_papers = pred_papers & gt_papers
    only_pred = pred_papers - gt_papers
    only_gt = gt_papers - pred_papers
    
    print(f"\n🔍 OVERLAP ANALYSIS:")
    print(f"  Paper comuni: {len(common_papers)}")
    print(f"  Solo in predizioni: {len(only_pred)}")
    print(f"  Solo in GT: {len(only_gt)}")
    print(f"  Overlap ratio: {len(common_papers)/max(len(pred_papers), len(gt_papers))*100:.1f}%")
    
    if len(only_pred) > 0:
        print(f"  Esempi solo pred: {list(only_pred)[:3]}")
    if len(only_gt) > 0:
        print(f"  Esempi solo GT: {list(only_gt)[:3]}")
    
    # 4. Se c'è overlap, analizza le label
    if len(common_papers) > 0:
        print(f"\n📊 LABEL ANALYSIS sui paper comuni:")
        
        # Crea mapping per paper comuni
        predicted_pubs = dict()
        for idx, pids in enumerate(pred_clusters):
            for pid in pids:
                if pid in common_papers:
                    predicted_pubs[pid] = idx
        
        # Crea true labels per paper comuni
        true_pubs = dict()
        for idx, cluster in enumerate(gt_clusters):
            if isinstance(cluster, list):
                for pid in cluster:
                    if pid in common_papers:
                        true_pubs[pid] = idx
        
        print(f"  Paper mappati pred: {len(predicted_pubs)}")
        print(f"  Paper mappati GT: {len(true_pubs)}")
        
        # Analizza distribuzione cluster
        pred_labels = list(predicted_pubs.values())
        true_labels = list(true_pubs.values())
        
        print(f"  Pred clusters utilizzati: {sorted(set(pred_labels))}")
        print(f"  GT clusters utilizzati: {sorted(set(true_labels))}")
        
        # Conta paper per cluster
        from collections import Counter
        pred_counts = Counter(pred_labels)
        true_counts = Counter(true_labels)
        
        print(f"  Pred cluster sizes: {dict(pred_counts)}")
        print(f"  True cluster sizes: {dict(true_counts)}")
        
        return len(common_papers)
    else:
        print(f"\n❌ NESSUN PAPER IN COMUNE - IMPOSSIBILE VALUTARE")
        return 0

def debug_clustering_behavior(correct_labels, pred_labels, name):
    """Debug del comportamento del clustering"""
    print(f"\n🧮 CLUSTERING BEHAVIOR per {name}:")
    print(f"  Paper da valutare: {len(correct_labels)}")
    print(f"  Unique GT labels: {sorted(set(correct_labels))}")
    print(f"  Unique pred labels: {sorted(set(pred_labels))}")
    
    # Conta pair types
    total_pairs = 0
    same_author_gt = 0
    same_author_pred = 0
    correct_same = 0
    
    for i in range(len(correct_labels)):
        for j in range(i + 1, len(correct_labels)):
            total_pairs += 1
            
            gt_same = correct_labels[i] == correct_labels[j]
            pred_same = pred_labels[i] == pred_labels[j]
            
            if gt_same:
                same_author_gt += 1
            if pred_same:
                same_author_pred += 1
            if gt_same and pred_same:
                correct_same += 1
    
    print(f"  Totale coppie: {total_pairs}")
    print(f"  Coppie stesso autore (GT): {same_author_gt}")
    print(f"  Coppie stesso autore (Pred): {same_author_pred}")
    print(f"  Coppie corrette stesso autore: {correct_same}")
    
    if same_author_pred > 0:
        precision = correct_same / same_author_pred
        print(f"  Precision: {precision:.4f}")
    else:
        print(f"  Precision: 0 (nessuna coppia predetta insieme)")
        
    if same_author_gt > 0:
        recall = correct_same / same_author_gt
        print(f"  Recall: {recall:.4f}")
    else:
        print(f"  Recall: N/A (nessuna coppia GT insieme)")
    
    # Analizza pattern problematici
    if same_author_pred == 0:
        print(f"  🚨 PROBLEMA: Modello non raggruppa MAI paper insieme!")
    elif same_author_pred == total_pairs:
        print(f"  🚨 PROBLEMA: Modello raggruppa SEMPRE tutto insieme!")
    elif same_author_pred > same_author_gt * 3:
        print(f"  🚨 PROBLEMA: Modello raggruppa troppo (over-clustering)!")

def evaluate_with_debug(predict_result, ground_truth, debug_names=2):
    if isinstance(predict_result, str):
        predict_result = load_json(predict_result)
    if isinstance(ground_truth, str):
        ground_truth = load_json(ground_truth)
    
    # AGGIUNTO: Filtra solo i nomi comuni
    filtered_predict_result = {name: pred for name, pred in predict_result.items() 
                              if name in ground_truth}
    
    print(f"📊 EVALUATION OVERVIEW:")
    print(f"  Nomi in predictions: {len(predict_result)}")
    print(f"  Nomi in ground truth: {len(ground_truth)}")
    print(f"  Nomi comuni: {len(filtered_predict_result)}")
    
    name_nums = 0
    result_list = []
    debug_count = 0
    
    for name in filtered_predict_result:
        # Debug dettagliato per i primi nomi
        if debug_count < debug_names:
            common_papers = detailed_debug_single_name(predict_result, ground_truth, name)
            debug_count += 1
            
            if common_papers == 0:
                print(f"⏭️  Skipping {name} - no common papers")
                continue
        
        # Get clustering labels in predict_result
        predicted_pubs = dict()
        for idx, pids in enumerate(filtered_predict_result[name]):
            for pid in pids:
                predicted_pubs[pid] = idx
        
        # Ground truth processing
        pubs = []
        ilabel = 0
        true_labels = []
        
        if isinstance(ground_truth[name], dict):
            for aid in ground_truth[name]:
                pubs.extend(ground_truth[name][aid])
                true_labels.extend([ilabel] * len(ground_truth[name][aid]))
                ilabel += 1
        elif isinstance(ground_truth[name], list):
            for cluster in ground_truth[name]:
                if isinstance(cluster, list):
                    pubs.extend(cluster)
                    true_labels.extend([ilabel] * len(cluster))
                    ilabel += 1
                else:
                    pubs.append(cluster)
                    true_labels.append(ilabel)
                    ilabel += 1
        else:
            print(f"Warning: Unknown format for {name}: {type(ground_truth[name])}")
            continue
        
        # Filter common papers
        filtered_pubs = [pid for pid in pubs if pid in predicted_pubs]
        filtered_true_labels = [true_labels[i] for i, pid in enumerate(pubs) if pid in predicted_pubs]
        
        if len(filtered_pubs) == 0:
            print(f"Warning: No common papers for {name}")
            continue
        
        predict_labels = []
        for pid in filtered_pubs:
            predict_labels.append(predicted_pubs[pid])
        
        # Debug clustering behavior per i primi nomi
        if debug_count <= debug_names:
            debug_clustering_behavior(filtered_true_labels, predict_labels, name)
        
        pairwise_precision, pairwise_recall, pairwise_f1 = pairwise_evaluate(filtered_true_labels, predict_labels)
        result_list.append((pairwise_precision, pairwise_recall, pairwise_f1))
        name_nums += 1
        
        # Mostra risultati per i primi nomi
        if debug_count <= debug_names:
            print(f"  📈 Metriche per {name}: P={pairwise_precision:.3f}, R={pairwise_recall:.3f}, F1={pairwise_f1:.3f}")
    
    if name_nums == 0:
        print("Error: No names could be evaluated!")
        return 0.0
    
    avg_pairwise_f1 = sum([result[2] for result in result_list]) / name_nums
    avg_precision = sum([result[0] for result in result_list]) / name_nums
    avg_recall = sum([result[1] for result in result_list]) / name_nums
    
    print(f"\n{'='*50}")
    print(f"📊 FINAL RESULTS:")
    print(f"Average Pairwise Precision: {avg_precision:.3f}")
    print(f"Average Pairwise Recall: {avg_recall:.3f}")
    print(f"Average Pairwise F1: {avg_pairwise_f1:.3f}")
    print(f"Names evaluated: {name_nums}")
    print(f"{'='*50}")
    
    return avg_pairwise_f1

def pairwise_evaluate(correct_labels, pred_labels):
    TP = 0.0  # Pairs Correctly Predicted To SameAuthor
    TP_FP = 0.0  # Total Pairs Predicted To SameAuthor
    TP_FN = 0.0  # Total Pairs To SameAuthor

    for i in range(len(correct_labels)):
        for j in range(i + 1, len(correct_labels)):
            if correct_labels[i] == correct_labels[j]:
                TP_FN += 1
            if pred_labels[i] == pred_labels[j]:
                TP_FP += 1
            if (correct_labels[i] == correct_labels[j]) and (pred_labels[i] == pred_labels[j]):
                TP += 1

    if TP == 0:
        pairwise_precision = 0
        pairwise_recall = 0
        pairwise_f1 = 0
    else:
        pairwise_precision = TP / TP_FP if TP_FP > 0 else 0
        pairwise_recall = TP / TP_FN if TP_FN > 0 else 0
        pairwise_f1 = (2 * pairwise_precision * pairwise_recall) / (pairwise_precision + pairwise_recall)

    return pairwise_precision, pairwise_recall, pairwise_f1

def debug_data_formats(predict_result, ground_truth, max_names=3):
    """Debug function per capire i formati dei dati"""
    if isinstance(predict_result, str):
        predict_result = load_json(predict_result)
    if isinstance(ground_truth, str):
        ground_truth = load_json(ground_truth)
    
    print("=== DEBUG DATA FORMATS ===")
    
    # Controlla nomi comuni
    common_names = set(predict_result.keys()) & set(ground_truth.keys())
    print(f"Common names: {len(common_names)}")
    
    if len(common_names) == 0:
        print("ERROR: No common names!")
        print(f"Predict names sample: {list(predict_result.keys())[:5]}")
        print(f"GT names sample: {list(ground_truth.keys())[:5]}")
        return
    
    # Analizza alcuni nomi
    for i, name in enumerate(list(common_names)[:max_names]):
        print(f"\n--- {name} ---")
        print(f"Predict format: {len(predict_result[name])} clusters")
        print(f"GT format: {type(ground_truth[name])}")
        
        if isinstance(ground_truth[name], dict):
            print(f"  GT dict keys: {len(ground_truth[name])}")
            print(f"  GT sample: {dict(list(ground_truth[name].items())[:2])}")
        elif isinstance(ground_truth[name], list):
            print(f"  GT list length: {len(ground_truth[name])}")
            print(f"  GT sample: {ground_truth[name][:2]}")

if __name__ == '__main__':
    predict = r'C:\Users\franc\OneDrive - Alma Mater Studiorum Università di Bologna\Desktop\BOND-OC\WhoIsWho\bond\out\res.json'
    ground_truth = r'C:\Users\franc\OneDrive - Alma Mater Studiorum Università di Bologna\Desktop\BOND-OC\WhoIsWho\bond\dataset\data\src\sna-valid\sna_valid_ground_truth.json'
    
    # Prima il debug per capire i formati
    debug_data_formats(predict, ground_truth)
    
    # Poi la valutazione con debug dettagliato
    print("\n" + "="*50)
    evaluate_with_debug(predict, ground_truth, debug_names=3)
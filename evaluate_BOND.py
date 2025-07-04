import numpy as np
import os
from os.path import join
from tqdm import tqdm
from datetime import datetime
from whoiswho.dataset.data_process import read_pubs, read_raw_pubs
from whoiswho.utils import load_json, save_json

def evaluate(predict_result, ground_truth):
    if isinstance(predict_result, str):
        predict_result = load_json(predict_result)
    if isinstance(ground_truth, str):
        ground_truth = load_json(ground_truth)
    
    # AGGIUNTO: Filtra solo i nomi comuni
    filtered_predict_result = {name: pred for name, pred in predict_result.items() 
                              if name in ground_truth}
    
    name_nums = 0
    result_list = []
    
    for name in filtered_predict_result:  # CAMBIATO: usa filtered_predict_result
        # Get clustering labels in predict_result
        predicted_pubs = dict()
        for idx, pids in enumerate(filtered_predict_result[name]):  # CAMBIATO
            for pid in pids:
                predicted_pubs[pid] = idx
        
        # AGGIUNTO: Gestisce sia formato dict che list per ground_truth
        pubs = []
        ilabel = 0
        true_labels = []
        
        if isinstance(ground_truth[name], dict):
            # Formato originale: {"author_id": ["paper1", "paper2"]}
            for aid in ground_truth[name]:
                pubs.extend(ground_truth[name][aid])
                true_labels.extend([ilabel] * len(ground_truth[name][aid]))
                ilabel += 1
        elif isinstance(ground_truth[name], list):
            # Formato lista: [["paper1", "paper2"], ["paper3", "paper4"]]
            for cluster in ground_truth[name]:
                if isinstance(cluster, list):
                    pubs.extend(cluster)
                    true_labels.extend([ilabel] * len(cluster))
                    ilabel += 1
                else:
                    # Singolo paper
                    pubs.append(cluster)
                    true_labels.append(ilabel)
                    ilabel += 1
        else:
            print(f"Warning: Unknown format for {name}: {type(ground_truth[name])}")
            continue
        
        # AGGIUNTO: Filtra solo i paper che esistono in entrambi i dataset
        filtered_pubs = [pid for pid in pubs if pid in predicted_pubs]
        filtered_true_labels = [true_labels[i] for i, pid in enumerate(pubs) if pid in predicted_pubs]
        
        # AGGIUNTO: Controlla che ci siano paper in comune
        if len(filtered_pubs) == 0:
            print(f"Warning: No common papers for {name}")
            continue
        
        # CAMBIATO: Usa i paper filtrati
        predict_labels = []
        for pid in filtered_pubs:
            predict_labels.append(predicted_pubs[pid])
        
        pairwise_precision, pairwise_recall, pairwise_f1 = pairwise_evaluate(filtered_true_labels, predict_labels)
        result_list.append((pairwise_precision, pairwise_recall, pairwise_f1))
        name_nums += 1
    
    if name_nums == 0:
        print("Error: No names could be evaluated!")
        return 0.0
    
    avg_pairwise_f1 = sum([result[2] for result in result_list]) / name_nums
    print(f'Average Pairwise F1: {avg_pairwise_f1:.3f}')
    
    # AGGIUNTO: Statistiche utili
    print(f'Names evaluated: {name_nums}')
    print(f'Names in predict_result: {len(predict_result)}')
    print(f'Names in ground_truth: {len(ground_truth)}')
    
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
        pairwise_precision = TP / TP_FP if TP_FP > 0 else 0  # AGGIUNTO: controllo divisione per zero
        pairwise_recall = TP / TP_FN if TP_FN > 0 else 0     # AGGIUNTO: controllo divisione per zero
        pairwise_f1 = (2 * pairwise_precision * pairwise_recall) / (pairwise_precision + pairwise_recall)

    return pairwise_precision, pairwise_recall, pairwise_f1

# AGGIUNTO: Funzione di debug per capire il formato
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
    ground_truth = r'C:\Users\franc\OneDrive - Alma Mater Studiorum Università di Bologna\Desktop\BOND-OC\WhoIsWho\bond\dataset\data\src\sna-valid\sna_valid_example.json'
    
    # Prima il debug per capire i formati
    debug_data_formats(predict, ground_truth)
    
    # Poi la valutazione
    print("\n" + "="*50)
    evaluate(predict, ground_truth)
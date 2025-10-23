"""
TRAIN/TEST SPLIT ROBUSTO PER BOND DATASET

Versione migliorata con validazione integrata per prevenire paper mancanti.

OUTPUT STRUCTURE:
train/
  ├── train_pub.json          # Pubblicazioni del train set
  └── train_author.json       # Ground truth formato: {"author": {"author_id": ["papers"]}}

test/
  ├── sna_valid_pub.json      # Pubblicazioni del test set  
  ├── sna_valid_raw.json      # Formato semplice: {"author": ["papers"]}
  └── sna_valid_ground_truth.json  # Ground truth formato BOND
"""

import json
from os.path import join, exists
import os
from sklearn.model_selection import train_test_split
from collections import defaultdict
import hashlib
import re

# ======================== CONFIGURAZIONE ========================
BASE_PATH = r"C:\Users\franc\OneDrive - Alma Mater Studiorum Università di Bologna\Desktop"

# INPUT FILES
ALL_PUBS_FILE = join(BASE_PATH, "BondforOC", "results", "OC_results_with_citations", "converted_metadata_withcit.json")
ORIGINAL_GT_FILE = join(BASE_PATH, "BOND-OC", "WhoIsWho", "bond", "dataset", "data", "src", "sna-valid", "sna_valid_ground_truth.json")
SIMPLE_RAW_FILE = join(BASE_PATH, "BondforOC", "results", "converted_metadata_raw_withcit.json")

# OUTPUT
OUTPUT_DIR = join(BASE_PATH, "BOND-OC", "WhoIsWho", "train_test_split_robust")

TRAIN_RATIO = 0.8
RANDOM_SEED = 42
# ================================================================


def load_publications(file_path):
    """Carica tutte le pubblicazioni"""
    print(f"\n[1/7] Caricamento pubblicazioni...")
    
    if not exists(file_path):
        print(f"   ❌ ERRORE: File non trovato: {file_path}")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        pubs = json.load(f)
    
    print(f"   ✓ {len(pubs)} pubblicazioni caricate")
    
    # Normalizza citazioni se presenti
    for pub_id, pub_data in pubs.items():
        if 'outgoing_citations' in pub_data:
            if isinstance(pub_data['outgoing_citations'], str):
                pub_data['citations_out'] = pub_data['outgoing_citations'].split()
            elif isinstance(pub_data['outgoing_citations'], list):
                pub_data['citations_out'] = pub_data['outgoing_citations']
        
        if 'incoming_citations' in pub_data:
            if isinstance(pub_data['incoming_citations'], str):
                pub_data['citations_in'] = pub_data['incoming_citations'].split()
            elif isinstance(pub_data['incoming_citations'], list):
                pub_data['citations_in'] = pub_data['incoming_citations']
    
    return pubs


def normalize_name_for_matching(name):
    """Normalizza nome per matching"""
    normalized = name.strip().lower()
    normalized = re.sub(r'\b([a-z])\.\s*', r'\1 ', normalized)
    normalized = re.sub(r'\.', '', normalized)
    normalized = re.sub(r"[^\w\s\-]", "", normalized)
    normalized = re.sub(r'\s*-\s*', '-', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    parts = normalized.split()
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[-1]}"
    elif len(parts) == 1:
        return f"unknown_{parts[0]}"
    else:
        return "unknown_unknown"


def convert_simple_to_bond_format(simple_gt, all_pubs):
    """
    Converte formato semplice in formato BOND con validazione.
    
    IMPORTANTE: Filtra automaticamente paper che non esistono in all_pubs!
    """
    gt_bond = {}
    available_pubs = set(all_pubs.keys())
    
    stats = {
        'total_papers_in_gt': 0,
        'valid_papers': 0,
        'missing_papers': 0,
        'missing_paper_ids': []
    }
    
    for author_name, paper_list in simple_gt.items():
        if not isinstance(paper_list, list):
            paper_list = [paper_list]
        
        stats['total_papers_in_gt'] += len(paper_list)
        
        # VALIDAZIONE: Filtra paper esistenti
        valid_papers = []
        for p in paper_list:
            if p in available_pubs:
                valid_papers.append(p)
                stats['valid_papers'] += 1
            else:
                stats['missing_papers'] += 1
                if len(stats['missing_paper_ids']) < 10:  # Salva primi 10 per report
                    stats['missing_paper_ids'].append((author_name, p))
        
        if not valid_papers:
            continue
        
        # Raggruppa paper per author_id basandosi sull'organizzazione
        author_id_to_papers = defaultdict(list)
        
        for paper_id in valid_papers:
            paper = all_pubs[paper_id]
            
            # Cerca l'autore con nome corrispondente nel paper
            author_org = ""
            if 'authors' in paper:
                for author_data in paper['authors']:
                    if normalize_name_for_matching(author_data.get('name', '')) == author_name:
                        author_org = author_data.get('org', '')
                        break
            
            # Genera author_id deterministico
            unique_string = f"{author_name}||{author_org}"
            author_id = hashlib.md5(unique_string.encode()).hexdigest()[:8]
            
            author_id_to_papers[author_id].append(paper_id)
        
        gt_bond[author_name] = dict(author_id_to_papers)
    
    return gt_bond, stats


def clean_ground_truth(gt, available_pubs):
    """
    NUOVA FUNZIONE: Pulisce ground truth rimuovendo paper mancanti.
    
    Questa funzione garantisce che OGNI paper nella GT esista in available_pubs.
    """
    print(f"\n[2/7] Pulizia e validazione ground truth...")
    
    cleaned_gt = {}
    stats = {
        'original_authors': len(gt),
        'original_papers': 0,
        'removed_papers': 0,
        'removed_author_ids': 0,
        'removed_authors': 0,
        'kept_authors': 0,
        'kept_papers': 0,
        'missing_examples': []
    }
    
    for author, author_data in gt.items():
        if not isinstance(author_data, dict):
            continue
        
        cleaned_author_data = {}
        
        for author_id, papers in author_data.items():
            if isinstance(papers, list):
                stats['original_papers'] += len(papers)
                
                # VALIDAZIONE: Filtra paper esistenti
                valid_papers = [p for p in papers if p in available_pubs]
                removed = len(papers) - len(valid_papers)
                
                if removed > 0:
                    stats['removed_papers'] += removed
                    # Salva esempio per report
                    if len(stats['missing_examples']) < 10:
                        for p in papers:
                            if p not in available_pubs:
                                stats['missing_examples'].append((author, author_id, p))
                
                if valid_papers:
                    cleaned_author_data[author_id] = valid_papers
                    stats['kept_papers'] += len(valid_papers)
                else:
                    stats['removed_author_ids'] += 1
            
            elif isinstance(papers, str):
                stats['original_papers'] += 1
                
                if papers in available_pubs:
                    cleaned_author_data[author_id] = [papers]
                    stats['kept_papers'] += 1
                else:
                    stats['removed_papers'] += 1
                    stats['removed_author_ids'] += 1
                    if len(stats['missing_examples']) < 10:
                        stats['missing_examples'].append((author, author_id, papers))
        
        # Mantieni autore solo se ha almeno un author_id valido
        if cleaned_author_data:
            cleaned_gt[author] = cleaned_author_data
            stats['kept_authors'] += 1
        else:
            stats['removed_authors'] += 1
    
    # Stampa report pulizia
    print(f"   Paper nella GT originale: {stats['original_papers']}")
    print(f"   Paper rimossi (mancanti in pub): {stats['removed_papers']}")
    print(f"   Paper mantenuti: {stats['kept_papers']}")
    
    if stats['removed_papers'] > 0:
        pct = (stats['removed_papers'] / stats['original_papers']) * 100
        print(f"   ⚠️  {pct:.1f}% paper rimossi")
        
        print(f"\n   Esempi paper mancanti (primi 5):")
        for i, (author, aid, pid) in enumerate(stats['missing_examples'][:5], 1):
            print(f"      {i}. Author: {author}, ID: {aid}, Paper: {pid}")
    else:
        print(f"   ✅ Nessun paper mancante - GT già valida!")
    
    print(f"\n   Autori: {stats['original_authors']} → {stats['kept_authors']}")
    print(f"   Author IDs rimossi: {stats['removed_author_ids']}")
    print(f"   Autori completamente rimossi: {stats['removed_authors']}")
    
    return cleaned_gt, stats


def load_ground_truth(original_gt_path, simple_raw_path, all_pubs):
    """Carica ground truth con validazione integrata"""
    print(f"\n[Step 2] Caricamento ground truth...")
    
    available_pubs = set(all_pubs.keys())
    
    # Prova con GT originale
    if exists(original_gt_path):
        print(f"   ✓ Trovata ground truth originale")
        with open(original_gt_path, 'r', encoding='utf-8') as f:
            gt = json.load(f)
        
        # Verifica formato
        first_author = list(gt.keys())[0]
        first_data = gt[first_author]
        
        if isinstance(first_data, dict):
            print(f"   ✓ Formato BOND corretto")
            # PULIZIA AUTOMATICA
            cleaned_gt, stats = clean_ground_truth(gt, available_pubs)
            return cleaned_gt
        else:
            print(f"   ⚠️  Formato semplice rilevato")
    
    # Carica formato semplice
    if not exists(simple_raw_path):
        print(f"   ❌ ERRORE: Nessun file ground truth trovato")
        return None
    
    print(f"   Caricamento da file semplice")
    with open(simple_raw_path, 'r', encoding='utf-8') as f:
        simple_gt = json.load(f)
    
    # Conversione con validazione
    gt_bond, conv_stats = convert_simple_to_bond_format(simple_gt, all_pubs)
    
    print(f"\n   Conversione completata:")
    print(f"   Paper totali: {conv_stats['total_papers_in_gt']}")
    print(f"   Paper validi: {conv_stats['valid_papers']}")
    if conv_stats['missing_papers'] > 0:
        print(f"   ⚠️  Paper mancanti: {conv_stats['missing_papers']}")
        print(f"   Esempi (primi 5):")
        for author, pid in conv_stats['missing_paper_ids'][:5]:
            print(f"      Author: {author}, Paper: {pid}")
    
    print(f"   ✓ {len(gt_bond)} autori nel formato BOND")
    
    return gt_bond


def validate_consistency(pub, gt, name):
    """
    NUOVA FUNZIONE: Validazione finale di consistenza.
    
    Verifica che OGNI paper nella GT esista in pub.
    """
    print(f"\n[Validazione] Verifica consistenza {name}...")
    
    available_pubs = set(pub.keys())
    
    # Estrai tutti i paper dalla GT
    gt_papers = set()
    for author_data in gt.values():
        for papers in author_data.values():
            if isinstance(papers, list):
                gt_papers.update(papers)
            else:
                gt_papers.add(papers)
    
    missing = gt_papers - available_pubs
    
    print(f"   Paper in GT: {len(gt_papers)}")
    print(f"   Paper in pub: {len(available_pubs)}")
    
    if missing:
        print(f"   ❌ ERRORE: {len(missing)} paper nella GT ma non in pub!")
        print(f"   Primi 10 mancanti:")
        for i, pid in enumerate(list(missing)[:10], 1):
            print(f"      {i}. {pid}")
        return False
    else:
        print(f"   ✅ Consistenza verificata - tutti i paper esistono!")
        return True


def stratified_split_authors(gt, random_seed=42):
    """Split stratificato basato sul numero di paper per autore"""
    print(f"\n[3/7] Split stratificato autori (80/20)...")
    
    # Calcola metadati autori
    author_metadata = []
    for author, author_dict in gt.items():
        total_papers = sum(len(papers) for papers in author_dict.values())
        num_author_ids = len(author_dict)
        
        author_metadata.append({
            'name': author,
            'total_papers': total_papers,
            'num_author_ids': num_author_ids
        })
    
    # Crea bin per stratificazione
    import pandas as pd
    df = pd.DataFrame(author_metadata)
    
    # Stratifica per numero di paper (quartili)
    df['paper_bin'] = pd.qcut(df['total_papers'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    
    # Split stratificato
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=1-TRAIN_RATIO,
            random_state=random_seed,
            stratify=df['paper_bin']
        )
    except ValueError:
        print(f"   ⚠️  Stratificazione fallita, uso split casuale")
        train_df, test_df = train_test_split(
            df,
            test_size=1-TRAIN_RATIO,
            random_state=random_seed
        )
    
    train_authors = set(train_df['name'])
    test_authors = set(test_df['name'])
    
    print(f"   ✓ Train: {len(train_authors)} autori")
    print(f"   ✓ Test:  {len(test_authors)} autori")
    
    return train_authors, test_authors


def generate_train_files(train_authors, all_pubs, gt):
    """Genera file per training set"""
    print(f"\n[4/7] Generazione file TRAIN...")
    
    train_pub = {}
    train_author = {}
    
    for author in train_authors:
        if author not in gt:
            continue
        
        train_author[author] = gt[author]
        
        for author_id, paper_list in gt[author].items():
            for paper_id in paper_list:
                if paper_id in all_pubs and paper_id not in train_pub:
                    train_pub[paper_id] = all_pubs[paper_id]
    
    print(f"   ✓ train_pub.json: {len(train_pub)} pubblicazioni")
    print(f"   ✓ train_author.json: {len(train_author)} autori")
    
    return train_pub, train_author


def generate_test_files(test_authors, all_pubs, gt):
    """Genera file per test set"""
    print(f"\n[5/7] Generazione file TEST...")
    
    test_pub = {}
    test_raw = {}
    test_gt = {}
    
    for author in test_authors:
        if author not in gt:
            continue
        
        test_gt[author] = gt[author]
        
        # Raw (formato semplice)
        all_papers = []
        for author_id, paper_list in gt[author].items():
            all_papers.extend(paper_list)
            
            for paper_id in paper_list:
                if paper_id in all_pubs and paper_id not in test_pub:
                    test_pub[paper_id] = all_pubs[paper_id]
        
        test_raw[author] = all_papers
    
    print(f"   ✓ sna_valid_pub.json: {len(test_pub)} pubblicazioni")
    print(f"   ✓ sna_valid_raw.json: {len(test_raw)} autori")
    print(f"   ✓ sna_valid_ground_truth.json: {len(test_gt)} autori")
    
    return test_pub, test_raw, test_gt


def save_all_files(train_pub, train_author, test_pub, test_raw, test_gt):
    """Salva tutti i file JSON"""
    print(f"\n[6/7] Salvataggio file...")
    
    train_dir = join(OUTPUT_DIR, "train")
    test_dir = join(OUTPUT_DIR, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Salva train
    with open(join(train_dir, "train_pub.json"), 'w', encoding='utf-8') as f:
        json.dump(train_pub, f, indent=2, ensure_ascii=False)
    
    with open(join(train_dir, "train_author.json"), 'w', encoding='utf-8') as f:
        json.dump(train_author, f, indent=2, ensure_ascii=False)
    
    # Salva test
    with open(join(test_dir, "sna_valid_pub.json"), 'w', encoding='utf-8') as f:
        json.dump(test_pub, f, indent=2, ensure_ascii=False)
    
    with open(join(test_dir, "sna_valid_raw.json"), 'w', encoding='utf-8') as f:
        json.dump(test_raw, f, indent=2, ensure_ascii=False)
    
    with open(join(test_dir, "sna_valid_ground_truth.json"), 'w', encoding='utf-8') as f:
        json.dump(test_gt, f, indent=2, ensure_ascii=False)
    
    print(f"   ✓ Tutti i file salvati in: {OUTPUT_DIR}")


def final_validation(train_pub, train_author, test_pub, test_raw, test_gt):
    """NUOVA FUNZIONE: Validazione finale completa"""
    print(f"\n[7/7] Validazione finale...")
    
    all_valid = True
    
    # Valida train
    if not validate_consistency(train_pub, train_author, "TRAIN"):
        all_valid = False
    
    # Valida test
    if not validate_consistency(test_pub, test_gt, "TEST"):
        all_valid = False
    
    # Valida consistenza raw vs GT (test)
    print(f"\n[Validazione] Verifica raw vs ground_truth...")
    gt_papers = set()
    for author_data in test_gt.values():
        for papers in author_data.values():
            gt_papers.update(papers)
    
    raw_papers = set()
    for papers in test_raw.values():
        raw_papers.update(papers)
    
    if gt_papers == raw_papers:
        print(f"   ✅ Raw e GT consistenti ({len(gt_papers)} paper)")
    else:
        print(f"   ⚠️  Inconsistenza raw vs GT")
        all_valid = False
    
    return all_valid


def main():
    """Entry point"""
    print("=" * 80)
    print("TRAIN/TEST SPLIT ROBUSTO - BOND DATASET")
    print("Con validazione automatica paper mancanti")
    print("=" * 80)
    
    try:
        # 1. Carica pubblicazioni
        all_pubs = load_publications(ALL_PUBS_FILE)
        if all_pubs is None:
            return
        
        # 2. Carica e pulisci ground truth
        gt = load_ground_truth(ORIGINAL_GT_FILE, SIMPLE_RAW_FILE, all_pubs)
        if gt is None:
            return
        
        # 3. Split autori
        train_authors, test_authors = stratified_split_authors(gt, RANDOM_SEED)
        
        # 4. Genera file train
        train_pub, train_author = generate_train_files(train_authors, all_pubs, gt)
        
        # 5. Genera file test
        test_pub, test_raw, test_gt = generate_test_files(test_authors, all_pubs, gt)
        
        # 6. Salva tutto
        save_all_files(train_pub, train_author, test_pub, test_raw, test_gt)
        
        # 7. VALIDAZIONE FINALE
        all_valid = final_validation(train_pub, train_author, test_pub, test_raw, test_gt)
        
        print("\n" + "=" * 80)
        if all_valid:
            print("✅ SPLIT COMPLETATO CON SUCCESSO!")
            print("   Tutti i controlli di validazione sono passati.")
        else:
            print("⚠️  SPLIT COMPLETATO CON WARNING")
            print("   Alcuni controlli hanno fallito - verifica sopra")
        print("=" * 80)
        
        print(f"\nFile generati in: {OUTPUT_DIR}")
        print("\nTRAIN/")
        print(f"  ✓ train_pub.json ({len(train_pub)} pub)")
        print(f"  ✓ train_author.json ({len(train_author)} autori)")
        print("\nTEST/")
        print(f"  ✓ sna_valid_pub.json ({len(test_pub)} pub)")
        print(f"  ✓ sna_valid_raw.json ({len(test_raw)} autori)")
        print(f"  ✓ sna_valid_ground_truth.json ({len(test_gt)} autori)")
        
        if all_valid:
            print("\n💡 File pronti per il preprocessing BOND!")
            print("   Nessun KeyError dovrebbe verificarsi.")
        
        print("\n" + "=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
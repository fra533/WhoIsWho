"""
TRAIN/TEST SPLIT PER BOND DATASET - VERSIONE SEMPLIFICATA

DIFFERENZE CON LO SCRIPT ORIGINALE:
- Rimossa la gestione del file GT originale in formato BOND (non disponibile)
- Usa SOLO il file ground truth in formato semplice (converted_metadata_raw_withcit.json)
- Semplificata la funzione load_ground_truth() per usare un solo file input
- Stessa logica di validazione e conversione formato BOND
- Output identico: train/ e test/ con tutti i file necessari

INPUT RICHIESTI:
1. converted_metadata_withcit.json → Tutte le pubblicazioni con metadati completi
2. converted_metadata_raw_withcit.json → Ground truth formato: {"author": ["papers"]}

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
ALL_PUBS_FILE = join(BASE_PATH, "BondforOC", "results", "OC_results_with_citations", r"C:\Users\franc\OneDrive - Alma Mater Studiorum Università di Bologna\Desktop\BondforOC\scientometrics_complete.json")
SIMPLE_GT_FILE = join(BASE_PATH, "BondforOC", "results", r"C:\Users\franc\OneDrive - Alma Mater Studiorum Università di Bologna\Desktop\BondforOC\results\scientometrics_raw.json")
# OUTPUT
OUTPUT_DIR = join(BASE_PATH, "BOND-OC", "WhoIsWho", "train_test_split_scientometrics")

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


def load_ground_truth(simple_gt_path, all_pubs):
    """Carica ground truth da file semplice (VERSIONE SEMPLIFICATA)"""
    print(f"\n[2/7] Caricamento ground truth...")
    
    if not exists(simple_gt_path):
        print(f"   ❌ ERRORE: File ground truth non trovato: {simple_gt_path}")
        return None
    
    print(f"   📄 Caricamento da: {simple_gt_path}")
    with open(simple_gt_path, 'r', encoding='utf-8') as f:
        simple_gt = json.load(f)
    
    print(f"   ✓ {len(simple_gt)} autori caricati dal file semplice")
    
    # Conversione con validazione automatica
    gt_bond, conv_stats = convert_simple_to_bond_format(simple_gt, all_pubs)
    
    print(f"\n   📊 Conversione in formato BOND:")
    print(f"   Paper totali nella GT: {conv_stats['total_papers_in_gt']}")
    print(f"   Paper validi: {conv_stats['valid_papers']}")
    
    if conv_stats['missing_papers'] > 0:
        pct = (conv_stats['missing_papers'] / conv_stats['total_papers_in_gt']) * 100
        print(f"   ⚠️  Paper mancanti: {conv_stats['missing_papers']} ({pct:.1f}%)")
        print(f"\n   Esempi paper mancanti (primi 5):")
        for i, (author, pid) in enumerate(conv_stats['missing_paper_ids'][:5], 1):
            print(f"      {i}. Author: {author}, Paper: {pid}")
    else:
        print(f"   ✅ Nessun paper mancante!")
    
    print(f"\n   ✓ {len(gt_bond)} autori convertiti in formato BOND")
    
    return gt_bond


def validate_consistency(pub, gt, name):
    """
    Validazione finale di consistenza.
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
    
    # Stratifica per numero di paper
    try:
        # Prova con quartili
        df['paper_bin'] = pd.qcut(df['total_papers'], q=4, labels=False, duplicates='drop')
        
        # Verifica che ci siano abbastanza bin
        unique_bins = df['paper_bin'].nunique()
        if unique_bins < 2:
            raise ValueError("Non abbastanza bin per stratificazione")
        
        # Split stratificato
        train_df, test_df = train_test_split(
            df,
            test_size=1-TRAIN_RATIO,
            random_state=random_seed,
            stratify=df['paper_bin']
        )
        print(f"   ✓ Split stratificato per {unique_bins} bin di paper")
        
    except (ValueError, TypeError) as e:
        # Fallback: prova con meno bin o split casuale
        print(f"   ⚠️  Stratificazione quartili fallita: {e}")
        
        try:
            # Prova con 3 bin
            df['paper_bin'] = pd.qcut(df['total_papers'], q=3, labels=False, duplicates='drop')
            unique_bins = df['paper_bin'].nunique()
            
            if unique_bins >= 2:
                train_df, test_df = train_test_split(
                    df,
                    test_size=1-TRAIN_RATIO,
                    random_state=random_seed,
                    stratify=df['paper_bin']
                )
                print(f"   ✓ Split stratificato per {unique_bins} bin di paper (terzili)")
            else:
                raise ValueError("Non abbastanza bin anche con 3 quartili")
                
        except (ValueError, TypeError):
            # Ultimo fallback: split casuale
            print(f"   ⚠️  Uso split casuale (troppi valori duplicati)")
            train_df, test_df = train_test_split(
                df,
                test_size=1-TRAIN_RATIO,
                random_state=random_seed
            )
    
    train_authors = set(train_df['name'])
    test_authors = set(test_df['name'])
    
    print(f"   ✓ Train: {len(train_authors)} autori ({len(train_authors)/len(df)*100:.1f}%)")
    print(f"   ✓ Test:  {len(test_authors)} autori ({len(test_authors)/len(df)*100:.1f}%)")
    
    # Mostra distribuzione paper
    train_papers = sum(df[df['name'].isin(train_authors)]['total_papers'])
    test_papers = sum(df[df['name'].isin(test_authors)]['total_papers'])
    print(f"   📄 Train papers: {train_papers} ({train_papers/(train_papers+test_papers)*100:.1f}%)")
    print(f"   📄 Test papers: {test_papers} ({test_papers/(train_papers+test_papers)*100:.1f}%)")
    
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
    
    print(f"   ✓ Train files → {train_dir}")
    print(f"   ✓ Test files → {test_dir}")


def final_validation(train_pub, train_author, test_pub, test_raw, test_gt):
    """Validazione finale completa"""
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
        missing_in_raw = gt_papers - raw_papers
        missing_in_gt = raw_papers - gt_papers
        if missing_in_raw:
            print(f"      Paper in GT ma non in raw: {len(missing_in_raw)}")
        if missing_in_gt:
            print(f"      Paper in raw ma non in GT: {len(missing_in_gt)}")
        all_valid = False
    
    return all_valid


def print_statistics(train_pub, train_author, test_pub, test_gt):
    """Stampa statistiche finali dettagliate"""
    print("\n" + "=" * 80)
    print("📊 STATISTICHE FINALI")
    print("=" * 80)
    
    # Train stats
    train_papers_total = sum(len(papers) for author_data in train_author.values() 
                             for papers in author_data.values())
    print(f"\n🏋️  TRAINING SET:")
    print(f"   Autori: {len(train_author)}")
    print(f"   Pubblicazioni uniche: {len(train_pub)}")
    print(f"   Paper totali (con duplicati): {train_papers_total}")
    print(f"   Media paper per autore: {train_papers_total/len(train_author):.1f}")
    
    # Test stats
    test_papers_total = sum(len(papers) for author_data in test_gt.values() 
                            for papers in author_data.values())
    print(f"\n🧪 TEST SET:")
    print(f"   Autori: {len(test_gt)}")
    print(f"   Pubblicazioni uniche: {len(test_pub)}")
    print(f"   Paper totali (con duplicati): {test_papers_total}")
    print(f"   Media paper per autore: {test_papers_total/len(test_gt):.1f}")
    
    # Overall
    total_authors = len(train_author) + len(test_gt)
    total_papers = train_papers_total + test_papers_total
    print(f"\n📈 TOTALE:")
    print(f"   Autori: {total_authors}")
    print(f"   Paper: {total_papers}")
    print(f"   Split ratio: {len(train_author)/total_authors:.1%} train / {len(test_gt)/total_authors:.1%} test")
    
    print("=" * 80)


def main():
    """Entry point"""
    print("=" * 80)
    print("TRAIN/TEST SPLIT - BOND DATASET (VERSIONE SEMPLIFICATA)")
    print("Con validazione automatica paper mancanti")
    print("=" * 80)
    
    try:
        # 1. Carica pubblicazioni
        all_pubs = load_publications(ALL_PUBS_FILE)
        if all_pubs is None:
            return
        
        # 2. Carica e converti ground truth (VERSIONE SEMPLIFICATA)
        gt = load_ground_truth(SIMPLE_GT_FILE, all_pubs)
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
        
        # 8. Statistiche
        print_statistics(train_pub, train_author, test_pub, test_gt)
        
        print("\n" + "=" * 80)
        if all_valid:
            print("✅ SPLIT COMPLETATO CON SUCCESSO!")
            print("   Tutti i controlli di validazione sono passati.")
        else:
            print("⚠️  SPLIT COMPLETATO CON WARNING")
            print("   Alcuni controlli hanno fallito - verifica i messaggi sopra")
        print("=" * 80)
        
        print(f"\n📁 File generati in: {OUTPUT_DIR}")
        print("\n   train/")
        print(f"     ├── train_pub.json ({len(train_pub)} pubblicazioni)")
        print(f"     └── train_author.json ({len(train_author)} autori)")
        print("\n   test/")
        print(f"     ├── sna_valid_pub.json ({len(test_pub)} pubblicazioni)")
        print(f"     ├── sna_valid_raw.json ({len(test_raw)} autori)")
        print(f"     └── sna_valid_ground_truth.json ({len(test_gt)} autori)")
        
        if all_valid:
            print("\n💡 File pronti per il preprocessing BOND!")
            print("   Nessun KeyError dovrebbe verificarsi durante il training.")
        
        print("\n" + "=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERRORE CRITICO: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
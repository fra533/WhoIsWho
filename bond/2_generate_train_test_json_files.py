"""
Genera file JSON train/test nel formato BOND corretto:
- train_author.json (formato dict con author_id)
- test_raw.json (formato list)
"""

import json
from os.path import join, exists
import os
import traceback

# ======================== CONFIGURAZIONE ========================
BASE_PATH = r"C:\Users\franc\OneDrive - Alma Mater Studiorum Università di Bologna\Desktop\BOND-OC\WhoIsWho\bond"

# Input files
SPLIT_DIR = join(BASE_PATH, "bond_train_test_split")
TRAIN_AUTHORS_FILE = join(SPLIT_DIR, "train_authors.txt")
TEST_AUTHORS_FILE = join(SPLIT_DIR, "test_authors.txt")

# File JSON originali BOND
ORIGINAL_PUB_JSON = join(BASE_PATH, "dataset", "data", "src", "sna-valid", "converted_metadata_pub.json")
ORIGINAL_GT_JSON = join(BASE_PATH, "dataset", "data", "src", "sna-valid", "sna_valid_ground_truth.json")

# Output directory
OUTPUT_DIR = join(BASE_PATH, "bond_train_test_split")
# ================================================================


def load_author_lists():
    """Carica liste autori train/test"""
    print("="*80)
    print("GENERAZIONE JSON FILES PER TRAIN/TEST - FORMATO CORRETTO")
    print("="*80)
    print("\n[1/5] Caricamento liste autori...")
    
    if not exists(TRAIN_AUTHORS_FILE):
        print(f"ERRORE: File non trovato: {TRAIN_AUTHORS_FILE}")
        print("Esegui prima split_dataset_80_20.py")
        return None, None
    
    with open(TRAIN_AUTHORS_FILE, 'r', encoding='utf-8') as f:
        train_authors = set(line.strip() for line in f if line.strip())
    
    with open(TEST_AUTHORS_FILE, 'r', encoding='utf-8') as f:
        test_authors = set(line.strip() for line in f if line.strip())
    
    print(f"      Train: {len(train_authors)} autori")
    print(f"      Test:  {len(test_authors)} autori")
    
    return train_authors, test_authors


def load_original_data():
    """Carica dati originali BOND"""
    print("\n[2/5] Caricamento dati originali BOND...")
    
    # Pubblicazioni
    with open(ORIGINAL_PUB_JSON, 'r', encoding='utf-8') as f:
        all_pubs = json.load(f)
    print(f"      Pubblicazioni totali: {len(all_pubs)}")
    
    # Ground truth
    with open(ORIGINAL_GT_JSON, 'r', encoding='utf-8') as f:
        all_gt = json.load(f)
    print(f"      Autori in ground truth: {len(all_gt)}")
    
    return all_pubs, all_gt


def clean_ground_truth(all_gt, all_pubs):
    """Pulisce il ground truth rimuovendo paper che non esistono in all_pubs"""
    print("\n[2b/5] Pulizia ground truth (rimuove paper mancanti)...")
    
    cleaned_gt = {}
    total_papers_original = 0
    total_papers_cleaned = 0
    total_papers_removed = 0
    
    available_pubs = set(all_pubs.keys())
    
    for author, clusters in all_gt.items():
        cleaned_clusters = {}
        
        for author_id, papers in clusters.items():
            total_papers_original += len(papers)
            
            # Filtra solo paper disponibili
            valid_papers = [p for p in papers if p in available_pubs]
            total_papers_cleaned += len(valid_papers)
            total_papers_removed += len(papers) - len(valid_papers)
            
            if valid_papers:
                cleaned_clusters[author_id] = valid_papers
        
        if cleaned_clusters:
            cleaned_gt[author] = cleaned_clusters
    
    print(f"      Paper originali nel GT: {total_papers_original}")
    print(f"      Paper validi (in pubs): {total_papers_cleaned}")
    print(f"      Paper rimossi (mancanti): {total_papers_removed}")
    print(f"      Autori mantenuti: {len(cleaned_gt)} / {len(all_gt)}")
    
    return cleaned_gt


def filter_publications_by_authors(all_pubs, all_gt, author_set, set_name):
    """Filtra pubblicazioni per un set di autori specifico"""
    print(f"\n      Filtraggio {set_name}...")
    
    paper_ids_for_authors = set()
    
    for author in author_set:
        if author in all_gt:
            for author_id, paper_list in all_gt[author].items():
                paper_ids_for_authors.update(paper_list)
    
    filtered_pubs = {
        pid: pub_data 
        for pid, pub_data in all_pubs.items() 
        if pid in paper_ids_for_authors
    }
    
    print(f"        Paper per {len(author_set)} autori: {len(filtered_pubs)}")
    
    return filtered_pubs


def filter_ground_truth_by_authors(all_gt, author_set, set_name):
    """Filtra ground truth per un set di autori"""
    filtered_gt = {
        author: clusters 
        for author, clusters in all_gt.items() 
        if author in author_set
    }
    
    total_papers = sum(
        len(papers) 
        for author_clusters in filtered_gt.values() 
        for papers in author_clusters.values()
    )
    
    print(f"        Autori: {len(filtered_gt)}, Total papers in GT: {total_papers}")
    
    return filtered_gt


def build_train_author_json(all_gt, train_authors):
    """Crea train_author.json nel formato corretto per BOND (dict con author_id)"""
    train_author = {}
    
    for author in train_authors:
        if author in all_gt:
            train_author[author] = all_gt[author]
    
    print(f"        Train author mapping: {len(train_author)} autori")
    
    total_papers = sum(
        len(papers) 
        for author_clusters in train_author.values() 
        for papers in author_clusters.values()
    )
    print(f"        Total papers in mapping: {total_papers}")
    
    return train_author


def build_test_raw_json(all_gt, test_authors):
    """Crea test_raw.json nel formato corretto per BOND (list di paper per autore)"""
    test_raw = {}
    
    for author in test_authors:
        if author in all_gt:
            all_papers = []
            for author_id, papers in all_gt[author].items():
                all_papers.extend(papers)
            test_raw[author] = all_papers
    
    print(f"        Test raw mapping: {len(test_raw)} autori")
    
    total_papers = sum(len(papers) for papers in test_raw.values())
    print(f"        Total papers in mapping: {total_papers}")
    
    return test_raw


def generate_train_test_json(train_authors, test_authors, all_pubs, all_gt):
    """Genera tutti i file JSON necessari per train e test"""
    print("\n[3/5] Generazione file JSON...")
    
    print("\n   === TRAIN SET ===")
    train_pubs = filter_publications_by_authors(all_pubs, all_gt, train_authors, "train")
    train_gt = filter_ground_truth_by_authors(all_gt, train_authors, "train")
    train_author = build_train_author_json(all_gt, train_authors)
    
    print("\n   === TEST SET ===")
    test_pubs = filter_publications_by_authors(all_pubs, all_gt, test_authors, "test")
    test_gt = filter_ground_truth_by_authors(all_gt, test_authors, "test")
    test_raw = build_test_raw_json(all_gt, test_authors)
    
    return train_pubs, train_gt, train_author, test_pubs, test_gt, test_raw


def save_json_files(train_pubs, train_gt, train_author, test_pubs, test_gt, test_raw):
    """Salva tutti i file JSON nel formato BOND corretto"""
    print("\n[4/5] Salvataggio file JSON...")
    
    train_dir = join(OUTPUT_DIR, "train")
    test_dir = join(OUTPUT_DIR, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # TRAIN FILES
    with open(join(train_dir, "train_pub.json"), 'w', encoding='utf-8') as f:
        json.dump(train_pubs, f, indent=2, ensure_ascii=False)
    with open(join(train_dir, "train_ground_truth.json"), 'w', encoding='utf-8') as f:
        json.dump(train_gt, f, indent=2, ensure_ascii=False)
    with open(join(train_dir, "train_author.json"), 'w', encoding='utf-8') as f:
        json.dump(train_author, f, indent=2, ensure_ascii=False)
    
    # TEST FILES
    with open(join(test_dir, "test_pub.json"), 'w', encoding='utf-8') as f:
        json.dump(test_pubs, f, indent=2, ensure_ascii=False)
    with open(join(test_dir, "test_ground_truth.json"), 'w', encoding='utf-8') as f:
        json.dump(test_gt, f, indent=2, ensure_ascii=False)
    with open(join(test_dir, "test_raw.json"), 'w', encoding='utf-8') as f:
        json.dump(test_raw, f, indent=2, ensure_ascii=False)
    
    print(f"      ✓ File salvati in {train_dir} e {test_dir}")
    return train_dir, test_dir


def verify_formats(train_dir, test_dir):
    """Verifica che i formati siano corretti"""
    print("\n[5/5] Verifica formati...")

    with open(join(train_dir, "train_author.json"), 'r', encoding='utf-8') as f:
        train_author = json.load(f)
    with open(join(test_dir, "test_raw.json"), 'r', encoding='utf-8') as f:
        test_raw = json.load(f)

    example_train = next(iter(train_author.items()))
    example_test = next(iter(test_raw.items()))

    print(f"\n   Esempio train_author.json → {example_train[0]}: {type(example_train[1])}")
    print(f"   Esempio test_raw.json → {example_test[0]}: {type(example_test[1])}")


def main():
    try:
        train_authors, test_authors = load_author_lists()
        if train_authors is None:
            return

        all_pubs, all_gt = load_original_data()
        cleaned_gt = clean_ground_truth(all_gt, all_pubs)

        train_pubs, train_gt, train_author, test_pubs, test_gt, test_raw = generate_train_test_json(
            train_authors, test_authors, all_pubs, cleaned_gt
        )

        train_dir, test_dir = save_json_files(
            train_pubs, train_gt, train_author,
            test_pubs, test_gt, test_raw
        )

        verify_formats(train_dir, test_dir)

        print("\n" + "="*80)
        print("✓ FILE JSON GENERATI CON FORMATO CORRETTO E ALLINEATI!")
        print("="*80)

    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()

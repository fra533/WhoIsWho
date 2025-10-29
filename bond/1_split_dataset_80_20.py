"""
Split dataset in train (80%) e test (20%)
Usa il CSV già estratto con features pulite generate da debug_graphs_and_expand.py 
Modificare INPUT_CSV 
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from os.path import join, exists
import os

# ======================== CONFIGURAZIONE ========================
BASE_PATH = r"C:\Users\franc\OneDrive - Alma Mater Studiorum Università di Bologna\Desktop\BOND-OC\WhoIsWho\bond"

# Input: CSV con features già estratte
INPUT_CSV = join(BASE_PATH, "bond_unified_refined_analysis", "refined_features_results.csv")

# Output
OUTPUT_DIR = join(BASE_PATH, "bond_train_test_split")

TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
RANDOM_SEED = 42
# ================================================================


def load_dataset():

    print("="*80)
    print("STRATIFIED TRAIN/TEST SPLIT (80/20)")
    print("="*80)
    print(f"\n[1/5] Caricamento dataset...")
    
    if not exists(INPUT_CSV):
        print(f"ERRORE: File non trovato: {INPUT_CSV}")
        print(f"\nEsegui prima unified_refined_analysis.py per generare il CSV")
        return None
    
    df = pd.read_csv(INPUT_CSV, index_col=0)
    print(f"      Dataset: {len(df)} autori, {len(df.columns)} colonne")
    
    return df


def create_stratification_bins(df):
    """Crea bin per stratificazione su più dimensioni"""
    print(f"\n[2/5] Creazione bin per stratificazione...")
    
    # STRATEGIA SEMPLIFICATA: solo F1 e cluster
    # Evita troppi bin per dataset piccolo (79 autori)
    
    # 1. Bin F1 score (performance) - solo 3 bin invece di 5
    df['f1_bin'] = pd.qcut(df['f1'], q=3, labels=['low', 'medium', 'high'], duplicates='drop')
    
    # 2. Usa cluster già presente (3 cluster nel dataset)
    # df['cluster'] già nel dataset
    
    # 3. Crea stratification key combinata (max 3x3 = 9 classi)
    df['stratify_key'] = (
        df['f1_bin'].astype(str) + '_' + 
        df['cluster'].astype(str)
    )
    
    print(f"      Chiavi di stratificazione univoche: {df['stratify_key'].nunique()}")
    
    print(f"\n      Distribuzione F1 bins:")
    for bin_name, count in df['f1_bin'].value_counts().sort_index().items():
        print(f"        {bin_name:12s}: {count:3d} autori")
    
    print(f"\n      Distribuzione cluster:")
    for cluster, count in df['cluster'].value_counts().sort_index().items():
        print(f"        Cluster {cluster}:   {count:3d} autori")
    
    print(f"\n      Distribuzione chiavi combinate:")
    for key, count in df['stratify_key'].value_counts().sort_values(ascending=False).head(9).items():
        print(f"        {key:20s}: {count:3d} autori")
    
    return df


def perform_split(df):
    """Esegue split stratificato"""
    print(f"\n[3/5] Split stratificato ({TRAIN_RATIO:.0%} train / {TEST_RATIO:.0%} test)...")
    
    # Identifica chiavi con 1 solo sample (non stratificabili)
    stratify_counts = df['stratify_key'].value_counts()
    rare_keys = stratify_counts[stratify_counts == 1].index
    
    if len(rare_keys) > 0:
        print(f"      ⚠ {len(rare_keys)} chiavi con 1 solo sample")
        print(f"      Soluzione: assegnazione casuale per questi sample")
        
        # Separa rare da stratifiable
        df_rare = df[df['stratify_key'].isin(rare_keys)]
        df_stratifiable = df[~df['stratify_key'].isin(rare_keys)]
        
        # Split stratificato su parte stratificabile
        train_strat, test_strat = train_test_split(
            df_stratifiable,
            test_size=TEST_RATIO,
            random_state=RANDOM_SEED,
            stratify=df_stratifiable['stratify_key']
        )
        
        # Split casuale su rare
        train_rare, test_rare = train_test_split(
            df_rare,
            test_size=TEST_RATIO,
            random_state=RANDOM_SEED
        )
        
        # Combina
        train_df = pd.concat([train_strat, train_rare])
        test_df = pd.concat([test_strat, test_rare])
    else:
        # Split stratificato standard
        train_df, test_df = train_test_split(
            df,
            test_size=TEST_RATIO,
            random_state=RANDOM_SEED,
            stratify=df['stratify_key']
        )
    
    # Rimuovi colonne temporanee
    cols_to_drop = ['f1_bin', 'papers_bin', 'stratify_key']
    train_df = train_df.drop(columns=cols_to_drop, errors='ignore')
    test_df = test_df.drop(columns=cols_to_drop, errors='ignore')
    
    print(f"      Train: {len(train_df)} autori ({len(train_df)/len(df)*100:.1f}%)")
    print(f"      Test:  {len(test_df)} autori ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, test_df


def validate_split(train_df, test_df, original_df):
    """Valida rappresentatività dello split"""
    print(f"\n[4/5] Validazione split...")
    
    from scipy.stats import ks_2samp
    
    print(f"\n      Confronto distribuzioni:")
    print(f"      {'Metrica':25s} Original    Train       Test        p-value")
    print(f"      {'-'*80}")
    
    # F1 score
    ks_stat, ks_pval = ks_2samp(train_df['f1'], test_df['f1'])
    print(f"      {'F1 score':25s} {original_df['f1'].mean():8.4f}    {train_df['f1'].mean():8.4f}    {test_df['f1'].mean():8.4f}    {ks_pval:.4f} {'✓' if ks_pval > 0.05 else '⚠'}")
    
    # Num papers
    ks_stat, ks_pval = ks_2samp(train_df['num_papers'], test_df['num_papers'])
    print(f"      {'num_papers':25s} {original_df['num_papers'].mean():8.1f}    {train_df['num_papers'].mean():8.1f}    {test_df['num_papers'].mean():8.1f}    {ks_pval:.4f} {'✓' if ks_pval > 0.05 else '⚠'}")
    
    # Graph clustering coef (feature più correlata)
    if 'graph_clustering_coef' in train_df.columns:
        ks_stat, ks_pval = ks_2samp(train_df['graph_clustering_coef'], test_df['graph_clustering_coef'])
        print(f"      {'graph_clustering_coef':25s} {original_df['graph_clustering_coef'].mean():8.4f}    {train_df['graph_clustering_coef'].mean():8.4f}    {test_df['graph_clustering_coef'].mean():8.4f}    {ks_pval:.4f} {'✓' if ks_pval > 0.05 else '⚠'}")
    
    # Emb effective dim
    if 'emb_effective_dim' in train_df.columns:
        ks_stat, ks_pval = ks_2samp(train_df['emb_effective_dim'], test_df['emb_effective_dim'])
        print(f"      {'emb_effective_dim':25s} {original_df['emb_effective_dim'].mean():8.4f}    {train_df['emb_effective_dim'].mean():8.4f}    {test_df['emb_effective_dim'].mean():8.4f}    {ks_pval:.4f} {'✓' if ks_pval > 0.05 else '⚠'}")
    
    print(f"\n      Distribuzione Cluster:")
    print(f"      Cluster  Original  Train  Test")
    print(f"      {'-'*40}")
    for cluster in sorted(original_df['cluster'].unique()):
        orig_pct = (original_df['cluster'] == cluster).sum() / len(original_df) * 100
        train_pct = (train_df['cluster'] == cluster).sum() / len(train_df) * 100
        test_pct = (test_df['cluster'] == cluster).sum() / len(test_df) * 100
        print(f"      {cluster:7d}  {orig_pct:7.1f}%  {train_pct:6.1f}%  {test_pct:5.1f}%")
    
    print(f"\n      Interpretazione KS-test:")
    print(f"        p > 0.05: Distribuzioni simili (BUONO) ✓")
    print(f"        p < 0.05: Distribuzioni diverse (ATTENZIONE) ⚠")


def save_splits(train_df, test_df):
    """Salva i dataset splittati"""
    print(f"\n[5/5] Salvataggio split...")
    
    if not exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # CSV principali
    train_file = join(OUTPUT_DIR, 'train_set.csv')
    test_file = join(OUTPUT_DIR, 'test_set.csv')
    
    train_df.to_csv(train_file)
    test_df.to_csv(test_file)
    
    print(f"      ✓ Train CSV: train_set.csv")
    print(f"      ✓ Test CSV:  test_set.csv")
    
    # Liste nomi autori
    train_authors_file = join(OUTPUT_DIR, 'train_authors.txt')
    test_authors_file = join(OUTPUT_DIR, 'test_authors.txt')
    
    with open(train_authors_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_df.index.tolist()))
    
    with open(test_authors_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_df.index.tolist()))
    
    print(f"      ✓ Train authors: train_authors.txt")
    print(f"      ✓ Test authors:  test_authors.txt")
    
    # Report dettagliato
    report = generate_report(train_df, test_df)
    report_file = join(OUTPUT_DIR, 'split_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"      ✓ Report: split_report.txt")


def generate_report(train_df, test_df):
    """Genera report completo"""
    lines = []
    lines.append("="*80)
    lines.append("TRAIN/TEST SPLIT REPORT")
    lines.append("="*80)
    lines.append("")
    lines.append("CONFIGURAZIONE:")
    lines.append(f"  Random seed:  {RANDOM_SEED}")
    lines.append(f"  Train ratio:  {TRAIN_RATIO:.1%}")
    lines.append(f"  Test ratio:   {TEST_RATIO:.1%}")
    lines.append("")
    lines.append("DIMENSIONI DATASET:")
    lines.append(f"  Train set:    {len(train_df)} autori ({len(train_df)/(len(train_df)+len(test_df))*100:.1f}%)")
    lines.append(f"  Test set:     {len(test_df)} autori ({len(test_df)/(len(train_df)+len(test_df))*100:.1f}%)")
    lines.append(f"  Total:        {len(train_df) + len(test_df)} autori")
    lines.append("")
    
    lines.append("STATISTICHE F1 SCORE:")
    lines.append("-"*80)
    lines.append(f"{'':20s} Train      Test       Delta      Delta %")
    lines.append("-"*80)
    for stat in ['mean', 'std', 'min', 'max']:
        train_val = getattr(train_df['f1'], stat)()
        test_val = getattr(test_df['f1'], stat)()
        delta = abs(train_val - test_val)
        delta_pct = (delta / train_val * 100) if train_val != 0 else 0
        lines.append(f"{stat:20s} {train_val:8.4f}   {test_val:8.4f}   {delta:8.4f}   {delta_pct:6.2f}%")
    
    lines.append("")
    lines.append("DISTRIBUZIONE CLUSTER:")
    lines.append("-"*80)
    lines.append(f"Cluster    Train (n)  Train (%)   Test (n)   Test (%)    Delta")
    lines.append("-"*80)
    for cluster in sorted(train_df['cluster'].unique()):
        train_n = (train_df['cluster'] == cluster).sum()
        train_pct = train_n / len(train_df) * 100
        test_n = (test_df['cluster'] == cluster).sum()
        test_pct = test_n / len(test_df) * 100
        delta = abs(train_pct - test_pct)
        lines.append(f"{cluster:7d}    {train_n:8d}   {train_pct:7.2f}%   {test_n:8d}   {test_pct:7.2f}%   {delta:6.2f}%")
    
    lines.append("")
    lines.append("="*80)
    lines.append("COME USARE QUESTI DATASET")
    lines.append("="*80)
    lines.append("")
    lines.append("TRAIN SET (80% - per sviluppo modello):")
    lines.append("  1. Hyperparameter tuning con cross-validation")
    lines.append("  2. Feature engineering e selezione")
    lines.append("  3. Training modello finale")
    lines.append("")
    lines.append("TEST SET (20% - SACRED HOLDOUT):")
    lines.append("  1. NON guardare durante tuning")
    lines.append("  2. NON usare per prendere decisioni sul modello")
    lines.append("  3. Usare SOLO UNA VOLTA per valutazione finale")
    lines.append("  4. Simula performance su dati mai visti")
    lines.append("")
    lines.append("="*80)
    lines.append("TRAIN AUTHORS:")
    lines.append("-"*80)
    for i, author in enumerate(sorted(train_df.index), 1):
        f1 = train_df.loc[author, 'f1']
        cluster = int(train_df.loc[author, 'cluster'])
        lines.append(f"  {i:2d}. {author:30s} F1={f1:.4f} Cluster={cluster}")
    
    lines.append("")
    lines.append("TEST AUTHORS:")
    lines.append("-"*80)
    for i, author in enumerate(sorted(test_df.index), 1):
        f1 = test_df.loc[author, 'f1']
        cluster = int(test_df.loc[author, 'cluster'])
        lines.append(f"  {i:2d}. {author:30s} F1={f1:.4f} Cluster={cluster}")
    
    lines.append("")
    lines.append("="*80)
    
    return '\n'.join(lines)


def main():
    """Entry point"""
    
    # 1. Carica dataset
    df = load_dataset()
    if df is None:
        return
    
    # 2. Crea bin stratificazione
    df_with_bins = create_stratification_bins(df)
    
    # 3. Split
    train_df, test_df = perform_split(df_with_bins)
    
    # 4. Valida
    validate_split(train_df, test_df, df)
    
    # 5. Salva
    save_splits(train_df, test_df)
    
    print("\n" + "="*80)
    print("✓ SPLIT COMPLETATO CON SUCCESSO!")
    print("="*80)
    print(f"\nRisultati in: {OUTPUT_DIR}")
    print(f"\nFile generati:")
    print(f"  • train_set.csv      - {len(train_df)} autori per hyperparameter tuning")
    print(f"  • test_set.csv       - {len(test_df)} autori per valutazione finale")
    print(f"  • train_authors.txt  - lista nomi train")
    print(f"  • test_authors.txt   - lista nomi test")
    print(f"  • split_report.txt   - report dettagliato")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
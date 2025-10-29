"""
Split dataset 80/20 CONSAPEVOLE
Basato su features estratte da TUTTI gli autori (train + test primo split)

PREREQUISITO: Aver eseguito debug_graphs_and_expand.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from os.path import join, exists
import os

# ======================== CONFIGURAZIONE ========================
BASE_PATH = r"C:\Users\franc\OneDrive - Alma Mater Studiorum Università di Bologna\Desktop\BOND-OC\WhoIsWho\bond"

# Input: CSV con features estratte per TUTTI gli autori
INPUT_CSV = join(BASE_PATH, "bond_full_features_raw", "full_features_raw.csv")

# Output
OUTPUT_DIR = join(BASE_PATH, "bond_train_test_split_b3")

TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
RANDOM_SEED = 42
# ================================================================


def load_dataset():
    """Carica dataset con features"""
    print("="*80)
    print("STRATIFIED TRAIN/TEST SPLIT (80/20) - V2 CONSAPEVOLE")
    print("Basato su features estratte post-BOND")
    print("="*80)
    print(f"\n[1/5] Caricamento dataset...")
    
    if not exists(INPUT_CSV):
        print(f"ERRORE: File non trovato: {INPUT_CSV}")
        print(f"\nDevi prima eseguire:")
        print(f"  python debug_graphs_and_expand_FULL.py")
        print(f"\nQuesto script analizza TUTTI gli autori (train + test primo split)")
        return None
    
    df = pd.read_csv(INPUT_CSV, index_col=0)
    print(f"      Dataset: {len(df)} autori, {len(df.columns)} colonne")
    
    # Verifica colonne essenziali
    if 'f1' not in df.columns:
        print(f"      ❌ ERRORE: Colonna 'f1' mancante!")
        return None
    
    print(f"      ✓ F1 score range: [{df['f1'].min():.4f}, {df['f1'].max():.4f}]")
    
    return df


def create_stratification_bins(df):
    """Crea bin per stratificazione su più dimensioni"""
    print(f"\n[2/5] Creazione bin per stratificazione...")
    
    # 1. Bin F1 score (performance) - adattivo al dataset size
    n_bins = min(4, max(2, len(df) // 20))  # 2-4 bin a seconda del size
    
    try:
        df['f1_bin'] = pd.qcut(df['f1'], q=n_bins, labels=False, duplicates='drop')
        print(f"      ✓ F1 bins: {df['f1_bin'].nunique()} livelli (qcut)")
    except ValueError:
        df['f1_bin'] = pd.cut(df['f1'], bins=n_bins, labels=False)
        print(f"      ⚠️  F1 bins: {df['f1_bin'].nunique()} livelli (cut - fallback)")
    
    # 2. Bin su feature più correlata con F1 (se disponibile)
    feature_cols = [c for c in df.columns if c not in ['f1', 'precision', 'recall', 'cluster']]
    
    if len(feature_cols) > 0:
        # Trova feature più correlata
        correlations = df[feature_cols].corrwith(df['f1']).abs()
        top_feature = correlations.idxmax()
        
        print(f"      Top feature correlata: {top_feature} (r={correlations[top_feature]:.3f})")
        
        # Crea bin su questa feature
        try:
            df['feature_bin'] = pd.qcut(df[top_feature], q=min(3, len(df)//15), labels=False, duplicates='drop')
            print(f"      ✓ Feature bins: {df['feature_bin'].nunique()} livelli")
        except ValueError:
            df['feature_bin'] = pd.cut(df[top_feature], bins=min(3, len(df)//15), labels=False)
            print(f"      ⚠️  Feature bins: {df['feature_bin'].nunique()} livelli (cut)")
        
        # Stratification key combinata
        df['stratify_key'] = (
            df['f1_bin'].astype(str) + '_' + 
            df['feature_bin'].astype(str)
        )
    else:
        # Solo F1
        df['stratify_key'] = df['f1_bin'].astype(str)
    
    print(f"      ✓ Chiavi di stratificazione univoche: {df['stratify_key'].nunique()}")
    
    # Mostra distribuzione F1 bins
    print(f"\n      Distribuzione F1 bins:")
    for bin_name in sorted(df['f1_bin'].unique()):
        count = (df['f1_bin'] == bin_name).sum()
        f1_range = df[df['f1_bin'] == bin_name]['f1']
        print(f"        Bin {bin_name}: {count:3d} autori (F1: {f1_range.min():.3f}-{f1_range.max():.3f})")
    
    return df


def perform_split(df):
    """Esegue split stratificato"""
    print(f"\n[3/5] Split stratificato ({TRAIN_RATIO:.0%} train / {TEST_RATIO:.0%} test)...")
    
    # Identifica chiavi con 1 solo sample (non stratificabili)
    stratify_counts = df['stratify_key'].value_counts()
    rare_keys = stratify_counts[stratify_counts == 1].index
    
    if len(rare_keys) > 0:
        print(f"      ⚠️  {len(rare_keys)} chiavi con 1 solo sample")
        print(f"      Soluzione: split casuale per questi sample")
        
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
        if len(df_rare) > 0:
            train_rare, test_rare = train_test_split(
                df_rare,
                test_size=TEST_RATIO,
                random_state=RANDOM_SEED
            )
            
            # Combina
            train_df = pd.concat([train_strat, train_rare])
            test_df = pd.concat([test_strat, test_rare])
        else:
            train_df = train_strat
            test_df = test_strat
    else:
        # Split stratificato standard
        train_df, test_df = train_test_split(
            df,
            test_size=TEST_RATIO,
            random_state=RANDOM_SEED,
            stratify=df['stratify_key']
        )
    
    # Rimuovi colonne temporanee
    cols_to_drop = ['f1_bin', 'feature_bin', 'stratify_key']
    train_df = train_df.drop(columns=cols_to_drop, errors='ignore')
    test_df = test_df.drop(columns=cols_to_drop, errors='ignore')
    
    print(f"      ✓ Train: {len(train_df)} autori ({len(train_df)/len(df)*100:.1f}%)")
    print(f"      ✓ Test:  {len(test_df)} autori ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, test_df


def validate_split(train_df, test_df, original_df):
    """Valida rappresentatività dello split"""
    print(f"\n[4/5] Validazione split...")
    
    from scipy.stats import ks_2samp
    
    print(f"\n      Confronto distribuzioni (KS-test):")
    print(f"      {'Metrica':30s} Train      Test       p-value  Status")
    print(f"      {'-'*75}")
    
    # F1 score
    ks_stat, ks_pval = ks_2samp(train_df['f1'], test_df['f1'])
    status = '✓ OK' if ks_pval > 0.05 else '⚠️ DIFF'
    print(f"      {'F1 score':30s} {train_df['f1'].mean():8.4f}   {test_df['f1'].mean():8.4f}   {ks_pval:.4f}   {status}")
    
    # Top features (se disponibili)
    feature_cols = [c for c in train_df.columns if c not in ['f1', 'precision', 'recall']]
    
    if len(feature_cols) > 0:
        # Top 3 features correlate con F1
        correlations = train_df[feature_cols].corrwith(train_df['f1']).abs().sort_values(ascending=False)
        
        for feat in correlations.head(3).index:
            ks_stat, ks_pval = ks_2samp(train_df[feat], test_df[feat])
            status = '✓ OK' if ks_pval > 0.05 else '⚠️ DIFF'
            feat_name = feat[:28]  # Truncate se troppo lungo
            print(f"      {feat_name:30s} {train_df[feat].mean():8.4f}   {test_df[feat].mean():8.4f}   {ks_pval:.4f}   {status}")
    
    print(f"\n      Interpretazione KS-test:")
    print(f"        ✓ OK:   Distribuzioni simili (p > 0.05) - BUONO")
    print(f"        ⚠️ DIFF: Distribuzioni diverse (p < 0.05) - ATTENZIONE")


def save_splits(train_df, test_df):
    """Salva i dataset splittati"""
    print(f"\n[5/5] Salvataggio split...")
    
    if not exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # CSV completi (con features)
    train_file = join(OUTPUT_DIR, 'train_set_v2.csv')
    test_file = join(OUTPUT_DIR, 'test_set_v2.csv')
    
    train_df.to_csv(train_file)
    test_df.to_csv(test_file)
    
    print(f"      ✓ train_set_v2.csv ({len(train_df)} autori)")
    print(f"      ✓ test_set_v2.csv ({len(test_df)} autori)")
    
    # Liste nomi autori (per generare JSON dopo)
    train_authors_file = join(OUTPUT_DIR, 'train_authors_v2.txt')
    test_authors_file = join(OUTPUT_DIR, 'test_authors_v2.txt')
    
    with open(train_authors_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_df.index.tolist()))
    
    with open(test_authors_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_df.index.tolist()))
    
    print(f"      ✓ train_authors_v2.txt")
    print(f"      ✓ test_authors_v2.txt")
    
    # Report dettagliato
    report = generate_report(train_df, test_df)
    report_file = join(OUTPUT_DIR, 'split_report_v2.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"      ✓ split_report_v2.txt")


def generate_report(train_df, test_df):
    """Genera report completo"""
    lines = []
    lines.append("="*80)
    lines.append("TRAIN/TEST SPLIT REPORT V2 - CONSAPEVOLE")
    lines.append("Split stratificato basato su features post-BOND")
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
    lines.append("TRAIN AUTHORS:")
    lines.append("-"*80)
    for i, author in enumerate(sorted(train_df.index), 1):
        f1 = train_df.loc[author, 'f1']
        lines.append(f"  {i:3d}. {author:40s} F1={f1:.4f}")
    
    lines.append("")
    lines.append("TEST AUTHORS:")
    lines.append("-"*80)
    for i, author in enumerate(sorted(test_df.index), 1):
        f1 = test_df.loc[author, 'f1']
        lines.append(f"  {i:3d}. {author:40s} F1={f1:.4f}")
    
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
    print("✅ SPLIT CONSAPEVOLE COMPLETATO!")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"\nFile generati:")
    print(f"  • train_set_v2.csv         - {len(train_df)} autori + features")
    print(f"  • test_set_v2.csv          - {len(test_df)} autori + features")
    print(f"  • train_authors_v2.txt     - lista nomi train")
    print(f"  • test_authors_v2.txt      - lista nomi test")
    print(f"  • split_report_v2.txt      - report dettagliato")
    print(f"\n{'='*80}")
    print("  Questo split è STRATIFICATO su:")
    print("    1. Performance (F1 score)")
    print("    2. Feature tecnica più correlata")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
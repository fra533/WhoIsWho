"""
BOND Demo Script - Pulito e compatibile con hyperopt
"""
import os
from training.autotrain_bond import BONDTrainer
from training.autotrain_bond_ensemble import ESBTrainer
from dataset.preprocess_SND import dump_name_pubs, dump_features_relations_to_file, build_graph
from params import set_params

args = set_params()


def check_preprocessing_done(mode):
    """
    Controlla se il preprocessing è già stato fatto per questo mode.
    """
    from pathlib import Path
    
    base_path = Path(args.save_path)
    
    # Controlla le directory principali create dal preprocessing
    names_pub_dir = base_path / 'names_pub' / mode
    relations_dir = base_path / 'relations' / mode
    graphs_dir = base_path / 'graph' / mode  # ← CORRETTO: 'graph' non 'graphs'
    
    # Verifica che esistano E non siano vuote
    for dir_path in [names_pub_dir, relations_dir, graphs_dir]:
        if not dir_path.exists() or not any(dir_path.iterdir()):
            return False
    
    return True


def pipeline(model):
    """
    Pipeline BOND con preprocessing intelligente.
    """
    print("="*70)
    print(f"BOND PIPELINE - Mode: {args.mode}")
    print("="*70)
    
    # Controlla se saltare il preprocessing
    skip_preprocess = os.getenv("SKIP_PREPROCESS") == "1"
    
    # Module-1: Data Loading (solo se necessario)
    if skip_preprocess:
        print("\n⏭️  Skipping preprocessing (SKIP_PREPROCESS=1)")
    #elif check_preprocessing_done(args.mode):
     #   print("\n✓ Preprocessing already done, skipping...")
    else:
        print("\n🔄 Running preprocessing...")
        print("\n[1/3] Loading and dumping name publications...")
        #dump_name_pubs()
        
        print("\n[2/3] Creating features and relations...")
        #dump_features_relations_to_file()
        
        print("\n[3/3] Building graph...")
        build_graph()
        
        print("\n✓ Preprocessing completed!")
    
    print("\n" + "="*70)
    print("Starting model training...")
    print("="*70 + "\n")
    
    # Modules-2: Feature Creation & Module-3: Model Construction
    if model == 'bond':
        trainer = BONDTrainer()
        trainer.fit(datatype=args.mode)
    elif model == 'bond+':
        trainer = ESBTrainer()
        trainer.fit(datatype=args.mode)

    # Modules-4: Evaluation
    # Please upload your result to http://whoiswho.biendata.xyz/#/
    print("\n✓ Training completed!")
    print(f"Results saved in: out/res.json")


if __name__ == "__main__":
    pipeline(model="bond")
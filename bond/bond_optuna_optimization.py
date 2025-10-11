"""
Script di ottimizzazione iperparametri per BOND
Pulito e compatibile con il demo.py di BOND
Con supporto per validazione sui migliori parametri
"""
import optuna
import json
import subprocess
import os
import sys
import time
import shutil
from datetime import datetime, timedelta
from optuna.samplers import TPESampler
from pathlib import Path

# ======================== CONFIGURAZIONE ========================
BASE_PATH = Path(__file__).parent
DEMO_SCRIPT = BASE_PATH / "demo.py"
DATA_PATH = BASE_PATH / "dataset" / "data"

RESULTS_DIR = BASE_PATH / "hyperopt_results"
RESULTS_DIR.mkdir(exist_ok=True)
# ================================================================


class ProgressTracker:
    """Tracker per mostrare il progresso dell'ottimizzazione"""
    def __init__(self, n_trials):
        self.n_trials = n_trials
        self.start_time = time.time()
        self.best_value = 0.0
        self.current_trial = 0
        
    def __call__(self, study, trial):
        self.current_trial += 1
        elapsed = time.time() - self.start_time
        
        if self.current_trial > 1:
            avg_time = elapsed / self.current_trial
            remaining = (self.n_trials - self.current_trial) * avg_time
            eta = str(timedelta(seconds=int(remaining)))
        else:
            eta = "Calculating..."
        
        if trial.value and trial.value > self.best_value:
            self.best_value = trial.value
        
        progress = (self.current_trial / self.n_trials) * 100
        print(f"\n{'='*70}")
        print(f"Progress: Trial {self.current_trial}/{self.n_trials} ({progress:.1f}%)")
        print(f"Best F1 so far: {self.best_value:.4f}")
        print(f"ETA: {eta}")
        print(f"{'='*70}\n")


def evaluate_predictions(predictions_file, ground_truth_file):
    """Valuta le predizioni usando F1 pairwise"""
    try:
        with open(predictions_file, 'r', encoding='utf-8') as f:
            predict_result = json.load(f)
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        
        # Filtra solo nomi comuni
        filtered_predict = {n: p for n, p in predict_result.items() 
                           if n in ground_truth}
        
        if not filtered_predict:
            print("  WARNING: No common names between predictions and ground truth")
            print(f"  Predictions has {len(predict_result)} names")
            print(f"  Ground truth has {len(ground_truth)} names")
            if predict_result and ground_truth:
                print(f"  Sample prediction names: {list(predict_result.keys())[:3]}")
                print(f"  Sample ground truth names: {list(ground_truth.keys())[:3]}")
            return 0.0

        f1_scores = []
        for name in filtered_predict:
            # Crea mapping predizioni
            predicted_pubs = {}
            for idx, pids in enumerate(filtered_predict[name]):
                for pid in pids:
                    predicted_pubs[pid] = idx

            # Estrai labels vere
            pubs, true_labels = [], []
            ilabel = 0
            gt = ground_truth[name]

            if isinstance(gt, dict):
                for aid in gt:
                    pubs.extend(gt[aid])
                    true_labels.extend([ilabel] * len(gt[aid]))
                    ilabel += 1
            elif isinstance(gt, list):
                for cluster in gt:
                    if isinstance(cluster, list):
                        pubs.extend(cluster)
                        true_labels.extend([ilabel] * len(cluster))
                        ilabel += 1

            # Filtra paper comuni
            filtered_pubs = [pid for pid in pubs if pid in predicted_pubs]
            if not filtered_pubs:
                continue

            filtered_labels = [true_labels[i] for i, pid in enumerate(pubs) 
                             if pid in predicted_pubs]
            predicted_labels = [predicted_pubs[pid] for pid in filtered_pubs]

            # Calcola pairwise metrics
            TP = TP_FP = TP_FN = 0
            for i in range(len(filtered_labels)):
                for j in range(i + 1, len(filtered_labels)):
                    if filtered_labels[i] == filtered_labels[j]:
                        TP_FN += 1
                    if predicted_labels[i] == predicted_labels[j]:
                        TP_FP += 1
                    if (filtered_labels[i] == filtered_labels[j]) and \
                       (predicted_labels[i] == predicted_labels[j]):
                        TP += 1

            if TP == 0:
                continue

            precision = TP / TP_FP if TP_FP > 0 else 0
            recall = TP / TP_FN if TP_FN > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) \
                 if (precision + recall) > 0 else 0

            f1_scores.append(f1)

        if not f1_scores:
            print("  WARNING: No valid F1 scores computed")
            return 0.0

        avg_f1 = sum(f1_scores) / len(f1_scores)
        print(f"  Evaluation: F1={avg_f1:.4f} ({len(f1_scores)} names evaluated)")
        return avg_f1

    except Exception as e:
        print(f"  ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def check_preprocessing():
    """Verifica che il preprocessing sia stato fatto"""
    train_names_pub = DATA_PATH / "names_pub" / "train"
    
    if not train_names_pub.exists() or not any(train_names_pub.iterdir()):
        print("\nERROR: Preprocessing not found!")
        print("Please run preprocessing first:")
        print("  python -m dataset.preprocess_SND")
        return False
    
    print("✓ Preprocessing data found")
    return True


def setup_validation_structure():
    """
    Prepara la struttura per il validation set
    Crea copia da sna-valid a valid se necessario (senza admin)
    
    Returns:
        tuple: (success, cleanup_needed, paths_created)
    """
    sna_valid_src = DATA_PATH / "src" / "sna-valid"
    std_valid_src = DATA_PATH / "src" / "valid"
    graph_valid = DATA_PATH / "graph" / "valid"
    
    paths_to_cleanup = []
    
    # Verifica che sna-valid esista
    if not sna_valid_src.exists():
        print(f"  ERROR: {sna_valid_src} not found")
        return False, False, paths_to_cleanup
    
    # Verifica che graph/valid esista
    if not graph_valid.exists():
        print(f"  ERROR: {graph_valid} not found")
        print(f"  The graph preprocessing for validation is missing!")
        return False, False, paths_to_cleanup
    
    print(f"  ✓ Found graph/valid with preprocessing")
    
    # Se src/valid esiste già, assumiamo sia corretto
    if std_valid_src.exists():
        print(f"  ✓ src/valid already exists")
        return True, False, paths_to_cleanup
    
    # Crea src/valid da sna-valid (prima prova symlink, poi copia)
    print(f"  Creating src/valid from src/sna-valid...")
    try:
        # Prova symlink
        std_valid_src.symlink_to(sna_valid_src, target_is_directory=True)
        print(f"  ✓ Symbolic link created: src/valid -> src/sna-valid")
        paths_to_cleanup.append(('symlink', std_valid_src))
    except (OSError, NotImplementedError, PermissionError):
        # Fallback: copia (funziona senza admin)
        print(f"  Symlink failed, copying directory...")
        shutil.copytree(sna_valid_src, std_valid_src)
        print(f"  ✓ Directory copied: src/sna-valid -> src/valid")
        paths_to_cleanup.append(('copy', std_valid_src))
    
    return True, True, paths_to_cleanup


def cleanup_validation_structure(paths_to_cleanup):
    """Rimuove i file/link temporanei creati per la validazione"""
    for link_type, path in paths_to_cleanup:
        try:
            if link_type == 'symlink' and path.is_symlink():
                path.unlink()
                print(f"  ✓ Cleaned up symbolic link: {path.name}")
            elif link_type == 'copy' and path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                print(f"  ✓ Cleaned up copy: {path.name}")
        except Exception as e:
            print(f"  Warning: Could not cleanup {path}: {e}")


def run_bond_validation(params):
    """
    Esegue BOND in modalità validation con i parametri forniti
    
    Args:
        params: dizionario con i parametri
    
    Returns:
        F1 score sul validation set
    """
    print("\n" + "="*70)
    print("RUNNING VALIDATION")
    print("="*70)
    
    # Setup struttura validation
    print("\nSetting up validation structure...")
    success, needs_cleanup, paths_to_cleanup = setup_validation_structure()
    
    if not success:
        print("Failed to setup validation structure")
        return 0.0
    
    try:
        # Trova ground truth
        sna_valid_dir = DATA_PATH / "src" / "sna-valid"
        possible_gt_names = [
            "mio_valid_ground_truth.json",
            "valid_ground_truth.json",
            "sna_valid_ground_truth.json"
        ]
        
        ground_truth_file = None
        for name in possible_gt_names:
            candidate = sna_valid_dir / name
            if candidate.exists():
                ground_truth_file = candidate
                break
        
        if ground_truth_file is None:
            print(f"\nERROR: Ground truth file not found in {sna_valid_dir}")
            print(f"Tried: {possible_gt_names}")
            return 0.0
        
        print(f"✓ Ground truth: {ground_truth_file.name}")
        
        # Environment
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env["SKIP_PREPROCESS"] = "1"
        
        # Directory output
        output_dir = RESULTS_DIR / "validation"
        output_dir.mkdir(exist_ok=True)
        
        # Comando BOND
        cmd = [
            'python', str(DEMO_SCRIPT),
            '--mode', 'valid',
            '--save_path', str(DATA_PATH),
            '--db_eps', str(params['db_eps']),
            '--db_min', str(params['db_min']),
            '--cluster_w', str(params['cluster_w']),
            '--lr', str(params['lr']),
            '--l2_coef', str(params['l2_coef']),
            '--hidden_dim', str(params['hidden_dim_0']), str(params['hidden_dim_1']),
            '--compress_ratio', str(params['compress_ratio']),
            '--th_a', str(params['th_a_0']), str(params['th_a_1']),
            '--th_o', str(params['th_o_0']), str(params['th_o_1']),
            '--th_v', str(params['th_v_0']), str(params['th_v_1']),
            '--epochs', str(params.get('epochs', 50))
        ]
        
        print(f"\nRunning BOND validation...")
        print(f"Parameters: db_eps={params['db_eps']}, cluster_w={params['cluster_w']}, lr={params['lr']:.2e}")
        
        # Esegui BOND
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=7200
        )
        
        if result.returncode != 0:
            print(f"\nWARNING: BOND exited with code {result.returncode}")
            # Salva log
            with open(output_dir / "stdout.txt", 'w', encoding='utf-8') as f:
                f.write(result.stdout)
            with open(output_dir / "stderr.txt", 'w', encoding='utf-8') as f:
                f.write(result.stderr)
            
            # Mostra ultimi errori
            error_lines = result.stderr.strip().split('\n')
            if error_lines:
                print("\nLast error lines:")
                for line in error_lines[-5:]:
                    print(f"  {line}")
        
        # Leggi predizioni
        predictions_file = BASE_PATH / "out" / "res.json"
        
        if not predictions_file.exists():
            print(f"\nERROR: Predictions file not found: {predictions_file}")
            print(f"Check logs in {output_dir}")
            return 0.0
        
        # Copia predizioni
        output_predictions = output_dir / "predictions.json"
        shutil.copy2(predictions_file, output_predictions)
        
        # Valuta
        print(f"\nEvaluating validation set...")
        f1 = evaluate_predictions(output_predictions, ground_truth_file)
        
        # Salva risultati
        result_data = {
            "split": "valid",
            "f1_score": f1,
            "params": params,
            "ground_truth_file": str(ground_truth_file),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(output_dir / "results.json", 'w') as f:
            json.dump(result_data, f, indent=2)
        
        return f1
        
    finally:
        # Cleanup se necessario
        if needs_cleanup:
            print("\nCleaning up temporary files...")
            cleanup_validation_structure(paths_to_cleanup)


def objective(trial):
    """Funzione obiettivo per Optuna"""
    
    # Parametri da ottimizzare
    params = {
        'db_eps': trial.suggest_float('db_eps', 0.05, 0.3, step=0.01),
        'db_min': trial.suggest_int('db_min', 3, 10),
        'cluster_w': trial.suggest_float('cluster_w', 0.1, 0.9, step=0.1),
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'l2_coef': trial.suggest_float('l2_coef', 1e-5, 1e-3, log=True),
        'hidden_dim_0': trial.suggest_categorical('hidden_dim_0', [128, 256, 512]),
        'hidden_dim_1': trial.suggest_categorical('hidden_dim_1', [256, 512, 1024]),
        'compress_ratio': trial.suggest_float('compress_ratio', 0.5, 1.0, step=0.1),
        'th_a_0': trial.suggest_float('th_a_0', 0.0, 0.5),
        'th_a_1': trial.suggest_float('th_a_1', 0.5, 1.5),
        'th_o_0': trial.suggest_float('th_o_0', 0.3, 0.8),
        'th_o_1': trial.suggest_float('th_o_1', 0.3, 0.8),
        'th_v_0': trial.suggest_float('th_v_0', 0.5, 2.0),
        'th_v_1': trial.suggest_float('th_v_1', 1.0, 3.0),
        'epochs': 50  # Fisso per velocizzare
    }

    # Environment con skip preprocessing
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env["SKIP_PREPROCESS"] = "1"

    # Directory per questo trial
    trial_dir = RESULTS_DIR / f"trial_{trial.number:03d}"
    trial_dir.mkdir(exist_ok=True)

    # Comando per eseguire BOND
    cmd = [
        'python', str(DEMO_SCRIPT),
        '--mode', 'train',
        '--save_path', str(DATA_PATH),
        '--db_eps', str(params['db_eps']),
        '--db_min', str(params['db_min']),
        '--cluster_w', str(params['cluster_w']),
        '--lr', str(params['lr']),
        '--l2_coef', str(params['l2_coef']),
        '--hidden_dim', str(params['hidden_dim_0']), str(params['hidden_dim_1']),
        '--compress_ratio', str(params['compress_ratio']),
        '--th_a', str(params['th_a_0']), str(params['th_a_1']),
        '--th_o', str(params['th_o_0']), str(params['th_o_1']),
        '--th_v', str(params['th_v_0']), str(params['th_v_1']),
        '--epochs', str(params['epochs'])
    ]

    print(f"\nTrial {trial.number} - Starting BOND training...")
    print(f"Parameters: db_eps={params['db_eps']}, cluster_w={params['cluster_w']}, "
          f"lr={params['lr']:.2e}")
    
    try:
        # Esegui BOND
        result = subprocess.run(
            cmd, 
            env=env,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=7200
        )
        
        if result.returncode != 0:
            print(f"  WARNING: BOND exited with code {result.returncode}")
            with open(trial_dir / "error.txt", 'w') as f:
                f.write(result.stderr)
        
        # BOND salva le predizioni in out/res.json
        predictions_file = BASE_PATH / "out" / "res.json"
        ground_truth_file = DATA_PATH / "src" / "train" / "train_ground_truth.json"
        
        if not predictions_file.exists():
            print(f"  ERROR: Predictions file not found: {predictions_file}")
            return 0.0
        
        # Copia predizioni nella directory del trial
        trial_predictions = trial_dir / "predictions.json"
        shutil.copy2(predictions_file, trial_predictions)
        
        # Valuta
        print(f"  Evaluating trial {trial.number}...")
        f1 = evaluate_predictions(trial_predictions, ground_truth_file)
        
        # Salva risultati
        result_data = {
            "trial": trial.number,
            "f1_score": f1,
            "params": params,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(trial_dir / "results.json", 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"  Trial {trial.number}: F1={f1:.4f}")
        return f1

    except subprocess.TimeoutExpired:
        print(f"  ERROR: Trial {trial.number} timed out (>2 hours)")
        return 0.0
    except Exception as e:
        print(f"  ERROR in trial {trial.number}: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def validate_best_params():
    """Valida i migliori parametri sul validation set"""
    best_params_file = RESULTS_DIR / 'best_params.json'
    
    if not best_params_file.exists():
        print(f"\nERROR: Best parameters file not found: {best_params_file}")
        print("Please run optimization first!")
        return
    
    # Carica migliori parametri
    with open(best_params_file, 'r') as f:
        best_params = json.load(f)
    
    print("="*70)
    print("VALIDATION ON BEST PARAMETERS")
    print("="*70)
    print("\nBest parameters from training:")
    print(f"  Train F1: {best_params.get('best_f1_score', 'N/A')}")
    print(f"  db_eps: {best_params['db_eps']}")
    print(f"  cluster_w: {best_params['cluster_w']}")
    print(f"  lr: {best_params['lr']:.2e}")
    
    # Esegui validazione
    valid_f1 = run_bond_validation(best_params)
    
    # Salva risultati
    validation_results = {
        "train_f1": best_params.get('best_f1_score'),
        "valid_f1": valid_f1,
        "params": best_params,
        "validation_date": datetime.now().isoformat()
    }
    
    results_file = RESULTS_DIR / 'validation_results.json'
    with open(results_file, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETED!")
    print("="*70)
    print(f"Train F1:      {best_params.get('best_f1_score', 'N/A')}")
    print(f"Validation F1: {valid_f1:.4f}")
    print(f"\nResults saved to: {results_file}")


def optimize_hyperparameters():
    """Esegue l'ottimizzazione degli iperparametri"""
    print("="*70)
    print("BOND HYPERPARAMETER OPTIMIZATION")
    print("="*70)
    
    # Verifica preprocessing
    if not check_preprocessing():
        return
    
    # Verifica struttura dati
    train_dir = DATA_PATH / "src" / "train"
    if not train_dir.exists():
        print(f"\nERROR: Train directory not found: {train_dir}")
        return
    
    print(f"✓ Train directory: {train_dir}")
    
    # Chiedi numero trial
    n_trials = input("\nNumber of trials (default 10): ").strip()
    n_trials = int(n_trials) if n_trials else 10
    
    print(f"\nStarting optimization:")
    print(f"  - Trials: {n_trials}")
    print(f"  - Epochs per trial: 50 (fixed)")
    print(f"  - Estimated time: {n_trials * 0.5:.1f} hours")
    print(f"  - Results will be saved in: {RESULTS_DIR}")
    
    input("\nPress ENTER to start...")
    
    # Crea study Optuna
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        study_name="bond_hyperopt"
    )

    # Ottimizza
    progress = ProgressTracker(n_trials)
    study.optimize(objective, n_trials=n_trials, callbacks=[progress])

    # Risultati finali
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETED!")
    print("="*70)
    print(f"\nBest trial F1: {study.best_value:.4f}")
    print("\nBest parameters:")
    print(json.dumps(study.best_params, indent=2))
    
    # Salva migliori parametri
    best_params = study.best_params.copy()
    best_params['best_f1_score'] = study.best_value
    best_params['optimization_date'] = datetime.now().isoformat()
    
    with open(RESULTS_DIR / 'best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"\nBest parameters saved to: {RESULTS_DIR / 'best_params.json'}")
    print(f"All trial results in: {RESULTS_DIR}/")
    
    # Chiedi se fare validazione
    print("\n" + "="*70)
    response = input("\nRun validation on best parameters? (y/n): ").strip().lower()
    if response == 'y':
        validate_best_params()


def main():
    """Entry point"""
    if len(sys.argv) > 1:
        if sys.argv[1] == '--validate':
            validate_best_params()
        elif sys.argv[1] == '--help':
            print("Usage:")
            print("  python bond_optuna_optimization.py           # Run optimization")
            print("  python bond_optuna_optimization.py --validate # Validate best params")
            print("  python bond_optuna_optimization.py --help     # Show this help")
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        optimize_hyperparameters()


if __name__ == "__main__":
    main()
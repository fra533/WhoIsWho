"""
Script di ottimizzazione iperparametri per BOND
CON OTTIMIZZAZIONE PESI CITAZIONI (cite_in_weight, cite_out_weight)

Parametri ottimizzati:
- Parametri standard BOND (db_eps, cluster_w, lr, hidden_dim, ecc.)
- CITAZIONI: cite_out_weight, cite_in_weight, cite_out_th, cite_in_th
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
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# CITAZIONI - Range da esplorare
# Questi parametri sono IMPORTANTI per sfruttare le citazioni!

CITATION_CONFIG = {
    'use_citations': True,
    'cite_out_weight': (0.5, 1.5),  # ← Range più stretto, centrato su 1.0
    'cite_in_weight': (0.5, 1.5),   # ← Evita lo zero!
    'cite_out_th': (0.0, 0.2),      # ← Soglie più basse
    'cite_in_th': (0.0, 0.2)        # ← Soglie più basse
}
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
            else:
                continue

            # Crea predicted labels
            predicted_labels = []
            for pid in pubs:
                if pid in predicted_pubs:
                    predicted_labels.append(predicted_pubs[pid])
                else:
                    predicted_labels.append(-1)

            # Calcola F1 pairwise
            tp = fp = fn = 0
            n = len(pubs)
            for i in range(n):
                for j in range(i + 1, n):
                    true_same = (true_labels[i] == true_labels[j])
                    pred_same = (predicted_labels[i] == predicted_labels[j] and 
                               predicted_labels[i] != -1)
                    
                    if true_same and pred_same:
                        tp += 1
                    elif pred_same and not true_same:
                        fp += 1
                    elif true_same and not pred_same:
                        fn += 1

            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0.0

            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0.0

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            f1_scores.append(f1)

        if f1_scores:
            avg_f1 = sum(f1_scores) / len(f1_scores)
            return avg_f1
        else:
            return 0.0

    except Exception as e:
        print(f"  ERROR in evaluation: {e}")
        return 0.0


def check_preprocessing():
    """Verifica che il preprocessing sia stato fatto"""
    train_dir = DATA_PATH / "src" / "train"
    
    required_files = [
        train_dir / "train_pub.json",
        train_dir / "train_author.json"
    ]
    
    # Cerca file preprocessati
    graph_files = list(train_dir.glob("*.pt")) + list(train_dir.glob("*.pkl"))
    
    if not all(f.exists() for f in required_files):
        print("ERROR: Train files not found")
        return False
    
    if not graph_files:
        print("WARNING: No preprocessed graph files found (.pt, .pkl)")
        print("         Make sure to run preprocessing first!")
        return False
    
    return True


def setup_validation_structure():
    """Setup struttura per validation"""
    sna_valid_src = DATA_PATH / "src" / "sna-valid"
    std_valid_src = DATA_PATH / "src" / "valid"
    
    paths_to_cleanup = []
    
    if not sna_valid_src.exists():
        print(f"ERROR: sna-valid directory not found: {sna_valid_src}")
        return False, False, paths_to_cleanup
    
    if std_valid_src.exists():
        return True, False, paths_to_cleanup
    
    try:
        std_valid_src.symlink_to(sna_valid_src, target_is_directory=True)
        paths_to_cleanup.append(('symlink', std_valid_src))
    except (OSError, NotImplementedError, PermissionError):
        shutil.copytree(sna_valid_src, std_valid_src)
        paths_to_cleanup.append(('copy', std_valid_src))
    
    return True, True, paths_to_cleanup


def cleanup_validation_structure(paths_to_cleanup):
    """Rimuove file/link temporanei"""
    for link_type, path in paths_to_cleanup:
        try:
            if link_type == 'symlink' and path.is_symlink():
                path.unlink()
            elif link_type == 'copy' and path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        except Exception as e:
            print(f"  Warning: Could not cleanup {path}: {e}")


def objective(trial):
    """
    Funzione obiettivo per Optuna
    
    INCLUDE OTTIMIZZAZIONE CITAZIONI!
    """
    # ============================================================
    # PARAMETRI DA OTTIMIZZARE (INCLUSE CITAZIONI)
    # ============================================================
    params = {
        # Clustering
        'db_eps': trial.suggest_float('db_eps', 0.05, 0.3, step=0.01),
        'db_min': trial.suggest_int('db_min', 3, 10),
        'cluster_w': trial.suggest_float('cluster_w', 0.1, 0.9, step=0.1),
        
        # Training
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'l2_coef': trial.suggest_float('l2_coef', 1e-5, 1e-3, log=True),
        
        # Architecture
        'hidden_dim_0': trial.suggest_categorical('hidden_dim_0', [128, 256, 512]),
        'hidden_dim_1': trial.suggest_categorical('hidden_dim_1', [256, 512, 1024]),
        'compress_ratio': trial.suggest_float('compress_ratio', 0.5, 1.0, step=0.1),
        
        # Thresholds
        'th_a_0': trial.suggest_float('th_a_0', 0.0, 0.5),
        'th_a_1': trial.suggest_float('th_a_1', 0.5, 1.5),
        'th_o_0': trial.suggest_float('th_o_0', 0.3, 0.8),
        'th_o_1': trial.suggest_float('th_o_1', 0.3, 0.8),
        'th_v_0': trial.suggest_float('th_v_0', 0.5, 2.0),
        'th_v_1': trial.suggest_float('th_v_1', 1.0, 3.0),
        
        # ⭐ CITAZIONI - PARAMETRI NUOVI ⭐
        'use_citations': True,  # Sempre abilitato
        'cite_out_weight': trial.suggest_float('cite_out_weight', 
                                               CITATION_CONFIG['cite_out_weight'][0],
                                               CITATION_CONFIG['cite_out_weight'][1],
                                               step=0.1),
        'cite_in_weight': trial.suggest_float('cite_in_weight',
                                              CITATION_CONFIG['cite_in_weight'][0],
                                              CITATION_CONFIG['cite_in_weight'][1],
                                              step=0.1),
        'cite_out_th': trial.suggest_float('cite_out_th',
                                           CITATION_CONFIG['cite_out_th'][0],
                                           CITATION_CONFIG['cite_out_th'][1],
                                           step=0.05),
        'cite_in_th': trial.suggest_float('cite_in_th',
                                          CITATION_CONFIG['cite_in_th'][0],
                                          CITATION_CONFIG['cite_in_th'][1],
                                          step=0.05),
        
        'epochs': 50  # Fisso
    }

    # Environment
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env["SKIP_PREPROCESS"] = "1"

    # Directory trial
    trial_dir = RESULTS_DIR / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # COMANDO BOND (CON PARAMETRI CITAZIONI)
    # ============================================================
    cmd = [
        'python', str(DEMO_SCRIPT),
        '--mode', 'train',
        '--save_path', str(DATA_PATH),
        
        # Clustering
        '--db_eps', str(params['db_eps']),
        '--db_min', str(params['db_min']),
        '--cluster_w', str(params['cluster_w']),
        
        # Training
        '--lr', str(params['lr']),
        '--l2_coef', str(params['l2_coef']),
        '--epochs', str(params['epochs']),
        
        # Architecture
        '--hidden_dim', str(params['hidden_dim_0']), str(params['hidden_dim_1']),
        '--compress_ratio', str(params['compress_ratio']),
        
        # Thresholds
        '--th_a', str(params['th_a_0']), str(params['th_a_1']),
        '--th_o', str(params['th_o_0']), str(params['th_o_1']),
        '--th_v', str(params['th_v_0']), str(params['th_v_1']),
        
        # ⭐ CITAZIONI ⭐
        #'--use_citations', str(params['use_citations']),
        '--cite_out_weight', str(params['cite_out_weight']),
        '--cite_in_weight', str(params['cite_in_weight']),
        '--cite_out_th', str(params['cite_out_th']),
        '--cite_in_th', str(params['cite_in_th'])
    ]

    print(f"\nTrial {trial.number} - Starting BOND training...")
    print(f"  Standard params: db_eps={params['db_eps']}, cluster_w={params['cluster_w']}, lr={params['lr']:.2e}")
    print(f"  ⭐ Citation params: out_w={params['cite_out_weight']:.2f}, in_w={params['cite_in_weight']:.2f}")
    
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

                
        
        # Predizioni
        predictions_file = BASE_PATH / "out" / "res.json"
        ground_truth_file = DATA_PATH / "src" / "train" / "train_author.json"
        
        if not predictions_file.exists():
            print(f"  ERROR: Predictions file not found: {predictions_file}")
            return 0.0
        
        # Copia predizioni
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
        print(f"    Citations: out_w={params['cite_out_weight']:.2f}, in_w={params['cite_in_weight']:.2f}")
        
        return f1

    except subprocess.TimeoutExpired:
        print(f"  ERROR: Trial {trial.number} timed out")
        return 0.0
    except Exception as e:
        print(f"  ERROR: Trial {trial.number} failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def run_bond_validation(params):
    """Esegue validation con i parametri migliori"""
    print("\n" + "="*70)
    print("RUNNING VALIDATION WITH BEST PARAMETERS")
    print("="*70)
    
    # Setup validation
    success, needs_cleanup, paths_to_cleanup = setup_validation_structure()
    
    if not success:
        print("Failed to setup validation")
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
            print(f"ERROR: Ground truth not found")
            return 0.0
        
        print(f"✓ Ground truth: {ground_truth_file.name}")
        
        # Environment
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env["SKIP_PREPROCESS"] = "1"
        
        # Output directory
        output_dir = RESULTS_DIR / "validation"
        output_dir.mkdir(exist_ok=True)
        
        # ============================================================
        # COMANDO VALIDATION (CON CITAZIONI)
        # ============================================================
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
            '--epochs', str(params.get('epochs', 50)),
            # ⭐ CITAZIONI ⭐
            #'--use_citations', str(params.get('use_citations', True)),
            '--cite_out_weight', str(params.get('cite_out_weight', 1.0)),
            '--cite_in_weight', str(params.get('cite_in_weight', 1.0)),
            '--cite_out_th', str(params.get('cite_out_th', 0.0)),
            '--cite_in_th', str(params.get('cite_in_th', 0.0))
        ]
        
        print(f"\nRunning BOND validation...")
        print(f"  Standard params: db_eps={params['db_eps']}, lr={params['lr']:.2e}")
        print(f"  ⭐ Citations: out_w={params.get('cite_out_weight', 1.0):.2f}, in_w={params.get('cite_in_weight', 1.0):.2f}")
        
        # Esegui
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
            print(f"WARNING: BOND exited with code {result.returncode}")
            with open(output_dir / "stderr.txt", 'w', encoding='utf-8') as f:
                f.write(result.stderr)
        
        # Predizioni
        predictions_file = BASE_PATH / "out" / "res.json"
        
        if not predictions_file.exists():
            print(f"ERROR: Predictions not found")
            return 0.0
        
        # Copia e valuta
        output_predictions = output_dir / "predictions.json"
        shutil.copy2(predictions_file, output_predictions)
        
        print(f"\nEvaluating validation set...")
        f1 = evaluate_predictions(output_predictions, ground_truth_file)
        
        # Salva risultati
        result_data = {
            "split": "valid",
            "f1_score": f1,
            "params": params,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(output_dir / "results.json", 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"VALIDATION F1: {f1:.4f}")
        print(f"{'='*70}")
        
        return f1
        
    finally:
        if needs_cleanup:
            cleanup_validation_structure(paths_to_cleanup)


def optimize_hyperparameters():
    """Ottimizzazione iperparametri principale"""
    print("\n" + "="*70)
    print("BOND HYPERPARAMETER OPTIMIZATION")
    print("CON OTTIMIZZAZIONE CITAZIONI")
    print("="*70)
    
    # Verifica preprocessing
    #if not check_preprocessing():
     #   print("\nERROR: Preprocessing not complete")
      #  sys.exit(1)
    print("✓ Preprocessing check disabled")

    
    print("✓ Preprocessing data found")
    print(f"✓ Train directory: {DATA_PATH / 'src' / 'train'}")
    
    # Numero trial
    try:
        n_trials = int(input("\nNumber of trials (default 10): ") or "10")
    except ValueError:
        n_trials = 10
    
    print(f"\nStarting optimization:")
    print(f"  - Trials: {n_trials}")
    print(f"  - Epochs per trial: 50 (fixed)")
    print(f"  - Estimated time: {n_trials * 0.5:.1f} hours")
    print(f"  - ⭐ INCLUDE CITATION OPTIMIZATION")
    print(f"  - Results: {RESULTS_DIR}")
    
    input("\nPress ENTER to start...")
    
    # Crea study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name='bond_hyperopt_with_citations'
    )
    
    # Progress tracker
    progress = ProgressTracker(n_trials)
    
    # Ottimizza
    study.optimize(objective, n_trials=n_trials, callbacks=[progress])
    
    # Risultati
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    
    best_trial = study.best_trial
    print(f"\nBest trial: #{best_trial.number}")
    print(f"Best F1 (train): {best_trial.value:.4f}")
    
    print(f"\nBest parameters:")
    for key, value in best_trial.params.items():
        if 'cite' in key:
            print(f"  ⭐ {key}: {value}")
        else:
            print(f"  {key}: {value}")
    
    # Salva best params
    best_params_file = RESULTS_DIR / "best_parameters.json"
    with open(best_params_file, 'w') as f:
        json.dump({
            'trial_number': best_trial.number,
            'f1_train': best_trial.value,
            'params': best_trial.params,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n✓ Best parameters saved: {best_params_file}")
    
    # Validation
    print(f"\n{'='*70}")
    user_input = input("\nRun validation with best parameters? (y/n): ")
    
    if user_input.lower() == 'y':
        # Prepara params per validation
        val_params = best_trial.params.copy()
        val_params['epochs'] = 50
        
        val_f1 = run_bond_validation(val_params)
        
        # Salva risultati validation
        final_results = {
            'best_trial': best_trial.number,
            'f1_train': best_trial.value,
            'f1_valid': val_f1,
            'params': val_params,
            'timestamp': datetime.now().isoformat()
        }
        
        final_file = RESULTS_DIR / "final_results.json"
        with open(final_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n✓ Final results saved: {final_file}")
        
        print(f"\n{'='*70}")
        print("FINAL RESULTS")
        print(f"{'='*70}")
        print(f"Train F1: {best_trial.value:.4f}")
        print(f"Valid F1: {val_f1:.4f}")
        print(f"Overfitting: {(best_trial.value - val_f1):.4f}")
    
    print(f"\n{'='*70}")
    print(f"All results saved in: {RESULTS_DIR}")
    print(f"{'='*70}\n")


def main():
    """Entry point"""
    try:
        optimize_hyperparameters()
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
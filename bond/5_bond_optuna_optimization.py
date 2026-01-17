"""
Script di ottimizzazione iperparametri per BOND con PipelineA
"""
import optuna
import json
import subprocess
import os
import sys
import time
import shutil
import numpy as np
from datetime import datetime, timedelta
from optuna.samplers import TPESampler
from pathlib import Path

# ======================== CONFIGURAZIONE ========================
BASE_PATH = Path(__file__).parent
PIPELINE_SCRIPT = BASE_PATH / "PipelineA.py"  
DATA_PATH = BASE_PATH / "dataset" / "data"

RESULTS_DIR = BASE_PATH / "hyperopt_multimetric_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==================== CONFIGURAZIONE EARLY STOPPING ====================
EARLY_STOPPING_CONFIG = {
    'patience': 15,
    'min_delta': 0.001,
    'min_trials': 20,
    'convergence_window': 10,
    'convergence_threshold': 0.0005
}


class EarlyStoppingCallback:
    """Callback per early stopping intelligente"""
    
    def __init__(self, patience=15, min_delta=0.001, min_trials=20, 
                 convergence_window=10, convergence_threshold=0.0005):
        self.patience = patience
        self.min_delta = min_delta
        self.min_trials = min_trials
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold
        
        self.best_value = -float('inf')
        self.best_trial = None
        self.trials_without_improvement = 0
        self.trial_values = []
        
    def __call__(self, study, trial):
        if len(study.trials) < self.min_trials:
            return
        
        current_value = trial.value
        if current_value is None:
            return
        
        self.trial_values.append(current_value)
        
        improvement = current_value - self.best_value
        
        if improvement > self.min_delta:
            self.best_value = current_value
            self.best_trial = trial.number
            self.trials_without_improvement = 0
            print(f"  ✓ New best score: {current_value:.4f} (+{improvement:.4f})")
        else:
            self.trials_without_improvement += 1
            print(f"  ○ No improvement for {self.trials_without_improvement} trials")
        
        if len(self.trial_values) >= self.convergence_window:
            recent_values = self.trial_values[-self.convergence_window:]
            variance = np.var(recent_values)
            
            if variance < self.convergence_threshold:
                print(f"\n  ⚠ Convergence detected! Stopping at trial {trial.number}")
                study.stop()
                return
        
        if self.trials_without_improvement >= self.patience:
            print(f"\n  ⚠ Early stopping! No improvement for {self.patience} trials")
            study.stop()


class ProgressTracker:
    """Tracker per mostrare il progresso"""
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
        print(f"Best Composite Score: {self.best_value:.4f}")
        print(f"ETA: {eta}")
        print(f"{'='*70}\n")


def objective(trial):
    """
    Funzione obiettivo per Optuna usando PipelineA
    USA I NOMI CORRETTI DA params.py!
    """
    # ========== PARAMETRI CON NOMI CORRETTI ==========
    params = {
        # Clustering
        'db_eps': trial.suggest_float('db_eps', 0.05, 0.3, step=0.01),
        'db_min': trial.suggest_int('db_min', 2, 10),
        'cluster_w': trial.suggest_float('cluster_w', 0.1, 0.9, step=0.1),
        
        # Training
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'l2_coef': trial.suggest_float('l2_coef', 1e-6, 1e-3, log=True),
        'compress_ratio': trial.suggest_float('compress_ratio', 0.5, 1.0, step=0.1),
        'epochs': 50,
        
        # Hidden dimensions (come LISTA - params.py li accetta così)
        'hidden_dim': [
            trial.suggest_categorical('hidden_dim_0', [128, 256, 512]),
            trial.suggest_categorical('hidden_dim_1', [128, 256, 512])
        ],
        
        # Co-author thresholds (come LISTA)
        'coa_th': trial.suggest_int('coa_th', 0, 2),
        
        # Co-org thresholds (come LISTA)
        'coo_th': trial.suggest_float('coo_th', 0.3, 0.9, step=0.05),
        
        # Co-venue thresholds (come LISTA) 
        'cov_th': trial.suggest_float('cov_th', 0.5, 3.0, step=0.5),
        
        # Citation thresholds (NOMI CORRETTI: coc_th, coi_th)
        'coc_th': trial.suggest_float('coc_th', 0.0, 0.5, step=0.05),
        'coi_th': trial.suggest_float('coi_th', 0.0, 0.5, step=0.05),
        
        # Citation weights (NOMI CORRETTI: già giusti)
        'cite_out_weight': trial.suggest_float('cite_out_weight', 0.5, 2.0, step=0.1),
        'cite_in_weight': trial.suggest_float('cite_in_weight', 0.3, 1.5, step=0.1),
    }
    # ================================================

    # Environment variables
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['HYPEROPT_MODE'] = '1'

    # Directory trial
    trial_dir = RESULTS_DIR / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # ========== COSTRUISCI COMANDO CON NOMI CORRETTI ==========
    cmd = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        '--mode', 'train',
        '--db_eps', str(params['db_eps']),
        '--db_min', str(params['db_min']),
        '--cluster_w', str(params['cluster_w']),
        '--lr', str(params['lr']),
        '--l2_coef', str(params['l2_coef']),
        '--compress_ratio', str(params['compress_ratio']),
        '--epochs', str(params['epochs']),
        
        # Hidden dim come lista separata da spazi
        '--hidden_dim', str(params['hidden_dim'][0]), str(params['hidden_dim'][1]),
        
        # Thresholds singoli (non liste multiple)
        '--coa_th', str(params['coa_th']),
        '--coo_th', str(params['coo_th']),
        '--cov_th', str(params['cov_th']),
        
        # Citation thresholds (NOMI CORRETTI!)
        '--coc_th', str(params['coc_th']),
        '--coi_th', str(params['coi_th']),
        
        # Citation weights
        '--cite_out_weight', str(params['cite_out_weight']),
        '--cite_in_weight', str(params['cite_in_weight']),
    ]
    # =========================================================

    print(f"\nTrial {trial.number} - Starting PipelineA...")
    print(f"  Clustering: eps={params['db_eps']}, min={params['db_min']}")
    print(f"  Citations: out_w={params['cite_out_weight']:.1f}, in_w={params['cite_in_weight']:.1f}")
    
    try:
        result = subprocess.run(
            cmd, 
            env=env,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=7200  # 2 ore
        )
        
        # Salva output
        with open(trial_dir / "stdout.txt", 'w', encoding='utf-8') as f:
            f.write(result.stdout)
        with open(trial_dir / "stderr.txt", 'w', encoding='utf-8') as f:
            f.write(result.stderr)
        
        if result.returncode != 0:
            print(f"  WARNING: PipelineA exited with code {result.returncode}")
            print(f"  Check {trial_dir / 'stderr.txt'} for details")
            return 0.0
        
        # Leggi risultati
        results_file = BASE_PATH / "evaluation_results" / "evaluation_results.json"
        
        if not results_file.exists():
            print(f"  ERROR: Results file not found: {results_file}")
            return 0.0
        
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Copia risultati
        shutil.copy2(results_file, trial_dir / "results.json")
        
        # Estrai composite score
        composite_score = results.get('composite_score', 0.0)
        
        # Estrai altre metriche per logging
        pairwise_f1 = results.get('pairwise', {}).get('f1', 0.0)
        k_metric = results.get('k_metric', {}).get('k', 0.0)
        cluster_f1 = results.get('cluster', {}).get('f1', 0.0)
        
        print(f"\n  Trial {trial.number} completed:")
        print(f"    Composite Score: {composite_score:.4f}")
        print(f"    Pairwise F1:     {pairwise_f1:.4f}")
        print(f"    K-metric:        {k_metric:.4f}")
        print(f"    Cluster F1:      {cluster_f1:.4f}")
        
        # Salva info trial
        trial_data = {
            "trial": trial.number,
            "composite_score": composite_score,
            "metrics": {
                "pairwise_f1": pairwise_f1,
                "k_metric": k_metric,
                "cluster_f1": cluster_f1
            },
            "params": params,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(trial_dir / "trial_info.json", 'w', encoding='utf-8') as f:
            json.dump(trial_data, f, indent=2)
        
        return composite_score
    
    except subprocess.TimeoutExpired:
        print(f"  ERROR: Trial {trial.number} timed out (>2 hours)")
        return 0.0
    except Exception as e:
        print(f"  ERROR: Trial {trial.number} failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def optimize_hyperparameters():
    """Ottimizzazione iperparametri con PipelineA"""
    print("\n" + "="*70)
    print("BOND HYPERPARAMETER OPTIMIZATION")
    print("Using PipelineA + Multi-Metric Evaluation")
    print("="*70)
    
    print("\n[CONFIGURATION]")
    print(f"  Pipeline: {PIPELINE_SCRIPT}")
    print(f"  Optimizing: Composite score")
    
    print("\n" + "="*70)
    print("PREPROCESSING CHECK")
    print("="*70)
    
    # Check se preprocessing è già disponibile
    preprocessing_done = (
        (DATA_PATH / 'graph' / 'train').exists() and
        any((DATA_PATH / 'graph' / 'train').iterdir())
    )
    
    if preprocessing_done:
        print("\n✅ Preprocessing appears to be done!")
        print("   PipelineA will skip preprocessing (HYPEROPT_MODE=1)")
    else:
        print("\n⚠️  Preprocessing not found!")
        do_preprocessing = input("\nRun preprocessing now? (y/N): ").strip().lower()
        
        if do_preprocessing == 'y':
            print("\nRunning preprocessing...")
            cmd = [sys.executable, str(PIPELINE_SCRIPT), '--mode', 'train']
            
            # Rimuovi HYPEROPT_MODE per fare preprocessing
            env = os.environ.copy()
            if 'HYPEROPT_MODE' in env:
                del env['HYPEROPT_MODE']
            
            try:
                result = subprocess.run(cmd, env=env, timeout=3600)
                if result.returncode != 0:
                    print("\n❌ Preprocessing failed!")
                    return
                print("\n✅ Preprocessing completed!")
            except Exception as e:
                print(f"\n❌ Preprocessing error: {e}")
                return
        else:
            print("\n❌ Cannot continue without preprocessing!")
            return
    
    try:
        n_trials = int(input("\nMax number of trials (default 100): ") or "100")
    except ValueError:
        n_trials = 100
    
    print(f"\nStarting optimization:")
    print(f"  - Max trials: {n_trials}")
    print(f"  - Early stopping enabled")
    print(f"  - Results: {RESULTS_DIR}")
    
    input("\nPress ENTER to start...")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name='bond_pipelineA_optimization'
    )
    
    progress = ProgressTracker(n_trials)
    early_stopping = EarlyStoppingCallback(**EARLY_STOPPING_CONFIG)
    
    # Optimize
    study.optimize(
        objective, 
        n_trials=n_trials, 
        callbacks=[progress, early_stopping],
        show_progress_bar=True
    )
    
    # Results
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    
    best_trial = study.best_trial
    print(f"\nBest trial: #{best_trial.number}")
    print(f"Best Composite Score: {best_trial.value:.4f}")
    
    print(f"\n[Best parameters]")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Salva best params
    best_params_file = RESULTS_DIR / "best_parameters.json"
    with open(best_params_file, 'w', encoding='utf-8') as f:
        json.dump({
            'best_trial': best_trial.number,
            'composite_score': best_trial.value,
            'params': best_trial.params,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n✓ Best parameters saved: {best_params_file}")
    print(f"\nAll results in: {RESULTS_DIR}")


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
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

CITATION_CONFIG = {
    'use_citations': True,
    'cite_out_weight': (0.5, 2.0),
    'cite_in_weight': (0.5, 2.0),
    'cite_out_th': (0.0, 0.5),
    'cite_in_th': (0.0, 0.5)
}

# ==================== CONFIGURAZIONE EARLY STOPPING ====================
EARLY_STOPPING_CONFIG = {
    'patience': 15,
    'min_delta': 0.001,
    'min_trials': 20,
    'convergence_window': 10,
    'convergence_threshold': 0.0005
}

# ==================== PESI METRICHE ====================
METRIC_WEIGHTS = {
    'pairwise_f1': 0.35,
    'k_metric': 0.30,
    'cluster_f1': 0.15,
    'splitting_error': 0.10,
    'lumping_error': 0.10
}

assert abs(sum(METRIC_WEIGHTS.values()) - 1.0) < 1e-6, "I pesi devono sommare a 1!"


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
        
        # Check miglioramento
        improvement = current_value - self.best_value
        
        if improvement > self.min_delta:
            self.best_value = current_value
            self.best_trial = trial.number
            self.trials_without_improvement = 0
            print(f"  ✓ New best score: {current_value:.4f} (+{improvement:.4f})")
        else:
            self.trials_without_improvement += 1
            print(f"  ○ No improvement for {self.trials_without_improvement} trials")
        
        # Check convergenza
        if len(self.trial_values) >= self.convergence_window:
            recent_values = self.trial_values[-self.convergence_window:]
            variance = np.var(recent_values)
            
            if variance < self.convergence_threshold:
                print(f"\n  ⚠ Convergence detected! Stopping at trial {trial.number}")
                study.stop()
                return
        
        # Check patience
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
    """
    params = {
        'db_eps': trial.suggest_float('db_eps', 0.05, 0.3, step=0.01),
        'db_min': trial.suggest_int('db_min', 2, 10),
        'cluster_w': trial.suggest_float('cluster_w', 0.1, 0.9, step=0.1),
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'l2_coef': trial.suggest_float('l2_coef', 1e-6, 1e-3, log=True),
        'hidden_dim_0': trial.suggest_categorical('hidden_dim_0', [128, 256, 512]),
        'hidden_dim_1': trial.suggest_categorical('hidden_dim_1', [128, 256, 512]),
        'compress_ratio': trial.suggest_float('compress_ratio', 0.5, 1.0, step=0.1),
        'th_a_0': trial.suggest_float('th_a_0', 0.0, 0.5),
        'th_a_1': trial.suggest_float('th_a_1', 0.5, 1.5),
        'th_o_0': trial.suggest_float('th_o_0', 0.3, 0.8),
        'th_o_1': trial.suggest_float('th_o_1', 0.3, 0.8),
        'th_v_0': trial.suggest_float('th_v_0', 0.5, 2.0),
        'th_v_1': trial.suggest_float('th_v_1', 1.0, 3.0),
        'cite_out_weight': trial.suggest_float('cite_out_weight', *CITATION_CONFIG['cite_out_weight'], step=0.1),
        'cite_in_weight': trial.suggest_float('cite_in_weight', *CITATION_CONFIG['cite_in_weight'], step=0.1),
        'cite_out_th': trial.suggest_float('cite_out_th', *CITATION_CONFIG['cite_out_th'], step=0.05),
        'cite_in_th': trial.suggest_float('cite_in_th', *CITATION_CONFIG['cite_in_th'], step=0.05),
        'epochs': 50
    }

    # ========== ENVIRONMENT VARIABLES ==========
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['HYPEROPT_MODE'] = '1'
    # ===========================================

    # Directory trial
    trial_dir = RESULTS_DIR / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # ========== COMANDO PIPELINEA ==========
    cmd = [
        'python', str(PIPELINE_SCRIPT),
        '--mode', 'train',
        '--db_eps', str(params['db_eps']),
        '--db_min', str(params['db_min']),
        '--cluster_w', str(params['cluster_w']),
        '--lr', str(params['lr']),
        '--l2_coef', str(params['l2_coef']),
        '--hidden_dim_0', str(params['hidden_dim_0']),
        '--hidden_dim_1', str(params['hidden_dim_1']),
        '--compress_ratio', str(params['compress_ratio']),
        '--th_a_0', str(params['th_a_0']),
        '--th_a_1', str(params['th_a_1']),
        '--th_o_0', str(params['th_o_0']),
        '--th_o_1', str(params['th_o_1']),
        '--th_v_0', str(params['th_v_0']),
        '--th_v_1', str(params['th_v_1']),
        '--cite_out_weight', str(params['cite_out_weight']),
        '--cite_in_weight', str(params['cite_in_weight']),
        '--cite_out_th', str(params['cite_out_th']),
        '--cite_in_th', str(params['cite_in_th']),
        '--epochs', str(params['epochs'])
    ]
    # ========================================

    print(f"\nTrial {trial.number} - Starting PipelineA...")
    print(f"  Mode: train (optimization on training set)")  
    print(f"  Clustering: eps={params['db_eps']}, min={params['db_min']}")
    
    try:
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
            print(f"  WARNING: PipelineA exited with code {result.returncode}")
            with open(trial_dir / "error.txt", 'w', encoding='utf-8') as f:
                f.write(result.stderr)
            return 0.0
        
        results_file = BASE_PATH / "evaluation_results" / "evaluation_results.json"
        
        if not results_file.exists():
            print(f"  ERROR: Results file not found: {results_file}")
            return 0.0
        
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        shutil.copy2(results_file, trial_dir / "results.json")
        
        # ========== ESTRAI METRICHE CON GESTIONE ERRORI ==========
        try:
            if not results:
                print(f"  ERROR: Results file is empty!")
                return 0.0
            
            composite_score = results.get('composite_score', 0.0)
            
            if 'pairwise' not in results:
                print(f"  ⚠️  WARNING: 'pairwise' not in results!")
                print(f"  Available keys: {list(results.keys())}")
                
                if 'summary' in results:
                    print(f"  → Trying to extract from 'summary'...")
                    results = results['summary']
                    composite_score = results.get('composite_score', 0.0)
                else:
                    print(f"  → Results appear to be empty/invalid")
                    return 0.0
            
            pairwise_f1 = results.get('pairwise', {}).get('f1', 0.0)
            k_metric = results.get('k_metric', {}).get('k', 0.0)
            cluster_f1 = results.get('cluster', {}).get('f1', 0.0)
            
            print(f"\n  Trial {trial.number} completed:")
            print(f"    Composite Score: {composite_score:.4f}")
            print(f"    Pairwise F1:     {pairwise_f1:.4f}")
            print(f"    K-metric:        {k_metric:.4f}")
            print(f"    Cluster F1:      {cluster_f1:.4f}")
            
            if composite_score == 0.0 and pairwise_f1 == 0.0 and k_metric == 0.0:
                print(f"\n  ⚠️  WARNING: All metrics are 0.0 - evaluation likely failed!")
                print(f"  Check: evaluation_results/evaluation_results.json")
            
        except KeyError as e:
            print(f"  ERROR: Missing key in results: {e}")
            print(f"  Available keys: {list(results.keys()) if results else 'None'}")
            return 0.0
        except Exception as e:
            print(f"  ERROR: Failed to extract metrics: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
        # =========================================================
        
        trial_data = {
            "trial": trial.number,
            "composite_score": composite_score,
            "metrics": results,
            "params": params,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(trial_dir / "trial_info.json", 'w', encoding='utf-8') as f:
            json.dump(trial_data, f, indent=2)
        
        return composite_score
    
    except subprocess.TimeoutExpired:
        print(f"  ERROR: Trial {trial.number} timed out")
        return 0.0
    except Exception as e:
        print(f"  ERROR: Trial {trial.number} failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def run_preprocessing_once():
    """
    Esegue preprocessing UNA VOLTA all'inizio
    """
    print("\n" + "="*70)
    print("PREPROCESSING STEP (ONE TIME ONLY)")
    print("="*70)
    print("\nRunning full pipeline once to generate preprocessing...")
    
    cmd = [
        'python', str(PIPELINE_SCRIPT),
        '--mode', 'train'
    ]
    
    env = os.environ.copy()
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            timeout=3600  
        )
        
        if result.returncode == 0:
            print("\n✅ Preprocessing completed successfully!")
            return True
        else:
            print("\n❌ Preprocessing failed!")
            return False
            
    except Exception as e:
        print(f"\n❌ Preprocessing error: {e}")
        return False


def optimize_hyperparameters():
    """Ottimizzazione iperparametri con PipelineA"""
    print("\n" + "="*70)
    print("BOND HYPERPARAMETER OPTIMIZATION")
    print("Using PipelineA + Multi-Metric Evaluation")
    print("="*70)
    
    print("\n[CONFIGURATION]")
    print(f"  Pipeline: {PIPELINE_SCRIPT}")
    print(f"  Optimizing: Composite score")
    print(f"  Metric weights:")
    for metric, weight in METRIC_WEIGHTS.items():
        print(f"    {metric:20s}: {weight:.2f}")
    
    print("\n" + "="*70)
    print("PREPROCESSING CHECK")
    print("="*70)
    do_preprocessing = input("\nRun preprocessing now? (y/N): ").strip().lower()
    
    if do_preprocessing == 'y':
        if not run_preprocessing_once():
            print("\n❌ Cannot continue without preprocessing!")
            return
    else:
        print("\n⚠️  Assuming preprocessing is already done...")
        print("   PipelineA will skip preprocessing steps (HYPEROPT_MODE=1)")
    
    try:
        n_trials = int(input("\nMax number of trials (default 100): ") or "100")
    except ValueError:
        n_trials = 100
    
    print(f"\nStarting optimization:")
    print(f"  - Max trials: {n_trials}")
    print(f"  - Early stopping enabled")
    print(f"  - Results: {RESULTS_DIR}")
    
    input("\nPress ENTER to start...")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name='bond_pipelineA_optimization'
    )
    
    progress = ProgressTracker(n_trials)
    early_stopping = EarlyStoppingCallback(**EARLY_STOPPING_CONFIG)
    
    study.optimize(
        objective, 
        n_trials=n_trials, 
        callbacks=[progress, early_stopping],
        show_progress_bar=True
    )
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    
    best_trial = study.best_trial
    print(f"\nBest trial: #{best_trial.number}")
    print(f"Best Composite Score: {best_trial.value:.4f}")
    
    print(f"\n[Best parameters]")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
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
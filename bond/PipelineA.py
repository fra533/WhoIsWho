"""
═══════════════════════════════════════════════════════════════════════
    BOND COMPLETE PIPELINE
═══════════════════════════════════════════════════════════════════════

Pipeline completa da preprocessing a confidence scoring.

Compatibile con:
- Single training run (full pipeline)
- Hyperparameter optimization (solo training+eval)
- Train e validation modes

Steps:
1. Preprocessing (part 1) - dump name pubs & features
2. Word2Vec training
3. Preprocessing (part 2) - build graphs
4. Model training (BOND)
5. Evaluation (multi-metric)
6. Confidence scoring & filtering

USAGE:
------
# Interactive mode:
python PipelineA.py --mode train

# Skip preprocessing:
SKIP_PREPROCESS=1 SKIP_W2V=1 python PipelineA.py --mode valid

# Hyperopt mode (chiamato da script Optuna):
HYPEROPT_MODE=1 python PipelineA.py --mode train [hyperparams...]
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from training.autotrain_bond import BONDTrainer
from training.autotrain_bond_ensemble import ESBTrainer
from dataset.preprocess_SND import dump_name_pubs, dump_features_relations_to_file, build_graph
from params import set_params

args = set_params()


class BondPipeline:
    """
    Pipeline orchestrator completo per BOND
    
    Modalità:
    - FULL: Tutti i 6 step (preprocessing -> training -> eval -> confidence)
    - HYPEROPT: Solo training + eval (preprocessing già fatto)
    - VALIDATION: Training + eval su validation set
    """
    
    def __init__(self):
        self.base_path = Path(args.save_path)
        self.mode = args.mode
        self.results = {}
        
        self.eval_mode = os.getenv("EVAL_MODE", self.mode)
    
        if self.eval_mode != self.mode:
            print(f"⚠️  Split mode: Training on '{self.mode}', Evaluating on '{self.eval_mode}'")


        env_skip_preprocess = os.getenv("SKIP_PREPROCESS") == "1"
        env_skip_w2v = os.getenv("SKIP_W2V") == "1"
        env_hyperopt = os.getenv("HYPEROPT_MODE") == "1"
        
        self.config = {
            'skip_preprocessing': env_skip_preprocess,
            'skip_w2v': env_skip_w2v,
            'skip_evaluation': os.getenv("SKIP_EVAL") == "1",
            'skip_confidence': os.getenv("SKIP_CONFIDENCE") == "1",
            'hyperopt_mode': env_hyperopt,
            'model_type': 'bond'
        }
        # ==================================================
        
        if self.config['hyperopt_mode']:
            self.config['skip_preprocessing'] = True
            self.config['skip_w2v'] = True
            self.config['skip_confidence'] = True
        
        print("\n" + "="*70)
        print(" BOND COMPLETE PIPELINE")
        print("="*70)
        print(f"Mode: {self.mode}")
        print(f"Base path: {self.base_path}")
        
        if self.config['hyperopt_mode']:
            print("⚡ HYPEROPT MODE - Fast training only")
        
        # ========== MOSTRA SE CI SONO ENV VARIABLES ATTIVE ==========
        if env_skip_preprocess or env_skip_w2v:
            print("\n⚠️  Environment variables detected:")
            if env_skip_preprocess:
                print("   SKIP_PREPROCESS=1")
            if env_skip_w2v:
                print("   SKIP_W2V=1")
            print("   (These will be used unless you choose to override)")
        # ===========================================================
        
        print(f"\nConfiguration:")
        for key, value in self.config.items():
            if value:
                print(f"  {key}: {value}")
    
    def check_preprocessing_availability(self):
        """
        Controlla se il preprocessing è già disponibile e quale
        
        Returns:
            dict con info su cosa è già disponibile
        """
        availability = {
            'part1_done': False,
            'part2_done': False,
            'w2v_done': False,
            'embeddings_done': False,
            'can_skip_preprocess': False,
            'can_skip_w2v': False
        }
        
        availability['part1_done'] = self._check_preprocessing_part1_done()
        
        availability['part2_done'] = self._check_preprocessing_part2_done()
        
        w2v_model_path = self.base_path / 'w2v_model' / 'w2v_256.model'
        availability['w2v_done'] = w2v_model_path.exists()
        
        paper_emb_dir = self.base_path / 'paper_emb' / self.mode
        availability['embeddings_done'] = (
            paper_emb_dir.exists() and any(paper_emb_dir.iterdir())
        )
        
        availability['can_skip_preprocess'] = (
            availability['part1_done'] and 
            availability['part2_done'] and
            availability['embeddings_done']
        )
        
        availability['can_skip_w2v'] = (
            availability['w2v_done'] and 
            availability['embeddings_done']
        )
        
        return availability
    
    def interactive_configuration(self):
        """
        Menu interattivo per configurare la pipeline
        
        Solo se NON siamo in hyperopt mode
        """
        if self.config['hyperopt_mode']:
            return  #
        
        avail = self.check_preprocessing_availability()
        
        print("\n" + "="*70)
        print(" PREPROCESSING STATUS")
        print("="*70)
        print(f"Mode: {self.mode}")
        print(f"\nAvailable preprocessing:")
        print(f"  ✓ Part 1 (names & features):  {'✅ Done' if avail['part1_done'] else '❌ Not found'}")
        print(f"  ✓ Part 2 (graphs):            {'✅ Done' if avail['part2_done'] else '❌ Not found'}")
        print(f"  ✓ Word2Vec model:             {'✅ Done' if avail['w2v_done'] else '❌ Not found'}")
        print(f"  ✓ Paper embeddings ({self.mode}):    {'✅ Done' if avail['embeddings_done'] else '❌ Not found'}")
        
        fully_available = avail['can_skip_preprocess'] and avail['can_skip_w2v']
        
        partially_available = (
            (avail['part1_done'] or avail['part2_done'] or avail['w2v_done'] or avail['embeddings_done'])
            and not fully_available
        )
        
        if fully_available:
            print("\n" + "="*70)
            print("🎯 PIPELINE CONFIGURATION")
            print("="*70)
            print("\n✅ All preprocessing is available!")
            print("\nOptions:")
            print("  1) Use existing preprocessing (FAST - recommended)")
            print("  2) Redo all preprocessing (SLOW)")
            print("  3) Custom (choose what to skip)")
            
            choice = input("\nYour choice [1/2/3, default=1]: ").strip() or "1"
            
            if choice == "1":
                self.config['skip_preprocessing'] = True
                self.config['skip_w2v'] = True
                print("\n✅ Will use existing preprocessing")
                
            elif choice == "2":
                self.config['skip_preprocessing'] = False
                self.config['skip_w2v'] = False
                print("\n⚙️  Will redo all preprocessing")
                
                confirm = input("\n⚠️  This will take time. Continue? [y/N]: ").strip().lower()
                if confirm != 'y':
                    print("\n❌ Aborted by user")
                    sys.exit(0)
                    
            elif choice == "3":
                self._custom_configuration(avail)
            
            else:
                print("\n❌ Invalid choice, using default (use existing)")
                self.config['skip_preprocessing'] = True
                self.config['skip_w2v'] = True
        
        elif partially_available:
            print("\n" + "="*70)
            print("⚠️  PARTIAL PREPROCESSING AVAILABLE")
            print("="*70)
            print("\n⚠️  Some preprocessing is missing!")
            print("    Missing components will be regenerated.")
            
            self._custom_configuration(avail)
        
        else:
            print("\n" + "="*70)
            print("ℹ️  NO PREPROCESSING FOUND")
            print("="*70)
            print("\nPreprocessing will be done from scratch.")
            print("This may take some time...")
            
            self.config['skip_preprocessing'] = False
            self.config['skip_w2v'] = False
            
            confirm = input("\nContinue? [Y/n]: ").strip().lower()
            if confirm == 'n':
                print("\n❌ Aborted by user")
                sys.exit(0)
        
        print("\n" + "="*70)
        print("📊 FINAL CONFIGURATION")
        print("="*70)
        print(f"Mode:                 {self.mode}")
        print(f"Skip preprocessing:   {self.config['skip_preprocessing']}")
        print(f"Skip Word2Vec:        {self.config['skip_w2v']}")
        print(f"Skip evaluation:      {self.config['skip_evaluation']}")
        print(f"Skip confidence:      {self.config['skip_confidence']}")
        
        # ========== DEBUG: Verifica che i flag siano corretti ==========
        print("\n[DEBUG] Internal flags:")
        print(f"  skip_preprocessing: {self.config['skip_preprocessing']}")
        print(f"  skip_w2v: {self.config['skip_w2v']}")
        # ==============================================================
        
        input("\nPress ENTER to start pipeline...")
        
    def _custom_configuration(self, avail):
        """Helper per configurazione custom"""
        print("\n" + "="*70)
        print("⚙️  CUSTOM CONFIGURATION")
        print("="*70)
        
        print("\nWhat needs to be done:")
        if not avail['part1_done']:
            print("  ❌ Preprocessing Part 1 (names & features) - REQUIRED")
        if not avail['part2_done']:
            print("  ❌ Preprocessing Part 2 (graphs) - REQUIRED")
        if not avail['embeddings_done']:
            print(f"  ❌ Paper embeddings for mode '{self.mode}' - REQUIRED")
        if not avail['w2v_done']:
            print("  ❌ Word2Vec model - REQUIRED")
        
        print("\nOptions:")
        print("  1) Do what's needed (recommended)")
        print("  2) Redo everything from scratch")
        
        choice = input("\nYour choice [1/2, default=1]: ").strip() or "1"
        
        if choice == "1":
            self.config['skip_preprocessing'] = (
                avail['part1_done'] and avail['part2_done']
            )
            self.config['skip_w2v'] = (
                avail['w2v_done'] and avail['embeddings_done']
            )
            print("\n✅ Will do only what's missing")
        else:
            self.config['skip_preprocessing'] = False
            self.config['skip_w2v'] = False
            print("\n⚙️  Will redo everything")
        
        skip_conf = input("\nSkip confidence scoring? [y/N]: ").strip().lower()
        self.config['skip_confidence'] = (skip_conf == 'y')

    def run_full_pipeline(self):
        """
        Esegue l'intera pipeline dall'inizio alla fine
        """
        if not self.config['hyperopt_mode']:
            print("\n" + "="*70)
            print("📋 PIPELINE STEPS")
            print("="*70)
            print("""
            Step 1: Preprocessing (Part 1) - Names & Features
            Step 2: Word2Vec Training & Embeddings
            Step 3: Preprocessing (Part 2) - Graph Building
            Step 4: Model Training
            Step 5: Multi-Metric Evaluation
            Step 6: Confidence Scoring & Filtering
            """)
            
            # ========== DEBUG ==========
            print(f"\n[DEBUG] Will skip preprocessing: {self.config['skip_preprocessing']}")
            print(f"[DEBUG] Will skip w2v: {self.config['skip_w2v']}")
            # ===========================
        
        # ========== DETERMINA SE FORZARE RERUN ==========
        # Se skip=False significa che vogliamo eseguire, quindi force=True
        force_preprocessing = not self.config['skip_preprocessing']
        force_w2v = not self.config['skip_w2v']
        
        if not self.config['hyperopt_mode']:
            print(f"\n[DEBUG] Force preprocessing: {force_preprocessing}")
            print(f"[DEBUG] Force w2v: {force_w2v}")
        # ================================================
        
        if not self.config['skip_preprocessing']:
            success = self.step1_preprocessing_part1(force_rerun=force_preprocessing)
            if not success:
                return False
        else:
            if not self.config['hyperopt_mode']:
                print("\n⏭️  Step 1: SKIPPED (using existing)")
        
        if not self.config['skip_w2v']:
            success = self.step2_word2vec(force_rerun=force_w2v)
            if not success:
                return False
        else:
            if not self.config['hyperopt_mode']:
                print("\n⏭️  Step 2: SKIPPED (using existing model & embeddings)")
        
            
        if not self.config['skip_preprocessing']:
            success = self.step3_preprocessing_part2(force_rerun=force_preprocessing)
            if not success:
                return False
        else:
            if not self.config['hyperopt_mode']:
                print("\n⏭️  Step 3: SKIPPED (using existing graphs)")
        
        success = self.step4_training()
        if not success:
            return False
        
        if not self.config['skip_evaluation']:
            success = self.step5_evaluation()
            if not success:
                print("⚠️  Evaluation failed, but continuing...")
        else:
            if not self.config['hyperopt_mode']:
                print("\n⏭️  Step 5: SKIPPED")
        
        if not self.config['skip_confidence']:
            success = self.step6_confidence_scoring()
            if not success:
                print("⚠️  Confidence scoring failed, but continuing...")
        else:
            if not self.config['hyperopt_mode']:
                print("\n⏭️  Step 6: SKIPPED")
        
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        
        if not self.config['hyperopt_mode']:
            self._print_summary()
        
        return True

    def step1_preprocessing_part1(self, force_rerun=False):
        """
        Step 1: Preprocessing Part 1
        - Dump name publications
        - Extract features and relations
        
        Args:
            force_rerun: Se True, rifa il preprocessing anche se già esiste
        """
        print("\n" + "="*70)
        print("📚 STEP 1: PREPROCESSING (PART 1)")
        print("="*70)
        
        if not force_rerun and self._check_preprocessing_part1_done():
            print("\n✓ Preprocessing part 1 already done, skipping...")
            return True
        
        if force_rerun:
            print("\n🔄 Forcing preprocessing rerun (will overwrite existing files)...")
        
        try:
            print("\n[1/2] Loading and dumping name publications...")
            dump_name_pubs()
            
            print("\n[2/2] Creating features and relations...")
            dump_features_relations_to_file()
            
            print("\n✅ Step 1 completed!")
            return True
            
        except Exception as e:
            print(f"\n❌ Step 1 failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step2_word2vec(self, force_rerun=False):
        """
        Step 2: Word2Vec Training & Paper Embeddings
        
        Args:
            force_rerun: Se True, rifa il training anche se già esiste
        """
        print("\n" + "="*70)
        print("📖 STEP 2: WORD2VEC TRAINING & EMBEDDINGS")
        print("="*70)
        
        paper_emb_dir = self.base_path / 'paper_emb' / self.mode
        
        if not force_rerun and paper_emb_dir.exists() and any(paper_emb_dir.iterdir()):
            print("\n✓ Paper embeddings already exist for this mode, skipping...")
            return True
        
        if force_rerun:
            print("\n🔄 Forcing Word2Vec rerun (will regenerate embeddings)...")
        
        try:
            print("\nRunning Word2Vec training + embeddings generation...")
            
            w2v_script = Path(r"C:\Users\franc\OneDrive - Alma Mater Studiorum Università di Bologna\Desktop\BOND-OC\WhoIsWho\bond\train_w2v.py")
            
            if not w2v_script.exists():
                print(f"⚠️  Word2Vec script not found: {w2v_script}")
                print("   Continuing without Word2Vec (will use basic features)")
                return True
            
            cmd = [sys.executable, str(w2v_script)]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 min max
            )
            
            if result.returncode == 0:
                print(result.stdout)
                print("\n✅ Step 2 completed!")
            else:
                print(f"⚠️  Word2Vec had issues:")
                print(result.stderr)
                print("   Continuing without Word2Vec...")
            
            return True
            
        except subprocess.TimeoutExpired:
            print("⚠️  Word2Vec timed out, continuing...")
            return True
        except Exception as e:
            print(f"⚠️  Step 2 warning: {e}")
            print("   Continuing without Word2Vec...")
            return True
    
    def step3_preprocessing_part2(self, force_rerun=False):
        """
        Step 3: Preprocessing Part 2
        - Build graphs
        
        Args:
            force_rerun: Se True, rifa il preprocessing anche se già esiste
        """
        print("\n" + "="*70)
        print("🕸️  STEP 3: PREPROCESSING (PART 2)")
        print("="*70)
        
        if not force_rerun and self._check_preprocessing_part2_done():
            print("\n✓ Preprocessing part 2 already done, skipping...")
            return True
        
        if force_rerun:
            print("\n🔄 Forcing graph building rerun (will rebuild graphs)...")
        
        try:
            print("\n[1/1] Building graphs...")
            build_graph()
            
            print("\n✅ Step 3 completed!")
            return True
            
        except Exception as e:
            print(f"\n❌ Step 3 failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step4_training(self):
        """
        Step 4: Model Training
        """
        print("\n" + "="*70)
        print("🧠 STEP 4: MODEL TRAINING")
        print("="*70)
        
        try:
            model_type = self.config['model_type']
            print(f"\nTraining model: {model_type}")
            print(f"Mode: {self.mode}")
            
            if model_type == 'bond':
                trainer = BONDTrainer()
                trainer.fit(datatype=self.mode)
            elif model_type == 'bond+':
                trainer = ESBTrainer()
                trainer.fit(datatype=self.mode)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            pred_file = Path('out') / 'res.json'
            if pred_file.exists():
                self.results['predictions'] = str(pred_file)
                print(f"\n✅ Step 4 completed!")
                print(f"   Predictions saved: {pred_file}")
                return True
            else:
                print(f"\n❌ Predictions file not found: {pred_file}")
                return False
            
        except Exception as e:
            print(f"\n❌ Step 4 failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step5_evaluation(self):
        """
        Step 5: Multi-Metric Evaluation
        """
        print("\n" + "="*70)
        print("📊 STEP 5: MULTI-METRIC EVALUATION")
        print("="*70)
        
        if self.eval_mode != self.mode:
            print(f"\n⚠️  Using separate evaluation mode:")
            print(f"   Model trained on: {self.mode}")
            print(f"   Evaluating on: {self.eval_mode}")

        if 'predictions' not in self.results:
            pred_file = self.base_path.parent / 'out' / 'res.json'
            if not pred_file.exists():
                print(f"⚠️  Predictions file not found: {pred_file}")
                return False
            self.results['predictions'] = str(pred_file)
        
        gt_file = self._get_ground_truth_file(mode=self.eval_mode)
        if gt_file is None:
            print(f"⚠️  Ground truth not available for mode '{self.eval_mode}'")
            return False
        
        print(f"\nPredictions: {self.results['predictions']}")
        print(f"Ground truth: {gt_file}")
        
        try:
            from evaluate_BOND import MultiMetricEvaluator
            
            evaluator = MultiMetricEvaluator(
                self.results['predictions'],
                str(gt_file)
            )
            
            results = evaluator.evaluate_all_metrics()
            
            if not self.config['hyperopt_mode']:
                evaluator.print_results(detailed=True)
            else:
                print(f"\n  Composite Score: {results['composite_score']:.4f}")
                print(f"  Pairwise-F1:     {results['pairwise']['f1']:.4f}")
                print(f"  K-metric:        {results['k_metric']['k']:.4f}")
                print(f"  Cluster-F1:      {results['cluster']['f1']:.4f}")
            
            output_dir = Path('evaluation_results') 
            output_dir.mkdir(exist_ok=True)

            output_file = output_dir / 'evaluation_results.json'
            evaluator.save_results(output_file)
            
            self.results['evaluation'] = str(output_file)
            self.results['metrics'] = results
            
            print("\n✅ Step 5 completed!")
            return True
            
        except Exception as e:
            print(f"⚠️  Step 5 warning: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step6_confidence_scoring(self):
        """
        Step 6: Confidence Scoring & Filtering
        """
        print("\n" + "="*70)
        print("🎯 STEP 6: CONFIDENCE SCORING & FILTERING")
        print("="*70)
        
        if 'predictions' not in self.results:
            pred_file = Path('out') / 'res.json'
            if not pred_file.exists():
                print(f"⚠️  Predictions file not found: {pred_file}")
                return False
            self.results['predictions'] = str(pred_file)
        
        print(f"\nPredictions: {self.results['predictions']}")
        
        try:
            from cluster_confidence_scoring import ImprovedClusterConfidenceScorer
            
            output_file = Path('out') / 'filtered_predictions.json'
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if self.eval_mode == 'train':
                pub_file = self.base_path / 'src' / 'train' / 'train_pub.json'
            elif self.eval_mode == 'valid':
                pub_file = self.base_path / 'src' / 'sna-valid' / 'sna_valid_pub.json'
            elif self.eval_mode == 'test':
                pub_file = self.base_path / 'src' / 'sna-test' / 'sna_test_pub.json'
            else:
                pub_file = None
            
            pub_file_path = str(pub_file) if pub_file and pub_file.exists() else None
            
            scorer = ImprovedClusterConfidenceScorer(
                str(Path(self.results['predictions']).resolve()),
                pubs_file=pub_file_path
            )
            
            print("\nComputing confidence scores...")
            scorer.compute_all_scores()
            
            print("\nAnalyzing confidence distribution...")
            scorer.analyze_confidence_distribution()
            
            print("\nComputing recommended threshold...")
            threshold = scorer.recommend_threshold()
            
            print(f"\nFiltering clusters with threshold: {threshold:.2f}")
            scorer.save_filtered_results(
                str(output_file.resolve()),  
                threshold
            )
            
            if output_file.exists():
                self.results['filtered_predictions'] = str(output_file)
                print(f"\n✅ Step 6 completed!")
                print(f"   Filtered predictions: {output_file}")
                return True
            else:
                print("⚠️  Filtered predictions file not created")
                return False
            
        except Exception as e:
            print(f"⚠️  Step 6 warning: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def _check_preprocessing_part1_done(self):
        names_pub_dir = self.base_path / 'names_pub' / self.mode
        relations_dir = self.base_path / 'relations' / self.mode
        
        for dir_path in [names_pub_dir, relations_dir]:
            if not dir_path.exists() or not any(dir_path.iterdir()):
                return False
        
        return True
    
    def _check_preprocessing_part2_done(self):
        graphs_dir = self.base_path / 'graph' / self.mode
        
        if not graphs_dir.exists() or not any(graphs_dir.iterdir()):
            return False
        
        return True
    
    def _get_ground_truth_file(self, mode=None):
        """
        Trova il file ground truth per il mode specificato
        
        Args:
            mode: 'train', 'valid', o 'test'. Se None, usa self.mode
        """
        if mode is None:
            mode = self.mode
        
        if mode == 'train':
            gt_file = self.base_path / 'src' / 'train' / 'train_author.json'
        elif mode == 'valid':
            gt_file = self.base_path / 'src' / 'sna-valid' / 'sna_valid_ground_truth.json'
            if not gt_file.exists():
                gt_file = self.base_path / 'src' / 'sna-valid' / 'valid_ground_truth.json'
        elif mode == 'test':
            gt_file = self.base_path / 'src' / 'sna-test' / 'sna_test_ground_truth.json'
        else:
            return None
        
        return gt_file if gt_file.exists() else None
        
    def _print_summary(self):
        print("\n📋 PIPELINE SUMMARY")
        print("="*70)
        
        if 'predictions' in self.results:
            print(f"✅ Predictions: {Path(self.results['predictions']).name}")
        
        if 'evaluation' in self.results:
            print(f"✅ Evaluation: {Path(self.results['evaluation']).name}")
            
            if 'metrics' in self.results:
                metrics = self.results['metrics']
                print(f"\n   📊 Metrics:")
                print(f"      Composite Score: {metrics.get('composite_score', 0):.4f}")
                print(f"      Pairwise-F1:     {metrics.get('pairwise', {}).get('f1', 0):.4f}")
                print(f"      K-metric:        {metrics.get('k_metric', {}).get('k', 0):.4f}")
                print(f"      Cluster-F1:      {metrics.get('cluster', {}).get('f1', 0):.4f}")
        
        if 'filtered_predictions' in self.results:
            print(f"✅ Filtered predictions: {Path(self.results['filtered_predictions']).name}")
        
        print("="*70)
        print(f"\nAll results in: {self.base_path.parent / 'out'}")
    
    def get_metrics_for_optuna(self):
        if 'metrics' not in self.results:
            return None
        
        return self.results['metrics']


def run_single_training():
    """Modalità single training: esegue pipeline completa una volta"""
    pipeline = BondPipeline()
    pipeline.interactive_configuration()
    
    success = pipeline.run_full_pipeline()
    
    if success:
        print("\n🎉 Single training completed successfully!")
        return 0
    else:
        print("\n❌ Single training failed!")
        return 1


def run_for_hyperopt():
    """Modalità hyperopt: esegue solo training+evaluation"""
    pipeline = BondPipeline()
    success = pipeline.run_full_pipeline()
    
    if not success:
        return False
    
    metrics = pipeline.get_metrics_for_optuna()
    
    if metrics:
        print(f"\n📊 METRICS FOR OPTUNA:")
        print(f"   Composite Score: {metrics.get('composite_score', 0):.4f}")
        print(f"   Pairwise-F1:     {metrics.get('pairwise', {}).get('f1', 0):.4f}")
        print(f"   K-metric:        {metrics.get('k_metric', {}).get('k', 0):.4f}")
    
    return True


if __name__ == "__main__":
    hyperopt_mode = os.getenv("HYPEROPT_MODE") == "1"
    
    if hyperopt_mode:
        print("\n⚡ HYPEROPT MODE ACTIVATED")
        success = run_for_hyperopt()
        
        if not success:
            print("\n❌ Training failed!")
            sys.exit(1)
        
        sys.exit(0)
    else:
        exit_code = run_single_training()
        sys.exit(exit_code)
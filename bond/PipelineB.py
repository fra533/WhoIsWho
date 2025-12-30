"""
═══════════════════════════════════════════════════════════════════════
    PIPELINE B: PRODUCTION INFERENCE
═══════════════════════════════════════════════════════════════════════

Pipeline per fare inference con i migliori hyperparameter trovati.

QUANDO USARE:
✅ Hai già fatto esperimenti con PipelineA --mode train
✅ Hai trovato i migliori hyperparameter (da Optuna o manualmente)
✅ Vuoi fare inference su valid/test con questi parametri
✅ Vuoi confidence scoring e export automatici

USAGE:
------
# 1. Salva i tuoi best params in un JSON:
{
  "metadata": {
    "best_trial": 49,
    "f1_train": 0.809,
    "f1_valid": 0.811
  },
  "params": {
    "db_eps": 0.1,
    "lr": 0.0001,
    ...
  }
}

# 2. Run inference:
python PipelineB.py --mode valid --params best_params.json

# 3. Trova risultati in:
production_valid_YYYYMMDD_HHMMSS/
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
import argparse
import subprocess


class ProductionInference:
    """
    Wrapper per inference in produzione
    
    Usa PipelineA ma con:
    - Best hyperparameters pre-configurati
    - Mode = valid (o test)
    - Confidence scoring automatico
    - Export per produzione
    """
    
    def __init__(self, best_params_file=None, mode='valid', 
                 redo_preprocessing=False, redo_embeddings=False):
        """
        Args:
            best_params_file: JSON con best hyperparameters
            mode: 'valid' o 'test'
            redo_preprocessing: Se True, rifà preprocessing (Part 1 & 2)
            redo_embeddings: Se True, rifà Word2Vec e embeddings
        """
        self.mode = mode
        self.best_params = self._load_best_params(best_params_file)
        self.redo_preprocessing = redo_preprocessing
        self.redo_embeddings = redo_embeddings
        
        # Setup output directory
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(f'production_{self.mode}_{self.session_id}')
        self.output_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*70)
        print("🚀 PIPELINE B: PRODUCTION INFERENCE")
        print("="*70)
        print(f"Mode: {self.mode}")
        print(f"Session: {self.session_id}")
        print(f"Output: {self.output_dir}")
        
        # ========== MOSTRA OPZIONI ==========
        if redo_preprocessing or redo_embeddings:
            print("\n⚙️  Rebuild options:")
            if redo_preprocessing:
                print("   🔄 Will redo preprocessing (Part 1 & 2)")
            if redo_embeddings:
                print("   🔄 Will redo Word2Vec and embeddings")
        else:
            print("\n✅ Will use existing preprocessing and embeddings")
        # ====================================
    
    def _load_best_params(self, params_file):
        """
        Carica best hyperparameters
        
        Se params_file è None, usa valori di default
        """
        if params_file and Path(params_file).exists():
            with open(params_file, 'r', encoding='utf-8') as f:
                params = json.load(f)
            print(f"✓ Loaded best params from: {params_file}")
            return params
        else:
            print("❌ ERROR: No parameters file provided!")
            print("\n💡 Create a params file with:")
            print("   python PipelineB.py --create-template")
            sys.exit(1)
    
    def run(self):
        """Execute production inference pipeline"""
        print("\n" + "="*70)
        print("📋 PRODUCTION WORKFLOW")
        print("="*70)
        print("""
        Step 1: Setup Environment with Best Params
        Step 2: Run PipelineA Inference
        Step 3: Confidence Scoring & Filtering
        Step 4: Export Results
        Step 5: Generate Report
        """)
        
        input("\nPress ENTER to start...")
        
        try:
            # Step 1: Setup
            self.step1_setup_environment()
            
            # Step 2: Run inference (PipelineA)
            if not self.step2_run_inference():
                return False
            
            # Step 3: Confidence scoring
            if not self.step3_confidence_scoring():
                return False
            
            # Step 4: Export
            self.step4_export_results()
            
            # Step 5: Report
            self.step5_generate_report()
            
            self._print_final_summary()
            return True
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Pipeline interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step1_setup_environment(self):
        """Step 1: Setup environment variables con best params"""
        print("\n" + "="*70)
        print("⚙️  STEP 1: SETUP ENVIRONMENT")
        print("="*70)
        
        # ========== CONFIGURA ENVIRONMENT IN BASE ALLE OPZIONI ==========
        # NON impostiamo HYPEROPT_MODE perché vogliamo permettere il redo
        
        # Preprocessing
        if self.redo_preprocessing:
            # NON impostare SKIP_PREPROCESS, così PipelineA lo rifà
            print("\n🔄 Preprocessing will be redone")
        else:
            os.environ['SKIP_PREPROCESS'] = '1'
            print("\n✅ Using existing preprocessing")
        
        # Word2Vec & Embeddings
        if self.redo_embeddings:
            # NON impostare SKIP_W2V, così PipelineA lo rifà
            print("🔄 Word2Vec & embeddings will be redone")
        else:
            os.environ['SKIP_W2V'] = '1'
            print("✅ Using existing Word2Vec & embeddings")
        
        # Questi li skippiamo sempre in produzione
        os.environ['SKIP_CONFIDENCE'] = '1'  # Lo facciamo noi in step3
        os.environ['SKIP_EVAL'] = '1'        # No ground truth per test
        
        print("\n✓ Environment configured:")
        print(f"   Skip preprocessing: {not self.redo_preprocessing}")
        print(f"   Skip Word2Vec: {not self.redo_embeddings}")
        print(f"   Skip evaluation: True")
        print(f"   Skip confidence: True (done separately in step 3)")
        print(f"   Mode: {self.mode}")
        # ================================================================
        
        # Estrai metadata se presente
        metadata = self.best_params.get('metadata', {})
        if metadata:
            print(f"\n✓ Best trial info:")
            print(f"   Trial: {metadata.get('best_trial', 'N/A')}")
            if 'f1_train' in metadata:
                print(f"   F1 train: {metadata['f1_train']:.4f}")
            if 'f1_valid' in metadata:
                print(f"   F1 valid: {metadata['f1_valid']:.4f}")
            if 'timestamp' in metadata:
                print(f"   Found at: {metadata['timestamp']}")
        
        # Mostra params principali
        actual_params = self.best_params.get('params', self.best_params)
        print(f"\n✓ Hyperparameters loaded: {len(actual_params)} params")
        print(f"   Key params:")
        for key in ['lr', 'epochs', 'hidden_dim_0', 'db_eps']:
            if key in actual_params:
                print(f"     {key}: {actual_params[key]}")
        
        # Salva config per reference
        config_file = self.output_dir / 'inference_config.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump({
                'mode': self.mode,
                'session_id': self.session_id,
                'best_params': self.best_params,
                'rebuild_options': {
                    'redo_preprocessing': self.redo_preprocessing,
                    'redo_embeddings': self.redo_embeddings
                },
                'environment': {
                    'skip_preprocessing': not self.redo_preprocessing,
                    'skip_w2v': not self.redo_embeddings,
                    'skip_eval': True,
                    'skip_confidence': True
                },
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\n✓ Config saved: {config_file}")
    
    def step2_run_inference(self):
        """Step 2: Run PipelineA in inference mode"""
        print("\n" + "="*70)
        print("🔮 STEP 2: RUN INFERENCE (PipelineA)")
        print("="*70)
        
        # Build command per PipelineA
        cmd = [
            sys.executable,
            'PipelineA.py',
            '--mode', self.mode
        ]
        
        # ========== PARAMETRI BOND ==========
        # Mappa i parametri agli argomenti command-line
        param_mapping = {
            # DBSCAN
            'db_eps': '--db_eps',
            'db_min': '--db_min',
            'cluster_w': '--cluster_w',
            
            # Network architecture
            'hidden_dim_0': '--hidden_dim_0',
            'hidden_dim_1': '--hidden_dim_1',
            'compress_ratio': '--compress_ratio',
            
            # Training
            'lr': '--lr',
            'l2_coef': '--l2_coef',
            'epochs': '--epochs',
            
            # Thresholds author
            'th_a_0': '--th_a_0',
            'th_a_1': '--th_a_1',
            
            # Thresholds org
            'th_o_0': '--th_o_0',
            'th_o_1': '--th_o_1',
            
            # Thresholds venue
            'th_v_0': '--th_v_0',
            'th_v_1': '--th_v_1',
            
            # Citation weights
            'cite_out_weight': '--cite_out_weight',
            'cite_in_weight': '--cite_in_weight',
            'cite_out_th': '--cite_out_th',
            'cite_in_th': '--cite_in_th'
        }
        
        # Estrai params (può essere nested in 'params' key o direttamente)
        actual_params = self.best_params.get('params', self.best_params)
        
        # Aggiungi parametri al comando
        params_added = 0
        for param_key, cmd_arg in param_mapping.items():
            if param_key in actual_params:
                cmd.extend([cmd_arg, str(actual_params[param_key])])
                params_added += 1
        # ====================================
        
        print(f"\n🏃 Running PipelineA with best parameters...")
        print(f"   Mode: {self.mode}")
        print(f"   Parameters passed: {params_added}")
        
        # Print command (solo primi elementi per brevità)
        cmd_preview = ' '.join(cmd[:6]) + f' ... (+{len(cmd)-6} more args)'
        print(f"   Command: {cmd_preview}")
        
        if self.redo_preprocessing or self.redo_embeddings:
            print(f"\n⚠️  WARNING: PipelineA will ask for confirmation")
            print(f"   Please respond to the prompts:")
            if self.redo_preprocessing:
                print(f"     - Choose option 2 (Redo all preprocessing)")
                print(f"     - Confirm with 'y'")
            
            # NON usare capture_output se vogliamo interazione
            try:
                print(f"\n⏳ Running inference (this may take a while)...")
                print("="*70 + "\n")
                
                # Run PipelineA in modalità interattiva
                result = subprocess.run(
                    cmd,
                    env=os.environ.copy(),
                    timeout=7200  # 2 hours max
                )
                
                print("\n" + "="*70)
                
                if result.returncode != 0:
                    print(f"⚠️  PipelineA exited with code {result.returncode}")
                    return False
            
            except subprocess.TimeoutExpired:
                print("\n❌ Inference timed out (>2 hours)")
                return False
            except Exception as e:
                print(f"\n❌ Inference failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        else:
            # Modalità non-interattiva (usa preprocessing esistente)
            # Imposta HYPEROPT_MODE per skippare il menu
            env = os.environ.copy()
            env['HYPEROPT_MODE'] = '1'
            
            env['PYTHONIOENCODING'] = 'utf-8' 

            try:
                print(f"\n⏳ Running inference (non-interactive mode)...")
                
                result = subprocess.run(
                    cmd,
                    env=env,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',         
                    errors='replace',
                    timeout=7200
                )
                
                # Print output
                if result.stdout:
                    print("\n" + "="*70)
                    print("PIPELINEA OUTPUT:")
                    print("="*70)
                    print(result.stdout)
                
                if result.returncode != 0:
                    print("\n" + "="*70)
                    print("⚠️  PIPELINEA WARNINGS/ERRORS:")
                    print("="*70)
                    print(result.stderr)
                    return False
            
            except subprocess.TimeoutExpired:
                print("\n❌ Inference timed out (>2 hours)")
                return False
            except Exception as e:
                print(f"\n❌ Inference failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        # =================================================
        
        # Check predictions file
        pred_file = Path('out') / 'res.json'
        if not pred_file.exists():
            print(f"\n❌ Predictions file not found: {pred_file}")
            return False

        # Copy predictions to output dir
        shutil.copy(pred_file, self.output_dir / 'predictions.json')

        # Get stats
        with open(pred_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        n_names = len(predictions)
        n_clusters = sum(len(clusters) for clusters in predictions.values())
        
        print(f"\n✅ Inference completed!")
        print(f"   Predictions: {self.output_dir / 'predictions.json'}")
        print(f"   Authors: {n_names}")
        print(f"   Clusters: {n_clusters}")
        
        return True
            
    def step3_confidence_scoring(self):
        """Step 3: Confidence Scoring & Filtering"""
        print("\n" + "="*70)
        print("🎯 STEP 3: CONFIDENCE SCORING")
        print("="*70)
        
        pred_file = self.output_dir / 'predictions.json'
        
        # Trova pub file per questo mode
        base_path = Path('dataset/data')
        
        if self.mode == 'valid':
            pub_file = base_path / 'src' / 'sna-valid' / 'sna_valid_pub.json'
        elif self.mode == 'test':
            pub_file = base_path / 'src' / 'sna-test' / 'sna_test_pub.json'
        else:
            pub_file = None
        
        # Check esistenza
        if pub_file and not pub_file.exists():
            print(f"⚠️  Pub file not found: {pub_file}")
            print("   Will use simplified scoring (size-based only)")
            pub_file = None
        
        print(f"\nPredictions: {pred_file.name}")
        print(f"Publications: {pub_file.name if pub_file else 'Not available (using simplified scoring)'}")
        
        try:
            from cluster_confidence_scoring import ImprovedClusterConfidenceScorer
            
            # Create scorer
            scorer = ImprovedClusterConfidenceScorer(
                str(pred_file),
                pubs_file=str(pub_file) if pub_file else None
            )
            # Compute scores
            print("\n📊 Computing confidence scores...")
            scorer.compute_all_scores()
            print("   ✓ Scores computed")
            
            # Analyze distribution
            print("\n📈 Analyzing confidence distribution...")
            scorer.analyze_confidence_distribution()
            
            # Recommend threshold
            print("\n🎯 Computing recommended threshold...")
            threshold = scorer.recommend_threshold()
            
            # Filter and save
            print(f"\n💾 Filtering clusters with threshold: {threshold:.2f}")
            filtered_file = self.output_dir / 'filtered_predictions.json'
            scorer.save_filtered_results(str(filtered_file), threshold)
            
            print(f"\n✅ Confidence scoring completed!")
            print(f"   Filtered predictions: {filtered_file}")
            
            return True
        
        except ImportError:
            print(f"\n⚠️  cluster_confidence_scoring.py not found")
            print("   Skipping confidence scoring...")
            return True
        except Exception as e:
            print(f"\n⚠️  Confidence scoring failed: {e}")
            print("   Continuing without filtering...")
            import traceback
            traceback.print_exc()
            return True  # Continue even if confidence fails
    
    def step4_export_results(self):
        """Step 4: Export Results"""
        print("\n" + "="*70)
        print("📦 STEP 4: EXPORT RESULTS")
        print("="*70)
        
        export_dir = self.output_dir / 'exports'
        export_dir.mkdir(exist_ok=True)
        
        # Check quale file usare
        filtered_file = self.output_dir / 'filtered_predictions.json'
        pred_file = self.output_dir / 'predictions.json'
        
        source_file = filtered_file if filtered_file.exists() else pred_file
        
        print(f"\nSource: {source_file.name}")
        
        try:
            # 1. Database format
            print("\n📊 Creating database export...")
            self._create_database_export(source_file, export_dir)
            
            # 2. API format
            print("🔌 Creating API export...")
            self._create_api_export(source_file, export_dir)
            
            # 3. CSV format
            print("📄 Creating CSV export...")
            self._create_csv_export(source_file, export_dir)
            
            print(f"\n✅ All exports saved to: {export_dir}")
        except Exception as e:
            print(f"\n⚠️  Export failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_database_export(self, source_file, export_dir):
        """Create database import format"""
        with open(source_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        # JSON format
        db_records = []
        cluster_id = 1
        
        for name, clusters in predictions.items():
            for cluster_idx, papers in enumerate(clusters):
                # Handle sia formato base che filtered (con confidence)
                if isinstance(papers, dict):
                    paper_list = papers.get('papers', papers.get('cluster', []))
                else:
                    paper_list = papers
                
                db_records.append({
                    'cluster_id': cluster_id,
                    'author_name': name,
                    'cluster_index': cluster_idx,
                    'paper_ids': paper_list,
                    'n_papers': len(paper_list),
                    'created_at': datetime.now().isoformat()
                })
                cluster_id += 1
        
        db_file = export_dir / 'database_import.json'
        with open(db_file, 'w', encoding='utf-8') as f:
            json.dump(db_records, f, indent=2, ensure_ascii=False)
        
        print(f"   ✓ {db_file.name} ({len(db_records)} records)")
    
    def _create_api_export(self, source_file, export_dir):
        """Create API format"""
        with open(source_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        api_format = {
            'metadata': {
                'mode': self.mode,
                'session_id': self.session_id,
                'generated_at': datetime.now().isoformat(),
                'n_authors': len(predictions),
                'n_clusters': sum(len(clusters) for clusters in predictions.values())
            },
            'clusters': predictions
        }
        
        api_file = export_dir / 'api_format.json'
        with open(api_file, 'w', encoding='utf-8') as f:
            json.dump(api_format, f, indent=2, ensure_ascii=False)
        
        print(f"   ✓ {api_file.name}")
    
    def _create_csv_export(self, source_file, export_dir):
        """Create CSV format"""
        import csv
        
        with open(source_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        csv_file = export_dir / 'clusters.csv'
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Author Name', 'Cluster Index', 'N Papers', 'Paper IDs'])
            
            for name, clusters in predictions.items():
                for idx, papers in enumerate(clusters):
                    if isinstance(papers, dict):
                        paper_list = papers.get('papers', papers.get('cluster', []))
                    else:
                        paper_list = papers
                    
                    writer.writerow([
                        name,
                        idx,
                        len(paper_list),
                        ', '.join(paper_list)
                    ])
        
        print(f"   ✓ {csv_file.name}")
    
    def step5_generate_report(self):
        """Step 5: Generate Report"""
        print("\n" + "="*70)
        print("📄 STEP 5: GENERATE REPORT")
        print("="*70)
        
        # Collect statistics
        pred_file = self.output_dir / 'predictions.json'
        
        with open(pred_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        n_names = len(predictions)
        n_clusters = sum(len(clusters) for clusters in predictions.values())
        n_papers = sum(
            len(papers if isinstance(papers, list) else papers.get('papers', []))
            for clusters in predictions.values()
            for papers in clusters
        )
        
        # Check if filtered exists
        filtered_file = self.output_dir / 'filtered_predictions.json'
        has_filtered = filtered_file.exists()
        
        if has_filtered:
            with open(filtered_file, 'r', encoding='utf-8') as f:
                filtered = json.load(f)
            
            n_filtered_names = len(filtered)
            n_filtered_clusters = sum(len(clusters) for clusters in filtered.values())
            n_filtered_papers = sum(
                len(papers if isinstance(papers, list) else papers.get('papers', []))
                for clusters in filtered.values()
                for papers in clusters
            )
        else:
            n_filtered_names = n_names
            n_filtered_clusters = n_clusters
            n_filtered_papers = n_papers
        
        # Create report
        report = {
            'session': {
                'id': self.session_id,
                'mode': self.mode,
                'timestamp': datetime.now().isoformat()
            },
            'statistics': {
                'raw': {
                    'n_authors': n_names,
                    'n_clusters': n_clusters,
                    'n_papers': n_papers,
                    'avg_clusters_per_author': n_clusters / max(1, n_names),
                    'avg_papers_per_cluster': n_papers / max(1, n_clusters)
                },
                'filtered': {
                    'n_authors': n_filtered_names,
                    'n_clusters': n_filtered_clusters,
                    'n_papers': n_filtered_papers,
                    'avg_clusters_per_author': n_filtered_clusters / max(1, n_filtered_names),
                    'avg_papers_per_cluster': n_filtered_papers / max(1, n_filtered_clusters)
                } if has_filtered else None
            },
            'configuration': {
                'best_params': self.best_params
            },
            'outputs': {
                'predictions': str(self.output_dir / 'predictions.json'),
                'filtered': str(filtered_file) if has_filtered else None,
                'exports': str(self.output_dir / 'exports')
            }
        }
        
        # Save report
        report_file = self.output_dir / 'production_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Report generated: {report_file}")
        
        # Print summary
        print("\n📊 Statistics:")
        print(f"\n   Raw predictions:")
        print(f"     Authors: {n_names}")
        print(f"     Clusters: {n_clusters}")
        print(f"     Papers: {n_papers}")
        print(f"     Avg clusters/author: {n_clusters/max(1,n_names):.1f}")
        print(f"     Avg papers/cluster: {n_papers/max(1,n_clusters):.1f}")
        
        if has_filtered:
            print(f"\n   Filtered (high confidence):")
            print(f"     Authors: {n_filtered_names}")
            print(f"     Clusters: {n_filtered_clusters}")
            print(f"     Papers: {n_filtered_papers}")
            print(f"     Retention: {100*n_filtered_clusters/max(1,n_clusters):.1f}%")
    
    def _print_final_summary(self):
        """Print final summary"""
        print("\n" + "="*70)
        print("✅ PRODUCTION INFERENCE COMPLETED")
        print("="*70)
        print(f"\n📁 Session: {self.session_id}")
        print(f"📂 Output directory: {self.output_dir}")
        print(f"\n📊 Files created:")
        print(f"   • predictions.json          - Raw predictions")
        
        filtered_file = self.output_dir / 'filtered_predictions.json'
        if filtered_file.exists():
            print(f"   • filtered_predictions.json - High confidence only")
        
        print(f"   • exports/                  - DB, API, CSV formats")
        print(f"   • production_report.json    - Complete statistics")
        print(f"   • inference_config.json     - Parameters used")
        


def main():
    parser = argparse.ArgumentParser(
        description='Pipeline B: Production Inference with Best Params',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create template params file
  python PipelineB.py --create-template

  # Run inference on validation set (use existing preprocessing)
  python PipelineB.py --mode valid --params best_params.json

  # Run inference with fresh preprocessing
  python PipelineB.py --mode valid --params best_params.json --redo-preprocessing

  # Run inference with fresh embeddings only
  python PipelineB.py --mode valid --params best_params.json --redo-embeddings

  # Run inference redoing everything
  python PipelineB.py --mode valid --params best_params.json --redo-all
        """
    )
    parser.add_argument('--mode', '-m', default='valid',
                       choices=['valid', 'test'],
                       help='Inference mode (default: valid)')
    parser.add_argument('--params', '-p', default=None,
                       help='JSON file with best hyperparameters')
    parser.add_argument('--create-template', action='store_true',
                       help='Create template best_params.json and exit')
    
    # ========== NUOVE OPZIONI ==========
    parser.add_argument('--redo-preprocessing', action='store_true',
                       help='Redo preprocessing (Part 1 & 2)')
    parser.add_argument('--redo-embeddings', action='store_true',
                       help='Redo Word2Vec and embeddings')
    parser.add_argument('--redo-all', action='store_true',
                       help='Redo everything (preprocessing + embeddings)')
    # ===================================
    
    args = parser.parse_args()
    
    # Create template
    if args.create_template:
        # ... codice template uguale ...
        return
    
    # Check params file provided
    if not args.params:
        print("❌ ERROR: --params argument required")
        print("\n💡 Usage:")
        print("   1. Create template: python PipelineB.py --create-template")
        print("   2. Edit the template with your best params")
        print("   3. Run: python PipelineB.py --params your_params.json")
        sys.exit(1)
    
    # ========== PASSA LE OPZIONI A ProductionInference ==========
    pipeline = ProductionInference(
        best_params_file=args.params,
        mode=args.mode,
        redo_preprocessing=args.redo_preprocessing or args.redo_all,
        redo_embeddings=args.redo_embeddings or args.redo_all
    )
    # ============================================================
    
    success = pipeline.run()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
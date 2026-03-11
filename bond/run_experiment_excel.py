import subprocess
import os
import sys
import json
import pandas as pd
import time
from datetime import datetime
from pathlib import Path

# ==============================================================================
# ⚙️ CONFIGURAZIONE PARAMETRI DI DEFAULT (Da params.py)
# ==============================================================================
DEFAULT_PARAMS = {
    'mode': 'valid',
    'cuda': True,
    'gpu': 0,
    'seed': 0,
    'epochs': 0,
    'lr': 5e-6,
    'save_path': 'dataset/data',
    'l2_coef': 2.714679239042756e-05,
    'compress_ratio': 0.5,
    'hidden_dim': [768, 512],
    'rel_on': 'aov',
    'emb_type': 'specter',
    'cluster_w': 0.0,
    'prob_v': 0.9,
    'coa_th': 2,
    'coo_th': 0.85,
    'cov_th': 1,
    'coc_th': 2,
    'coi_th': 1,
    'db_eps': 0.09,
    'db_min': 2,
    'post_match': False,
    'th_a': [0, 1],
    'th_o': [0.6, 0.5],
    'th_v': [1, 2],
    'th_c': [0, 0],
    'th_i': [0, 0],
    'repeat_num': 1,
    'cite_out_weight': 0.3,
    'cite_in_weight': 0.2,
    'use_citations': True
}

# ==============================================================================
# 🔧 FUNZIONI DI UTILITÀ
# ==============================================================================

def get_user_choice():
    """Chiede all'utente quale embedding usare"""
    print("\n" + "="*60)
    print("🎛️  CONFIGURAZIONE SESSIONE SPERIMENTALE")
    print("="*60)
    print("Quale tipo di embedding vuoi utilizzare per questi test?")
    print("  1) Specter (Default - Consigliato per Paper)")
    print("  2) Word2Vec (w2v)")
    print("  3) SBERT (all-mpnet-base-v2)")
    
    while True:
        choice = input("\nInserisci scelta [1, 2 o 3]: ").strip().lower()
        if choice in ['1', 'specter', 's']:
            print("✅ Selezionato: SPECTER")
            return 'specter'
        elif choice in ['2', 'w2v', 'w']:
            print("✅ Selezionato: WORD2VEC")
            return 'w2v'
        elif choice in ['3', 'sbert', 'sb']: 
            print("✅ Selezionato: SBERT")
            return 'sbert'
        else:
            print("❌ Scelta non valida. Riprova.")

def add_params_to_cmd(cmd, params_dict):
    """Helper per aggiungere parametri al comando"""
    for k, v in params_dict.items():
        if isinstance(v, list):
            cmd.append(f"--{k}")
            cmd.extend([str(x) for x in v])
        else:
            cmd.extend([f"--{k}", str(v)])

def save_excel(data, filename):
    if not data:
        return
    df = pd.DataFrame(data)
    
    # Colonne Prioritarie
    priority_cols = [
        "Experiment_ID","Experiment_Name", "Status", "GNN_Active", "Emb_Type", "Mode", "Composite_Score",
        "Pairwise_F1", "K_Metric_K", "B3_F1", "Cluster_F1",
        "Splitting_Error", "Lumping_Error",
        "Pairwise_Prec", "Pairwise_Rec",
        "B3_Prec", "B3_Rec",
        "Full_Configuration",
        "Duration_sec"
    ]
    
    other_cols = [c for c in df.columns if c not in priority_cols]
    final_order = [c for c in priority_cols if c in df.columns] + other_cols
    
    df = df[final_order]
    
    try:
        df.to_excel(filename, index=False)
    except PermissionError:
        print(f"⚠️ Impossibile salvare Excel (forse è aperto?). Riprovo con nome diverso...")
        new_name = str(filename).replace(".xlsx", f"_backup_{int(time.time())}.xlsx")
        df.to_excel(new_name, index=False)
        print(f"   Salvato in: {new_name}")

# ==============================================================================
# 🚀 MAIN RUNNER
# ==============================================================================

def run_session():
    # 1. Chiedi input all'utente
    selected_emb = get_user_choice()
    
    # 2. Configura dimensioni in base alla scelta
    if selected_emb == 'specter':
        current_hidden_dim = [768, 512]
    elif selected_emb == 'sbert':
        current_hidden_dim = [768, 512]
    else: # w2v
        current_hidden_dim = [256, 128] # Modifica qui se i tuoi w2v hanno dim diverse

    # 3. Parametri Comuni per questa sessione
    common_params = {
        "cuda": True,
        "gpu": 0,
        "emb_type": selected_emb,
        "hidden_dim": current_hidden_dim,
        # PATH CORRETTO
        "save_path": r"C:\Users\franc\OneDrive - Alma Mater Studiorum Università di Bologna\Desktop\BOND-OC\WhoIsWho\bond\dataset\data"
    }

    # 4. Lista Esperimenti (Template Generico)
    # Nota: I parametri qui sotto (eps, min) sono specifici per l'esperimento
    experiments_template = [
    # ===========================================================================
    # 🧪 D3: SBERT(all-mpnet-base-v2) + OC + CIT + GNN - TARGET
    # ===========================================================================
    {
        "Experiment_ID": "D3_v2_TR",
        "suffix": "SBERT_OC_",
        "mode": "train",
        "no_gnn": "True",
        "params": {
            "db_eps": 0.3, #[0.15-1]
            "db_min": 4, 
            "hidden_dim": [768, 512],
            #"cite_out_weight": 0.3,
            #"cite_in_weight":0.1,
            #"epochs": 0,
            #"lr": 1e-4,
            #"cluster_w":0.2,

        }
    },
    {
        "Experiment_ID": "D3_v2_VAL",
        "suffix": "SBERT_OC",
        "mode": "valid",
        "no_gnn": "True",
        "params": {
            "db_eps": 0.3,
            "db_min": 4, 
            "hidden_dim": [768, 512],
            #"cite_out_weight": 0.3,
            #"cite_in_weight":0.1,
            #"epochs": 10,
            #"lr": 1e-4,
            #"cluster_w":0.2,

        }
    }
    # ===========================================================================
    # 🧪 D2: SBERT(all-mpnet-base-v2) + WhoIsWho + CIT + GNN
    # ===========================================================================
    #{
    #    "suffix": "WHOISWHO_CIT_TRAIN_GNN",
    #    "mode": "train",
    #    "no_gnn": "0",
    #    "params": {
    #        "db_eps": 0.20,
    #        "db_min": 5,
    #        "epochs": 10,
    #        "lr": 5e-6,
    #        "use_citations": True,
    #        "emb_type": "sbert"
    #    },
    #    "Experiment_ID": "D2"
    #},
    #{
    #    "suffix": "WHOISWHO_CIT_VALID_GNN",
    #    "mode": "valid",
    #    "no_gnn": "0",
    #    "params": {
    #        "db_eps": 0.20,
    #        "db_min": 5,
    #        "use_citations": True,
    #        "emb_type": "sbert"
    #    },
    #    "Experiment_ID": "D2"
    #}
    # ===========================================================================
    # 🧪 D4: SBERT + WhoIsWho + CIT +NO GNN
    # ===========================================================================
    #{
    #    "suffix": "WHOISWHO_CIT_TRAIN_NO_GNN",
    #    "mode": "train",
    #    "no_gnn": "1",
    #    "params": {
    #    "db_eps": 0.13,
    #    "db_min": 4,
    #    "lr": 5e-6,
    #    "use_citations": True,
    #},
    #    "Experiment_ID": "D4"
    #},
    #{
    #    "suffix": "WHOISWHO_CIT_VALID_NO_GNN",
    #    "mode": "valid",
    #    "no_gnn": "1",
    #    "params": {
    #        "db_eps": 0.13,
    #        "db_min": 4,
    #        "use_citations": True,
    #    },
    #    "Experiment_ID": "D4"
    #}
    # ===========================================================================
    # 🧪 C2: SPECTER + WhoIsWho + CIT + GNN
    # ===========================================================================
    #{
    #    "id": "C2",
    #    "suffix": "SPECTER_CIT_TRAIN_GNN",
    #    "mode": "train",
    #    "no_gnn": "0",
    #    "params": {
    #        "epochs": 50,
    #        "lr": 5e-5,
    #        "cluster_w": 0.1,
    #        "compress_ratio": 0.25,
    #        "db_eps": 0.2,
    #        "db_min": 2,
    #        "use_citations": True
    #        }
    #},
    #{
    #    "suffix": "SPECTER_CIT_VALID_GNN",
    #    "mode": "valid",
    #    "no_gnn": "0",
    #    "params": {"db_eps": 0.20, "db_min": 2,"epochs": 0, "use_citations": True}},

    # ===========================================================================
    # 🧪 C4: SPECTER + WhoIsWho + CIT + NO GNN
    # ===========================================================================
    #{
    #    "id": "C4",  
    #   "suffix": "SPECTER_CIT_TRAIN_NO_GNN",
    #    "mode": "train",
    #    "no_gnn": "1",
    #    "params": {"db_eps": 0.20, "db_min": 2, "use_citations": True}
    #},
    #{
    #    "suffix": "SPECTER_CIT_VALID_NO_GNN",
    #    "mode": "valid",
    #    "no_gnn": "1",
    #    "params": {"db_eps": 0.20, "db_min": 2, "use_citations": True}
    #},
    # ===========================================================================
    # 🧪 B1: W2V + WhoIsWho + CIT + GNN
    # ===========================================================================
    #{
    #    "suffix": "W2V_WHOISWHO_CIT_TRAIN_GNN",
    #    "mode": "train",
    #    "no_gnn": "0",
    #    "params": {"db_eps": 0.01, "db_min": 5, "epochs": 0, "lr": 5e-6, "use_citations": True}
    #},
    #{
    #    "suffix": "W2V_WHOISWHO_CIT_VALID_GNN",
    #    "mode": "valid",
    #    "no_gnn": "0",
    #    "params": {"db_eps": 0.01, "db_min": 5, "use_citations": True}
    #},
    # ===========================================================================
    # 🧪 B2: W2V + OC + CIT + GNN
    # ===========================================================================
    #{
    #    "suffix": "W2V_OC_CIT_TRAIN_GNN",
    #    "mode": "train",
    #    "no_gnn": "0",
    #    "params": {"db_eps": 0.01, "db_min": 5, "epochs": 0, "lr": 5e-6, "use_citations": True}
    #},
    #{
    #    "suffix": "W2V_OC_CIT_VALID_GNN",
    #    "mode": "valid",
    #    "no_gnn": "0",
    #    "params": {"db_eps": 0.01, "db_min": 5, "use_citations": True}
    #},    
     # ===========================================================================1
        
    # 🧪 C1: SPECTER + WhoIsWho + NO CIT + GNN
    # ===========================================================================
     #   {
    #        "suffix": "WHOISWHO_NO_CIT_TRAIN_GNN",
     #       "mode": "train",
      #      "no_gnn": "0",
      #      "params": {"db_eps": 0.01, "db_min": 5, "epochs": 0, "lr": 5e-6, "use_citations": False}
      #  },
      #  {
      #      "suffix": "WHOISWHO_NO_CIT_VALID_GNN",
       #     "mode": "valid",
      #      "no_gnn": "0",
       #     "params": {"db_eps": 0.01, "db_min": 5, "use_citations": False}
       # },
        
        # ===========================================================================
        # 🧪 C5: SPECTER + WhoIsWho + NO CIT + NO GNN (ablation)
        # ===========================================================================
        #
        #    "suffix": "WHOISWHO_NO_CIT_TRAIN_NO_GNN",
        #    "mode": "train",
        #    "no_gnn": "1",
        #    "params": {"db_eps": 0.09, "db_min": 2, "epochs": 0, "lr": 5e-6, "use_citations": False}
        #},
        #{
        #    "suffix": "WHOISWHO_NO_CIT_VALID_NO_GNN",
        #    "mode": "valid",
        #    "no_gnn": "1",
        #    "params": {"db_eps": 0.09, "db_min": 2, "use_citations": False}
        #},
            # ===========================================================================
        # 🧪 C1: SPECTER + WhoIsWho + NO CIT + GNN
        # ===========================================================================
        #{
        #    "suffix": "WHOISWHO_NO_CIT_TRAIN_GNN",
        #    "mode": "train",
        #    "no_gnn": "0",
        #    "params": {"db_eps": 0.01, "db_min": 5, "epochs": 0, "lr": 5e-6, "use_citations": False}
        #},
        #{
        #    "suffix": "WHOISWHO_NO_CIT_VALID_GNN",
        #    "mode": "valid",
        #    "no_gnn": "0",
        #    "params": {"db_eps": 0.01, "db_min": 5, "use_citations": False}
        #},
    
        # --- TRAIN ---
        #{
        #    "suffix": "TRAIN_GNN",
        #    "mode": "train",
        #    "no_gnn": "0",
        #    "params": {"db_eps": 0.01, "db_min": 5, "epochs": 0, "lr": 5e-6}
        #},
        #{
        #    "suffix": "TRAIN_NO_GNN",
        #    "mode": "train",
        #    "no_gnn": "1",
        #    "params": {"db_eps": 0.09, "db_min": 2, "epochs": 0, "lr": 5e-6}
        #},
        # --- VALID ---
        #{
        #    "suffix": "VALID_GNN",
        #    "mode": "valid",
        #    "no_gnn": "0",
        #    "params": {"db_eps": 0.01, "db_min": 5}
        #},
        #{
        #    "suffix": "VALID_NO_GNN",
        #    "mode": "valid",
        #    "no_gnn": "1",
        #    "params": {"db_eps": 0.09, "db_min": 2}
        #},
        
    ]

    results_list = []
    
    # Setup Output
    output_dir = Path("out/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Nome file dinamico
    excel_filename = output_dir / f"results_{selected_emb.upper()}_{timestamp}.xlsx"

    print(f"\n📂 Report Excel sarà salvato in: {excel_filename}")
    
    python_exe = sys.executable
    script_name = r"C:\Users\franc\OneDrive - Alma Mater Studiorum Università di Bologna\Desktop\BOND-OC\WhoIsWho\bond\PipelineA.py"

    # 5. Ciclo Esperimenti
    for idx, exp_data in enumerate(experiments_template):
        exp_id = exp_data.get("Experiment_ID", "N/A")
        exp_name = f"{selected_emb.upper()}_{exp_data['suffix']}"
        
        print("\n" + "="*80)
        print(f"[{exp_id}]🧪 ESECUZIONE {idx+1}/{len(experiments_template)}: {exp_name}")
        print("="*80)

        # Costruisci Configurazione Completa (per il salvataggio)
        full_config = DEFAULT_PARAMS.copy()
        full_config.update(common_params)
        full_config.update(exp_data['params'])
        full_config['mode'] = exp_data['mode']
        full_config['GNN_Active'] = "NO" if exp_data["no_gnn"] == "1" else "YES"

        full_config['Experiment_ID'] = exp_id
        
        # Prepara Environment
        env = os.environ.copy()
        env["NO_GNN"] = exp_data["no_gnn"]
        env["SKIP_PREPROCESS"] = "0"
        
        # Gestione W2V skip
        if selected_emb == 'w2v':
             env["SKIP_W2V"] = "0" # Non saltare se serve w2v
        else:
             env["SKIP_W2V"] = "1" # Salta w2v se usi specter
        
        # Costruisci comando
        cmd = [python_exe, script_name]
        cmd.extend(["--mode", exp_data["mode"]])
        add_params_to_cmd(cmd, common_params)
        add_params_to_cmd(cmd, exp_data["params"])

        print(f"📝 Info: mode={exp_data['mode']}, emb={selected_emb}, GNN={full_config['GNN_Active']}")
        
        # Inizializza metriche
        metrics = {
            "Composite_Score": None,
            "Pairwise_F1": None, "Pairwise_Prec": None, "Pairwise_Rec": None,
            "K_Metric_K": None, "K_Metric_AAP": None, "K_Metric_ACP": None,
            "B3_F1": None, "B3_Prec": None, "B3_Rec": None,
            "Cluster_F1": None, "Cluster_Prec": None, "Cluster_Rec": None,
            "Splitting_Error": None, "Lumping_Error": None, "Split_Lump_F1": None
        }

        # Esegui
        start_time = time.time()
        duration = 0
        status = "UNKNOWN"

        try:
            process = subprocess.run(cmd, env=env, text=True)
            duration = time.time() - start_time
            
            if process.returncode != 0:
                print(f"❌ Errore nell'esecuzione dell'esperimento {exp_name}")
                status = "FAILED"
            else:
                status = "SUCCESS"
                json_path = Path("evaluation_results/evaluation_results.json")
                
                if json_path.exists():
                    try:
                        with open(json_path, 'r') as f:
                            d = json.load(f)
                        
                        pw = d.get("pairwise", {})
                        km = d.get("k_metric", {})
                        b3 = d.get("b3", {})
                        cl = d.get("cluster", {})
                        sl = d.get("split_lump", {})
                        
                        metrics = {
                            "Composite_Score": d.get("composite_score"),
                            "Pairwise_F1": pw.get("f1"), "Pairwise_Prec": pw.get("precision"), "Pairwise_Rec": pw.get("recall"),
                            "K_Metric_K": km.get("k"), "K_Metric_AAP": km.get("aap"), "K_Metric_ACP": km.get("acp"),
                            "B3_F1": b3.get("f1"), "B3_Prec": b3.get("precision"), "B3_Rec": b3.get("recall"),
                            "Cluster_F1": cl.get("f1"), "Cluster_Prec": cl.get("precision"), "Cluster_Rec": cl.get("recall"),
                            "Splitting_Error": sl.get("splitting_error"), "Lumping_Error": sl.get("lumping_error"), "Split_Lump_F1": sl.get("f1")
                        }
                    except Exception as e:
                        print(f"⚠️ Errore lettura JSON: {e}")
                        status = "JSON_ERROR"
                else:
                    print("⚠️ File risultati JSON non trovato.")
                    status = "NO_RESULTS"

        except Exception as e:
            print(f"❌ Exception critica: {e}")
            status = "CRASHED"
            duration = time.time() - start_time

        row = {
            "Experiment_ID": exp_id,  
            "Experiment_Name": exp_name,
            "Status": status,
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Duration_sec": round(duration, 2),
            "Mode": exp_data["mode"],
            "Emb_Type": selected_emb,
            "GNN_Active": full_config['GNN_Active'],
        }
        
        row['Full_Configuration'] = json.dumps(full_config, indent=2)
        row.update(metrics)
        results_list.append(row)
        
        save_excel(results_list, excel_filename)

    print("\n" + "="*80)
    print(f"✅ SESSIONE COMPLETATA ({selected_emb.upper()})")
    print(f"📊 Report finale: {excel_filename}")
    print("="*80)

if __name__ == "__main__":
    try:
        import pandas
        import openpyxl
    except ImportError:
        print("❌ Mancano librerie necessarie. Esegui: pip install pandas openpyxl")
        sys.exit(1)
        
    run_session()
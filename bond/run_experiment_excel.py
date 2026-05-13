import subprocess
import os
import sys
import json
import pandas as pd
import time
from datetime import datetime
from pathlib import Path

# ==============================================================================
# ℹ️  NOTA SUI DEFAULT
# ==============================================================================
# I default dei parametri vivono in UN SOLO POSTO: params.py.
# Qui si dichiarano SOLO i parametri che cambiano tra una sessione/esperimento
# e l'altro. Tutto il resto viene gestito da argparse con i suoi default.
# ==============================================================================

SAVE_PATH = r"C:\Users\franc\OneDrive - Alma Mater Studiorum Università di Bologna\Desktop\BOND-OC\WhoIsWho\bond\dataset\data"
PIPELINE_SCRIPT = r"C:\Users\franc\OneDrive - Alma Mater Studiorum Università di Bologna\Desktop\BOND-OC\WhoIsWho\bond\PipelineA.py"

# ==============================================================================
# 🔧 FUNZIONI DI UTILITÀ
# ==============================================================================

def get_session_config():
    """Chiede all'utente Dataset e Embedding per la sessione."""
    print("\n" + "="*60)
    print("🎛️  CONFIGURAZIONE SESSIONE SPERIMENTALE")
    print("="*60)
    
    # 1. Scelta Dataset
    print("Seleziona il DATASET di riferimento:")
    print("  1) WhoIsWho")
    print("  2) OpenCitations (OC)")
    
    dataset = 'whoiswho'
    while True:
        ds_choice = input("\nInserisci scelta [1 o 2]: ").strip()
        if ds_choice == '1':
            dataset = 'whoiswho'
            print("✅ Dataset: WHOISWHO")
            break
        elif ds_choice == '2':
            dataset = 'oc'
            print("✅ Dataset: OC")
            break
        else:
            print("❌ Scelta non valida.")

    # 2. Scelta Embedding
    print("\nQuale tipo di embedding vuoi utilizzare?")
    print("  1) Specter2")
    print("  2) SBERT")

    embedding = 'specter'
    while True:
        emb_choice = input("\nInserisci scelta [1 o 2]: ").strip()
        if emb_choice == '1':
            embedding = 'specter'
            print("✅ Embedding: SPECTER")
            break
        elif emb_choice == '2':
            embedding = 'sbert'
            print("✅ Embedding: SBERT")
            break
        else:
            print("❌ Scelta non valida.")

    return dataset, embedding


def build_cmd(python_exe, script, mode, no_gnn, common_params, exp_params):
    """
    Costruisce il comando subprocess passando tutti i parametri via argparse.
    NO_GNN viene passato come --no_gnn (flag), non solo come env var.
    USE_CITATIONS viene passato come --use_citations (flag).
    """
    cmd = [python_exe, script, "--mode", mode]

    # Parametri comuni di sessione (emb_type, hidden_dim, save_path, ecc.)
    for k, v in common_params.items():
        if isinstance(v, list):
            cmd.append(f"--{k}")
            cmd.extend([str(x) for x in v])
        else:
            cmd.extend([f"--{k}", str(v)])

    # Parametri specifici dell'esperimento
    for k, v in exp_params.items():
        if k == "use_citations":
            # store_true: il flag va aggiunto solo se True
            if v:
                cmd.append("--use_citations")
            # se False non si aggiunge nulla — il default in params.py è False
        elif k == "no_gnn":
            # gestito separatamente dal parametro no_gnn dell'esperimento
            pass
        elif isinstance(v, list):
            cmd.append(f"--{k}")
            cmd.extend([str(x) for x in v])
        else:
            cmd.extend([f"--{k}", str(v)])

    # --no_gnn è uno store_true: aggiungilo solo se richiesto
    if no_gnn:
        cmd.append("--no_gnn")

    return cmd


def build_env(use_citations, no_gnn, selected_emb):
    """
    Costruisce l'environment del subprocess.
    Le env var rimangono come canale di controllo secondario per
    compatibilità con parti della pipeline che le leggono ancora,
    ma il canale primario è ora argparse (cmd).
    """
    env = os.environ.copy()
    env["NO_GNN"]          = "1" if no_gnn else "0"
    env["USE_CITATIONS"]   = "1" if use_citations else "0"
    env["SKIP_PREPROCESS"] = "1"   # default: non rifare il preprocessing
    env["SKIP_W2V"]        = "0" if selected_emb == 'w2v' else "1"
    return env


def save_excel(data, filename):
    if not data:
        return
    df = pd.DataFrame(data)

    priority_cols = [
        "Experiment_ID", "Experiment_Name", "Status", "GNN_Active", "Emb_Type",
        "Mode", "Composite_Score",
        "Pairwise_F1", "K_Metric_K", "B3_F1", "Cluster_F1",
        "Splitting_Error", "Lumping_Error",
        "Pairwise_Prec", "Pairwise_Rec",
        "B3_Prec", "B3_Rec",
        "Full_Command",
        "Duration_sec",
    ]
    other_cols  = [c for c in df.columns if c not in priority_cols]
    final_order = [c for c in priority_cols if c in df.columns] + other_cols
    df = df[final_order]

    try:
        df.to_excel(filename, index=False)
    except PermissionError:
        new_name = str(filename).replace(".xlsx", f"_backup_{int(time.time())}.xlsx")
        df.to_excel(new_name, index=False)
        print(f"   ⚠️  Excel aperto, salvato in: {new_name}")


# ==============================================================================
# 🧪 LISTA ESPERIMENTI
# ==============================================================================
# Ogni esperimento dichiara SOLO i parametri che devia dai default di params.py.
# I campi obbligatori sono:
#   Experiment_ID : stringa identificativa
#   suffix        : aggiunto al nome file Excel
#   mode          : 'train' | 'valid' | 'test'
#   no_gnn        : True | False
#   params        : dict con i parametri argparse da sovrascrivere
# ==============================================================================

EXPERIMENTS = [


    # ---------------------------------------------------------------------------
    # Baseline no-GNN — TRAIN
    # ---------------------------------------------------------------------------
        {
        "Experiment_ID": "baseline_TRAIN",
        "suffix":        "NOCIT_NOGNN_TRAIN",
        "mode":          "train",
        "no_gnn":        True,
        "params": {
            "db_eps":         0.35,
            "db_min":         2,
        
        },
    },
      # Baseline no-GNN — VALID
    # ---------------------------------------------------------------------------
    {
        "Experiment_ID": "baseline_VALID",
        "suffix":        "NOCIT_NOGNN_VALID",
        "mode":          "valid",
        "no_gnn":        True,
        "params": {
            "db_eps": 0.35,
            "db_min": 2,
        },
    },

    # ---------------------------------------------------------------------------
    
    # ---------------------------------------------------------------------------
    # Aggiungi altri esperimenti qui sotto, decommentando o copiando il blocco.
    # Esempi commentati:
    # ---------------------------------------------------------------------------

    # --- Baseline con citazioni ---
    # {
    #     "Experiment_ID": "cit_VALID",
    #     "suffix":        "CIT_NOGNN_VALID",
    #     "mode":          "valid",
    #     "no_gnn":        True,
    #     "params": {
    #         "db_eps":       0.30,
    #         "db_min":       2,
    #         "use_citations": True,
    #     },
    # },

    # --- GNN con Specter ---
    # {
    #     "Experiment_ID": "gnn_VALID",
    #     "suffix":        "CIT_GNN_VALID",
    #     "mode":          "valid",
    #     "no_gnn":        False,
    #     "params": {
    #         "epochs":       50,
    #         "lr":           5e-5,
    #         "cluster_w":    0.1,
    #         "db_eps":       0.20,
    #         "db_min":       2,
    #         "use_citations": True,
    #     },
    # },

]


# ==============================================================================
# 🚀 MAIN RUNNER
# ==============================================================================

def run_session():
    # 1. Otteniamo la configurazione (WhoIsWho/OC e Specter/SBERT) dall'utente
    selected_ds, selected_emb = get_session_config()

    # 2. Definiamo la dimensione nascosta in base all'embedding
    # Sia Specter2 che SBERT (all-mpnet-base-v2) usano 768
    hidden_dim = [768, 512] 
    
    # 3. Parametri fissi per tutta la sessione di test
    common_params = {
        "dataset_type": selected_ds,  # Passa 'whoiswho' o 'oc'
        "emb_type":     selected_emb, # Passa 'specter' o 'sbert'
        "hidden_dim":   hidden_dim,
        "save_path":    SAVE_PATH,
        "seed":         0,            # Garantisce riproducibilità
    }

    # 4. Preparazione cartelle e file di report
    output_dir    = Path("out/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Il nome del file Excel ora riflette sia il dataset che l'embedding
    excel_filename = output_dir / f"results_{selected_ds.upper()}_{selected_emb.upper()}_{timestamp}.xlsx"
    print(f"\n📂 Report Excel: {excel_filename}")

    python_exe   = sys.executable
    results_list = []

    # 5. Loop degli esperimenti definiti nella lista EXPERIMENTS
    for idx, exp in enumerate(EXPERIMENTS):
        exp_id   = exp["Experiment_ID"]
        no_gnn   = exp["no_gnn"]
        use_cit  = exp["params"].get("use_citations", False)
        
        # Nome identificativo dell'esperimento per i log
        exp_name = f"{selected_ds.upper()}_{selected_emb.upper()}_{exp['suffix']}"

        print("\n" + "="*80)
        print(f"[{exp_id}] 🧪 ESPERIMENTO {idx+1}/{len(EXPERIMENTS)}: {exp_name}")
        print("="*80)

        # Costruzione del comando subprocess
        cmd = build_cmd(python_exe, PIPELINE_SCRIPT,
                        mode=exp["mode"],
                        no_gnn=no_gnn,
                        common_params=common_params,
                        exp_params=exp["params"])

        # Costruzione dell'ambiente (env vars)
        env = build_env(use_citations=use_cit,
                        no_gnn=no_gnn,
                        selected_emb=selected_emb)

        print(f"📝 dataset={selected_ds} | mode={exp['mode']} | emb={selected_emb} | GNN={'NO' if no_gnn else 'YES'}")
        print(f"🔧 cmd: {' '.join(cmd)}")

        # Inizializzazione metriche vuote
        metrics = {k: None for k in [
            "Composite_Score", "Pairwise_F1", "Pairwise_Prec", "Pairwise_Rec",
            "K_Metric_K", "B3_F1", "B3_Prec", "B3_Rec", "Splitting_Error", "Lumping_Error"
        ]}

        start_time = time.time()
        status     = "UNKNOWN"

        try:
            # Esecuzione della PipelineA.py
            process  = subprocess.run(cmd, env=env, text=True)
            duration = time.time() - start_time

            if process.returncode != 0:
                print(f"❌ Esperimento fallito (returncode={process.returncode})")
                status = "FAILED"
            else:
                status = "SUCCESS"
                # Lettura dei risultati dal JSON generato dalla pipeline
                json_path = Path("evaluation_results/evaluation_results.json")

                if json_path.exists():
                    try:
                        with open(json_path, 'r') as f:
                            d = json.load(f)
                        
                        pw = d.get("pairwise",   {})
                        km = d.get("k_metric",   {})
                        b3 = d.get("b3",         {})
                        sl = d.get("structural", {})  


                        metrics = {
                            "Composite_Score":  d.get("composite_score"),
                            "Pairwise_F1":      pw.get("f1"),
                            "Pairwise_Prec":    pw.get("precision"),
                            "Pairwise_Rec":     pw.get("recall"),
                            "K_Metric_K":       km.get("k"),
                            "B3_F1":            b3.get("f1"),
                            "B3_Prec":          b3.get("precision"),
                            "B3_Rec":           b3.get("recall"),  
                            "Splitting_Error":  sl.get("splitting_error"),
                            "Lumping_Error":    sl.get("lumping_error"),
                        }
                    except Exception as e:
                        print(f"⚠️ Errore lettura JSON: {e}")
                        status = "JSON_ERROR"
                else:
                    print("⚠️ File risultati JSON non trovato.")
                    status = "NO_RESULTS"

        except Exception as e:
            print(f"❌ Exception critica: {e}")
            status   = "CRASHED"
            duration = time.time() - start_time

        # 6. Creazione riga per il report Excel
        row = {
            "Experiment_ID":   exp_id,
            "Experiment_Name": exp_name,
            "Status":          status,
            "Dataset":         selected_ds,
            "Emb_Type":        selected_emb,
            "Date":            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Duration_sec":    round(duration, 2),
            "Mode":            exp["mode"],
            "GNN_Active":      "NO" if no_gnn else "YES",
            "Use_Citations":   use_cit,
            "Full_Command":    " ".join(cmd),
        }
        row.update(metrics)
        results_list.append(row)
        
        # Salvataggio progressivo (utile se lo script si interrompe)
        save_excel(results_list, excel_filename)

    print("\n" + "="*80)
    print(f"✅ SESSIONE COMPLETATA ({selected_ds.upper()} - {selected_emb.upper()})")
    print(f"📊 Report finale: {excel_filename}")
    print("="*80)

if __name__ == "__main__":
    # Verifica che le librerie necessarie siano installate
    try:
        import pandas
        import openpyxl
    except ImportError:
        print("❌ Mancano librerie necessarie: pip install pandas openpyxl")
        sys.exit(1)

    # Verifica che la lista degli esperimenti non sia vuota
    if not EXPERIMENTS:
        print("❌ Errore: La lista EXPERIMENTS è vuota.")
        sys.exit(1)

    # AVVIA LA SESSIONE
    run_session()
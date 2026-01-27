import sys
import os
import torch
# import types # <--- Non serve più con questo metodo

# 1. DISABILITA IL RE-IMPORT AUTOMATICO
os.environ["DEBUG_MODE"] = "1"

import params
import bond.dataset.preprocess_SND as snd
import bond.dataset.dump_graph as dg
import bond.dataset.load_data as ld
import training.autotrain_bond as ab # Il modulo del trainer

def run_mini_debug_bypass_gnn_v2(num_authors=20):
    args = params.set_params()
    mode = args.mode
    
    print(f"🚀 Avvio Debug Pipeline (GNN BYPASSED v2 - Tuple Return) su {num_authors} autori")
    print("---------------------------------------------------------")
    print("⚠️ ATTENZIONE: IN QUESTO TEST LA GNN È DISABILITATA.")
    print("Il metodo forward della classe ATTGNN verrà sostituito da un'identità che restituisce (x, x).")
    print("---------------------------------------------------------\n")

    # --- FUNZIONE DI FILTRAGGIO ---
    def get_subset(full_dict):
        return {k: full_dict[k] for i, k in enumerate(full_dict) if i < num_authors}

    # --- SURGICAL PATCHING PER IL DATASET ---
    original_read_raw = snd.read_raw_pubs
    snd.read_raw_pubs = lambda m: get_subset(original_read_raw(m))
    dg.load_json = lambda path: get_subset(ld.load_json(path)) if any(x in path for x in ["train_author", "sna_valid", "sna_test"]) else ld.load_json(path)

    def patched_load_dataset(m):
        full_names, full_pubs = ld.load_dataset(m)
        subset_names = full_names[:num_authors]
        subset_pubs = {name: full_pubs[name] for name in subset_names}
        print(f"  [DEBUG] Training limitato a: {subset_names}")
        return subset_names, subset_pubs

    ab.load_dataset = patched_load_dataset

    # --- ESECUZIONE ---
    try:
        # Se i grafi sono già pronti con i parametri giusti (es. db_eps=0.02), commenta questi:
        # print("\n--- STEP 1 & 3: Preprocessing Ridotto ---")
        # snd.dump_name_pubs()
        # snd.dump_features_relations_to_file()
        # dg.build_graph()

        print("\n--- STEP 4: Training Ridotto (GNN BYPASS v2) ---")

        # ============================================================
        # ⚡⚡⚡ INIZIO PATCH BYPASS GNN (APPROCCIO DI CLASSE v2) ⚡⚡⚡
        # ============================================================
        print("\n[PATCH] Applicazione bypass alla CLASSE del modello GNN...")

        if not hasattr(ab, 'ATTGNN'):
             print("❌ Errore: Impossibile trovare la classe ATTGNN in autotrain_bond.")
             return

        ModelClass = ab.ATTGNN

        # --- LA CORREZIONE È QUI ---
        # Definiamo la nuova funzione "identità" che restituisce una TUPLA.
        # Restituiamo (x, x) per soddisfare la richiesta di "logits, embd".
        # Il secondo 'x' sarà quello usato per il clustering.
        def identity_forward_tuple(self, x, *args, **kwargs):
            # print(" -> [GNN BYPASS] Forward chiamato: restituisco tupla (x, x).") # Verbose
            return x, x

        # SOVRASCRIVI il metodo forward della CLASSE
        ModelClass.forward = identity_forward_tuple

        print(f"[PATCH] Metodo forward della classe {ModelClass.__name__} sostituito con identità (tupla).\n")
        # ============================================================
        # ⚡⚡⚡ FINE PATCH BYPASS GNN ⚡⚡⚡
        # ============================================================
        
        # Inizializza e lancia il trainer
        trainer = ab.BONDTrainer()
        trainer.fit(datatype=mode)

    except Exception as e:
        print(f"❌ Errore durante il debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Assicurati che params.py abbia i settaggi di clustering che vuoi testare (es. db_eps=0.02, db_min=2)
    run_mini_debug_bypass_gnn_v2(num_authors=20)
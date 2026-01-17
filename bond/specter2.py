import os
import re
import torch
import numpy as np
import pickle
from tqdm import tqdm
from os.path import join
import sys
from transformers import AutoTokenizer, AutoModel

# Fix per i percorsi
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'bond'))

from bond.params import set_params
from bond.dataset.load_data import load_json, dump_data
from bond.dataset.save_results import check_mkdir
from bond.dataset.preprocess_SND import read_raw_pubs

# ==========================================
# CONFIGURAZIONE SPECTER2
# ==========================================
SPECTER_MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 1  # Riduci a 8 o 4 se vai in Out Of Memory
MAX_LENGTH = 256
FT_DIM = 768     # SPECTER2 ha dimensione 768 (W2V aveva 256)

args = set_params()
device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

def get_model():
    print(f"Loading SPECTER2 on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(SPECTER_MODEL_NAME)
    model = AutoModel.from_pretrained(SPECTER_MODEL_NAME).to(device)
    model.eval()
    return tokenizer, model

def dump_paper_emb_specter():
    tokenizer, model = get_model()
    
    # Modes da processare
    modes = ['train', 'valid', 'test']
    
    for mode in modes:
        try:
            raw_pubs = read_raw_pubs(mode)
        except Exception as e:
            print(f"Skip mode {mode}: {e}")
            continue
            
        print(f"\nProcessing embeddings for mode: {mode}")
        for name in tqdm(raw_pubs):
            name_pubs_path = join(args.save_path, 'names_pub', mode, name + '.json')
            if not os.path.exists(name_pubs_path): continue
            
            name_pubs = load_json(name_pubs_path)
            text_feature_path = join(args.save_path, 'paper_emb', mode, name)
            check_mkdir(text_feature_path)

            pids = list(name_pubs.keys())
            ptext_emb = {}
            tcp = set()
            
            # Prepariamo i testi (Titolo + Keywords + Org/Abstract)
            # SPECTER lavora meglio con un separatore tra titolo e resto
            batch_texts = []
            for pid in pids:
                pub = name_pubs[pid]
                title = pub.get('title', '').strip()
                
                # Cerchiamo l'abstract, se manca usiamo keywords + org come fallback
                abstract = pub.get('abstract', '')
                if not abstract or len(abstract) < 10:
                    kw = " ".join(pub.get('keywords', []))
                    org = ""
                    if pub.get('authors'):
                        org = pub['authors'][0].get('org', '')
                    abstract = f"{kw} {org}".strip()
                
                # Formato SPECTER: Title + SEP + Abstract
                full_text = f"{title} {tokenizer.sep_token} {abstract}"
                batch_texts.append(full_text)

            # Generazione embeddings in batch
            all_embeddings = []
            for i in range(0, len(batch_texts), BATCH_SIZE):
                batch = batch_texts[i:i + BATCH_SIZE]
                
                inputs = tokenizer(batch, padding=True, truncation=True, 
                                   return_tensors="pt", max_length=MAX_LENGTH).to(device)
                
                
                # --- All'interno del ciclo di generazione embeddings nello script specter2.py ---

                with torch.no_grad():
                    output = model(**inputs)
                    
                    # 1. ESTRAZIONE
                    raw_embeddings = output.last_hidden_state[:, 0, :] 
                    
                    # 2. NORMALIZZAZIONE
                    normalized_embeddings = torch.nn.functional.normalize(raw_embeddings, p=2, dim=1)
                    
                    # 3. IL FIX CRUCIALE: .cpu().numpy()
                    # .cpu() -> Sposta il dato dalla VRAM alla RAM di sistema (fondamentale!)
                    # .numpy() -> Trasforma il tensore in un array NumPy, recidendo ogni legame con PyTorch
                    # .copy() -> Forza una copia pulita dei dati per evitare riferimenti residui
                    all_embeddings.extend(normalized_embeddings.cpu().numpy().copy())

                # Svuota la cache dopo ogni batch per essere sicuri
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            # Mappatura risultati
            for idx, pid in enumerate(pids):
                emb = all_embeddings[idx]
                if np.all(emb == 0) or np.isnan(emb).any():
                    tcp.add(idx)
                ptext_emb[pid] = emb

            # Salvataggio identico al formato originale
            dump_data(ptext_emb, join(text_feature_path, 'ptext_emb.pkl'))
            dump_data(tcp, join(text_feature_path, 'tcp.pkl'))

if __name__ == "__main__":
    # Nota: non servono più dump_corpus() o train_w2v_model()
    # perché SPECTER è pre-addestrato.
    dump_paper_emb_specter()
    print(f"\n✅ Embeddings SPECTER2 (dim={FT_DIM}) generati con successo.")
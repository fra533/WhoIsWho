import argparse
import sys

def set_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3.615749513383116e-05)

    parser.add_argument('--save_path', type=str, default='dataset/data')
    parser.add_argument('--l2_coef', type=float, default=2.714679239042756e-05)
    parser.add_argument('--compress_ratio', type=float, default=1)
    parser.add_argument('--hidden_dim', nargs='+', type=int, default=[256, 512])
    parser.add_argument('--rel_on', type=str, default='aov')
    
    parser.add_argument('--cluster_w', type=float, default=0.4)
    parser.add_argument('--prob_v', type=float, default=0.9)
    parser.add_argument('--coa_th', type=int, default=0)
    parser.add_argument('--coo_th', type=float, default=0.85)  # ← cambiato a float
    parser.add_argument('--cov_th', type=float, default=2)
    parser.add_argument('--coc_th', type=float, default=0.0)  # ← cambiato a float
    parser.add_argument('--coi_th', type=float, default=0.0)  # ← cambiato a float

    parser.add_argument('--db_eps', type=float, default=0.1)
    parser.add_argument('--db_min', type=int, default=3)
    parser.add_argument('--post_match', type=bool, default=False)

    # ✅ CORREZIONE: usa nargs='+' invece di type=list
    parser.add_argument('--th_a', nargs='+', type=float, default=[0, 1])
    parser.add_argument('--th_o', nargs='+', type=float, default=[0.6, 0.5])
    parser.add_argument('--th_v', nargs='+', type=float, default=[1, 2])
    parser.add_argument('--th_c', nargs='+', type=float, default=[0, 0])  # ← cambiato default
    parser.add_argument('--th_i', nargs='+', type=float, default=[0, 0])  # ← cambiato default

    parser.add_argument('--repeat_num', type=int, default=1)
    
    # ✅ AGGIUNTO: citation weights
    parser.add_argument('--cite_out_weight', type=float, default=1.7)
    parser.add_argument('--cite_in_weight', type=float, default=0.6)
    parser.add_argument('--use_citations', type=bool, default=True)
    
    args, _ = parser.parse_known_args()
    
    # Backward compatibility
    if not args.use_citations:
        original_rel_on = args.rel_on
        args.rel_on = args.rel_on.replace('c', '').replace('i', '')
        if original_rel_on != args.rel_on:
            print(f"⚠️  Citations disabled: rel_on adjusted from '{original_rel_on}' to '{args.rel_on}'")
    
    return args
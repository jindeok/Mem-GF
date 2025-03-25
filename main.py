import random
import numpy as np
import torch
import scipy.sparse as sp
import os
import argparse
import psutil  # For memory usage tracking
from utils import *
from models import *
from tqdm import tqdm
import time

random.seed(2022)
np.random.seed(2022)
torch.manual_seed(2022)

current_directory = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="gpu")
parser.add_argument("--dataset", type=str, default="ml-10m")
parser.add_argument("--model", type=str, default="mem-gf")
parser.add_argument("--verbose", type=int, default=1)
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--power", type=float, default=1)
parser.add_argument("--filter", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--eval_mode", type=bool, default=True)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.device == "gpu":
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    if args.verbose:
        print(f"Device: {device}")

    dataset = args.dataset
    path_tr = f"{current_directory}/dataset/{dataset}_train.npz"
    path_ts = f"{current_directory}/dataset/{dataset}_test.npz"
    R_tr_sparse = sp.load_npz(path_tr)
    R_ts_sparse = sp.load_npz(path_ts)

    n_users = R_tr_sparse.shape[0]
    n_items = R_tr_sparse.shape[1]
    if args.verbose:
        print(f"number of users: {n_users}")
        print(f"number of items: {n_items}")

    n_inters = R_tr_sparse.nnz + R_ts_sparse.nnz
    if args.verbose:
        print(f"number of interactions: {n_inters}")

    start_time = time.time()

    # 1) Normalize R without converting to dense
    R_norm_sparse, D_i, D_i_inv = normalize_sparse_adjacency_matrix_sparse(R_tr_sparse, args.alpha)
    #R_norm_sparse = R_norm_sparse.astype(np.float32)

    # 2) Convert to PyTorch sparse tensors
    R_norm = csr2torch(R_norm_sparse, device)
    R_tr = csr2torch(R_tr_sparse, device)
    R_ts = csr2torch(R_ts_sparse, device)

    # 4) Prepare polynomial coefficients a_k for poly.filters (not directly used by all models)
    a_k = get_polynomial_coeffs(args.filter)

    # 5) Batch processing config
    batch_size = args.batch_size
    num_batches = (n_users + batch_size - 1) // batch_size

    # For eval
    total_recall_10, total_recall_20 = 0.0, 0.0
    total_ndcg_10, total_ndcg_20 = 0.0, 0.0
    total_users = 0

    # Convert R_ts to CSR format for slicing
    R_ts_csr = R_ts_sparse.tocsr()
    R_tr_csr = R_tr_sparse.tocsr()

    # 6) Initialize model
    if args.model == 'gf-cf':
        model = GF_CF(args, R_norm, D_i, D_i_inv, device)
    elif args.model == 'turbo-cf':
        model = Turbo_CF(args, R_norm, args.filter, device)
    elif args.model == 'mem-gf':
        model = Mem_GF(args, R_norm, args.filter, a_k, device)
    elif args.model == 'pgsp':
        model = PGSP(args, R_norm, R_tr, device)
        # For personalized signal in PGSP
        P_U = torch.sparse.mm(R_norm, R_norm.transpose(0, 1))
    elif args.model == 'lgcn-ide':
        model = LGCN_IDE(args, R_norm, args.filter, device)
    elif args.model == 'ease':
        model = EASE(args, R_norm, device)
    elif args.model == 'svd-ae':
        model = SVD_AE(args, R_norm, R_tr, device)
    elif args.model == 'higsp':
        # Build item-item graph R_tilde from R_norm: (n_items x n_items) = R_norm^T * R_norm
        R_tilde = torch.sparse.mm(R_norm.transpose(0, 1), R_norm)
        # Assume you have a HiGSP class that takes these arguments
        model = HIGSP(args, R_norm, R_tilde, device)
    else:
        raise ValueError("Unknown model type")

    # 7) Batch loop for inference + evaluation
    for batch_idx in tqdm(range(num_batches), desc="Processing Batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(n_users, (batch_idx + 1) * batch_size)
        batch_users = end_idx - start_idx

        # Prepare user batch data
        if args.model == 'lgcn-ide' or args.model == 'svd-ae':
            r_u_sparse = R_norm_sparse[start_idx:end_idx]
        elif args.model == 'pgsp':
            # Construct personalized signal p_u
            p_u = P_U.to_dense()[start_idx:end_idx]
            r_u_sparse = R_tr_csr[start_idx:end_idx]
        elif args.model == 'higsp':
            r_u_sparse = R_tr_csr[start_idx:end_idx]
            user_indices = torch.arange(start_idx, end_idx, device=device, dtype=torch.long)
        else:
            r_u_sparse = R_tr_csr[start_idx:end_idx]

        r_u_coo = r_u_sparse.tocoo()
        indices = np.vstack((r_u_coo.row, r_u_coo.col))
        values = r_u_coo.data
        size = (batch_users, n_items)

        r_u = torch.sparse_coo_tensor(
            indices,
            values,
            size=size,
            device=device,
            dtype=torch.float32
        ).to_dense()

        # Model-specific forward
        if args.model == 'pgsp':
            s_u = model.predict_batch(p_u, r_u)
        elif args.model == 'higsp':
            s_u = model.predict_batch(r_u, user_indices)
        else:
            s_u = model.predict_batch(r_u)

        # Evaluate for the current batch
        if args.eval_mode:
            total_recall_10, total_ndcg_10, total_recall_20, total_ndcg_20, total_users = evaluate(
                s_u,
                R_tr_csr,
                R_ts_csr,
                start_idx,
                end_idx,
                batch_users,
                total_recall_10,
                total_ndcg_10,
                total_recall_20,
                total_ndcg_20,
                total_users
            )

    print(f"Total computing time: {time.time() - start_time:.2f} sec.")

    # 8) Compute final average metrics
    if args.eval_mode:
        avg_recall_10 = total_recall_10 / total_users
        avg_ndcg_10   = total_ndcg_10 / total_users
        avg_recall_20 = total_recall_20 / total_users
        avg_ndcg_20   = total_ndcg_20 / total_users

        print(f"Recall@10: {avg_recall_10:.4f}")
        print(f"NDCG@10:   {avg_ndcg_10:.4f}")
        print(f"Recall@20: {avg_recall_20:.4f}")
        print(f"NDCG@20:   {avg_ndcg_20:.4f}")

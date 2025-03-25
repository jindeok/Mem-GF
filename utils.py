import numpy as np
import torch
from scipy.stats import rankdata
import copy
from scipy.linalg import expm
import scipy.sparse as sp

def evaluate(s_u, R_tr_csr, R_ts_csr, start_idx, end_idx, batch_users,
             total_recall_10, total_ndcg_10, total_recall_20, total_ndcg_20, total_users):
    # Move to CPU
    s_u = s_u.cpu()

    # 5.5) Mask the training items
    R_tr_batch = R_tr_csr[start_idx:end_idx].toarray()  # np.ndarray
    s_u += (-99999) * torch.from_numpy(R_tr_batch)

    # 5.6) Evaluate metrics
    gt_batch = R_ts_csr[start_idx:end_idx].toarray()
    pred_batch = s_u.numpy()

    # -- Recall/NDCG for k=10 --
    recall_batch_10 = recall_at_k_batch(gt_batch, pred_batch, k=10)
    ndcg_batch_10 = ndcg_at_k_batch(gt_batch, pred_batch, k=10)

    # -- Recall/NDCG for k=20 --
    recall_batch_20 = recall_at_k_batch(gt_batch, pred_batch, k=20)
    ndcg_batch_20 = ndcg_at_k_batch(gt_batch, pred_batch, k=20)

    total_recall_10 += np.sum(recall_batch_10)
    total_ndcg_10 += np.sum(ndcg_batch_10)
    total_recall_20 += np.sum(recall_batch_20)
    total_ndcg_20 += np.sum(ndcg_batch_20)

    total_users += batch_users

    return total_recall_10, total_ndcg_10, total_recall_20, total_ndcg_20, total_users

def lanczos_algorithm(R_norm, q1, K, device):
    """
    Perform the Lanczos algorithm to compute the tridiagonal matrix T and the orthonormal basis Q.
    """
    q1 = q1.float()
    batch_size = q1.shape[0]
    n_items = q1.shape[1]
    Q = []
    alpha = []
    beta = []

    q_prev = torch.zeros_like(q1, device=device).float()
    q = q1 / torch.norm(q1, dim=1, keepdim=True)
    
    q = q.float()
    Q.append(q)
    beta_prev = torch.zeros((batch_size, 1), device=device).float()

    for j in range(K):
        # Compute w = P q_j = R_norm^T (R_norm q_j)
        Rn_q = torch.sparse.mm(R_norm, q.t()).t()  # [batch_size, n_users]
        w = torch.sparse.mm(R_norm.t(), Rn_q.t()).t()  # [batch_size, n_items]

        w = w - beta_prev * q_prev
        alpha_j = torch.sum(q * w, dim=1, keepdim=True)
        w = w - alpha_j * q
        beta_j = torch.norm(w, dim=1, keepdim=True)
        # Avoid division by zero
        beta_j[beta_j == 0] = 1e-10

        q_next = w / beta_j

        Q.append(q_next)
        alpha.append(alpha_j)
        beta.append(beta_j)

        q_prev = q
        q = q_next
        beta_prev = beta_j

    # Stack Q, alpha, beta
    Q = torch.stack(Q, dim=0)  # Shape: [K+1, batch_size, n_items]
    alpha = torch.stack(alpha, dim=0)  # Shape: [K, batch_size, 1]
    beta = torch.stack(beta, dim=0)  # Shape: [K, batch_size, 1]

    return Q, alpha, beta

def get_polynomial_coeffs(filter_choice):
    if filter_choice == 1:
        return [1.0]
    elif filter_choice == 2:
        return [2.0, -1.0]
    elif filter_choice == 3:
        return [-29.33, 10.03, -1.07]
    elif filter_choice == 4:
        tau = 0.5
        return [1, -tau, tau**2 / 2, -tau**3 / 6, tau**4 / 24]
    elif filter_choice == 5:
        return [7.974865109143177, 3.1018304070759397, -11.201654823809385, 6.208991135386687, -1.06302647725054] 
    else:
        raise ValueError("Invalid filter choice")

def compute_T(alpha, beta, device):
    """
    Construct the tridiagonal matrix T from alpha and beta.
    """
    batch_size = alpha.shape[1]
    K = alpha.shape[0]

    T = torch.zeros((batch_size, K, K), device=device)
    for i in range(K):
        T[:, i, i] = alpha[i, :, 0]
        if i < K - 1:
            T[:, i, i + 1] = beta[i + 1, :, 0].squeeze()
            T[:, i + 1, i] = beta[i + 1, :, 0].squeeze()
    return T



def polynomial_filter(T, a_k, device):
    """
    Apply the polynomial filter to the tridiagonal matrix T.
    """
    f_T = torch.zeros_like(T, device=device)
    T_power = T.clone()
    for idx, a in enumerate(a_k):
        if idx == 0:
            f_T += a * T_power
        else:
            T_power = torch.bmm(T_power, T)
            f_T += a * T_power
    return f_T



def recall_at_k(gt_mat, results, k=10):
    recall_sum = 0
    for i in range(gt_mat.shape[0]):
        relevant_items = set(np.where(gt_mat[i, :] > 0)[0])
        top_predicted_items = np.argsort(-results[i, :])[:k]
        num_relevant_items = len(relevant_items.intersection(top_predicted_items))
        recall_sum += num_relevant_items / len(relevant_items)
    recall = recall_sum / gt_mat.shape[0]
    return recall


def ndcg_at_k(gt_mat, results, k=10):
    ndcg_sum = 0
    for i in range(gt_mat.shape[0]):
        relevant_items = set(np.where(gt_mat[i, :] > 0)[0])
        top_predicted_items = np.argsort(-results[i, :])[:k]
        dcg = 0
        idcg = 0
        for j in range(k):
            if top_predicted_items[j] in relevant_items:
                dcg += 1 / np.log2(j + 2)
            if j < len(relevant_items):
                idcg += 1 / np.log2(j + 2)
        ndcg_sum += dcg / idcg if idcg > 0 else 0
    ndcg = ndcg_sum / gt_mat.shape[0]
    return ndcg


def calculate_row_correlations(matrix1, matrix2):
    base_value = 1  

    num_rows = matrix1.shape[0]
    correlations = torch.zeros(num_rows)

    for row in range(num_rows):
        nz_indices1 = matrix1.indices[matrix1.indptr[row] : matrix1.indptr[row + 1]]
        nz_indices2 = matrix2.indices[matrix2.indptr[row] : matrix2.indptr[row + 1]]

        common_indices = torch.intersect1d(nz_indices1, nz_indices2)

        nz_values1 = matrix1.data[matrix1.indptr[row] : matrix1.indptr[row + 1]][
            torch.searchsorted(nz_indices1, common_indices)
        ]
        nz_values2 = matrix2.data[matrix2.indptr[row] : matrix2.indptr[row + 1]][
            torch.searchsorted(nz_indices2, common_indices)
        ]

        if len(common_indices) > 0:
            correlation = torch.corrcoef(nz_values1, nz_values2)[0, 1]
            correlations[row] = correlation + base_value

    return correlations




# def csr2torch(csr_matrix):
#     # Convert CSR matrix to COO format (Coordinate List)
#     coo_matrix = csr_matrix.tocoo()

#     # Create a PyTorch tensor for data, row indices, and column indices
#     data = torch.FloatTensor(coo_matrix.data)
#     # indices = torch.LongTensor([coo_matrix.row, coo_matrix.col])
#     # -> This results in: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
#     indices = torch.LongTensor(np.vstack((coo_matrix.row, coo_matrix.col)))

#     # Create a sparse tensor using torch.sparse
#     # return torch.sparse.FloatTensor(indices, data, torch.Size(coo_matrix.shape))
#     # -> This results in: UserWarning: torch.sparse.SparseTensor(indices, values, shape, *, device=) is deprecated.  Please use torch.sparse_coo_tensor(indices, values, shape, dtype=, device=).
#     return torch.sparse_coo_tensor(indices, data, torch.Size(coo_matrix.shape))

def csr2torch(sparse_mx, device):
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    ).to(device)
    values = torch.from_numpy(sparse_mx.data.astype(np.float32)).to(device)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized = (tensor - min_val) / (max_val - min_val)
    return normalized



def normalize_sparse_adjacency_matrix(adj_matrix, alpha):
    # Calculate rowsum and columnsum using COO format
    rowsum = torch.sparse.mm(
        adj_matrix, torch.ones((adj_matrix.shape[1], 1), device=adj_matrix.device)
    ).squeeze()
    rowsum = torch.pow(rowsum, -alpha)
    colsum = torch.sparse.mm(
        adj_matrix.t(), torch.ones((adj_matrix.shape[0], 1), device=adj_matrix.device)
    ).squeeze()
    colsum = torch.pow(colsum, alpha - 1)
    indices = (
        torch.arange(0, rowsum.size(0)).unsqueeze(1).repeat(1, 2).to(adj_matrix.device)
    )
    # d_mat_rows = torch.sparse.FloatTensor(
    #     indices.t(), rowsum, torch.Size([rowsum.size(0), rowsum.size(0)])
    # ).to(device=adj_matrix.device)
    # -> This results in:  UserWarning: torch.sparse.SparseTensor(indices, values, shape, *, device=) is deprecated.  Please use torch.sparse_coo_tensor(indices, values, shape, dtype=, device=).
    d_mat_rows = torch.sparse_coo_tensor(
        indices.t(), rowsum, torch.Size([rowsum.size(0), rowsum.size(0)])
    ).to(device=adj_matrix.device)
    indices = (
        torch.arange(0, colsum.size(0)).unsqueeze(1).repeat(1, 2).to(adj_matrix.device)
    )
    # d_mat_cols = torch.sparse.FloatTensor(
    # indices.t(), colsum, torch.Size([colsum.size(0), colsum.size(0)])
    # ).to(device=adj_matrix.device)
    # -> This results in: UserWarning: torch.sparse.SparseTensor(indices, values, shape, *, device=) is deprecated.  Please use torch.sparse_coo_tensor(indices, values, shape, dtype=, device=).
    d_mat_cols = torch.sparse_coo_tensor(
        indices.t(), colsum, torch.Size([colsum.size(0), colsum.size(0)])
    ).to(device=adj_matrix.device)

    # Calculate d_inv for rows and columns
    # d_inv_rows = torch.pow(rowsum, -alpha)
    # d_inv_rows[d_inv_rows == float('inf')] = 0.
    # d_mat_rows = torch.diag(d_inv_rows)

    # d_inv_cols = torch.pow(colsum, alpha - 1)
    # d_inv_cols[d_inv_cols == float('inf')] = 0.
    # d_mat_cols = torch.diag(d_inv_cols)

    # Normalize adjacency matrix
    norm_adj = d_mat_rows.mm(adj_matrix).mm(d_mat_cols)

    return norm_adj



def freq_filter(s_values, mode=1, alpha=0.9, start=0):
    """
    input:
    - s_values: singular (eigen) values, list form

    output:
    - filterd_s_values
    """
    if mode == 0:
        filtered_s_values = s_values
    elif mode == 1:
        filtered_s_values = [(lambda x: 1 / (1 - alpha * x))(v) for v in s_values]
    elif mode == 2:
        filtered_s_values = [(lambda x: 1 / (alpha * x))(v) for v in s_values]
    elif mode == 3:
        filtered_s_values = [(lambda x: 1.5**x)(v) for v in s_values]
    elif mode == 3:
        filtered_s_values = [(lambda x: 1.5**x)(v) for v in s_values]
    elif mode == "band_pass":
        end = start + 5
        filtered_s_values = (
            [0] * int(start) + [1] * int(end - start) + [0] * int(len(s_values) - end)
        )

    return np.diag(filtered_s_values)


def get_norm_adj(alpha, adj_mat):
    # Calculate rowsum and columnsum using PyTorch operations
    rowsum = torch.sum(adj_mat, dim=1)
    colsum = torch.sum(adj_mat, dim=0)

    # Calculate d_inv for rows and columns
    d_inv_rows = torch.pow(rowsum, -alpha).flatten()
    d_inv_rows[torch.isinf(d_inv_rows)] = 0.0
    d_mat_rows = torch.diag(d_inv_rows)

    d_inv_cols = torch.pow(colsum, alpha - 1).flatten()
    d_inv_cols[torch.isinf(d_inv_cols)] = 0.0
    d_mat_cols = torch.diag(d_inv_cols)
    d_mat_i_inv_cols = torch.diag(1 / d_inv_cols)

    # Normalize adjacency matrix
    norm_adj = adj_mat.mm(d_mat_rows).mm(adj_mat).mm(d_mat_cols)
    norm_adj = norm_adj.to_sparse()  # Convert to sparse tensor

    # Convert d_mat_rows, d_mat_i_inv_cols to sparse tensors
    d_mat_rows_sparse = d_mat_rows.to_sparse()
    d_mat_i_inv_cols_sparse = d_mat_i_inv_cols.to_sparse()

    return norm_adj

def torch_sparse_to_scipy_csr(tensor: torch.Tensor):
    coalesced = tensor.coalesce()
    indices = coalesced.indices().cpu().numpy()
    values = coalesced.values().cpu().numpy()
    shape = coalesced.shape
    return sp.coo_matrix((values, (indices[0], indices[1])), shape=shape).tocsr()

def top_k(S, k=1, device="cpu"):
    """
    S: scores, numpy array of shape (M, N) where M is the number of source nodes,
        N is the number of target nodes
    k: number of predicted elements to return
    """
    if device == "cpu":
        top = torch.argsort(-S)[:, :k]
        result = torch.zeros(S.shape)
        for idx, target_elms in enumerate(top):
            for elm in target_elms:
                result[idx, elm] = 1
    else:
        top = torch.argsort(-S)[:, :k]
        result = torch.zeros(S.shape, device=device)
        for idx, target_elms in enumerate(top):
            for elm in target_elms:
                result[idx, elm] = 1
    return result, top



def precision_k(topk, gt, k, device="cpu"):
    """
    topk, gt: (U, X, I) array, where U is the number of users, X is the number of items per user, and I is the number of items in total.
    k: @k measurement
    """
    if device == "cpu":
        precision_values = []
        for i in range(topk.shape[0]):
            num_correct = np.multiply(topk[i], gt[i]).sum()
            precision_i = num_correct / (k)
            precision_values.append(precision_i)
        mean_precision = np.mean(precision_values)
    else:
        precision_values = []
        for i in range(topk.shape[0]):
            num_correct = torch.mul(topk[i], gt[i]).sum()
            precision_i = num_correct / (k)
            precision_values.append(precision_i)
        mean_precision = torch.mean(torch.tensor(precision_values))

    return mean_precision



def recall_k(topk, gt, k, device="cpu"):
    """
    topk, gt: (U, X, I) array, where U is the number of users, X is the number of items per user, and I is the number of items in total.
    k: @k measurement
    """
    if device == "cpu":
        recall_values = []
        for i in range(topk.shape[0]):
            recall_i = (
                np.multiply(topk[i], gt[i]).sum() / gt[i].sum()
                if gt[i].sum() != 0
                else 0
            )
            if gt[i].sum() != 0:
                recall_values.append(recall_i)
        mean_recall = np.mean(recall_values)
    else:
        recall_values = []
        for i in range(topk.shape[0]):
            recall_i = (
                torch.mul(topk[i], gt[i]).sum() / gt[i].sum() if gt[i].sum() != 0 else 0
            )
            if gt[i].sum() != 0:
                recall_values.append(recall_i)
        mean_recall = torch.mean(torch.tensor(recall_values))

    return mean_recall


def ndcg_k(rels, rels_ideal, gt, device="cpu"):
    """
    rels: sorted top-k arr
    rels_ideal: sorted top-k ideal arr
    """
    k = rels.shape[1]
    n = rels.shape[0]

    ndcg_values = []
    for row in range(n):
        dcg = 0
        idcg = 0
        for col in range(k):
            if gt[row, rels[row, col]] == 1:
                if col == 0:
                    dcg += 1
                else:
                    dcg += 1 / np.log2(col + 1)
            if gt[row, rels_ideal[row, col]] == 1:
                if col == 0:
                    idcg += 1
                else:
                    idcg += 1 / np.log2(col + 1)
        if idcg != 0:
            ndcg_values.append(dcg / idcg)

    mean_ndcg = torch.mean(torch.tensor(ndcg_values))

    return mean_ndcg


def recall_at_k_batch(gt, pred, k=20):
    """
    Compute per-user Recall@k in GPU-accelerated manner.

    Parameters
    ----------
    gt : np.ndarray, shape = [batch_size, n_items]
        Ground truth (binary) matrix for a batch of users.
    pred : np.ndarray, shape = [batch_size, n_items]
        Prediction scores for a batch of users.
    k : int
        The cut-off for top-k.

    Returns
    -------
    recall : np.ndarray, shape = [batch_size,]
        Per-user recall values.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1) Convert to torch tensors on GPU
    gt_t = torch.from_numpy(gt).float().to(device)        # shape=(batch_size, n_items)
    pred_t = torch.from_numpy(pred).float().to(device)    # shape=(batch_size, n_items)

    # 2) Top-k indices
    #    shape of top_idx: (batch_size, k)
    #    shape of top_val: (batch_size, k) => not necessarily used for recall
    top_val, top_idx = torch.topk(pred_t, k, dim=1)

    # 3) Convert ground truth to boolean(0/1)
    gt_bool = (gt_t > 0).float()  # shape=(batch_size, n_items)

    # 4) Gather hits from top_idx
    #    hits[i] = [gt_bool[i, idx] for idx in top_idx[i]]
    hits = torch.gather(gt_bool, 1, top_idx)             # shape=(batch_size, k)
    hits_sum = hits.sum(dim=1)                           # shape=(batch_size,)

    # 5) Compute recall = (# of hits) / (# of positives in ground truth)
    gt_sum = gt_bool.sum(dim=1)                          # shape=(batch_size,)
    # Avoid division by zero
    gt_sum = torch.where(gt_sum == 0, torch.tensor(1.0, device=device), gt_sum)
    
    recall = hits_sum / gt_sum                           # shape=(batch_size,)

    # 6) Move result to CPU and return as numpy array
    recall_cpu = recall.cpu().numpy() 
    return recall_cpu


def ndcg_at_k_batch(gt, pred, k=20):
    """
    Compute per-user NDCG@k in GPU-accelerated manner.

    Parameters
    ----------
    gt : np.ndarray, shape = [batch_size, n_items]
        Ground truth (binary) matrix for a batch of users.
    pred : np.ndarray, shape = [batch_size, n_items]
        Prediction scores for a batch of users.
    k : int
        The cut-off for top-k.

    Returns
    -------
    ndcg : np.ndarray, shape = [batch_size,]
        Per-user NDCG values.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1) Convert to torch tensors on GPU
    gt_t = torch.from_numpy(gt).float().to(device)     # shape=(batch_size, n_items)
    pred_t = torch.from_numpy(pred).float().to(device) # shape=(batch_size, n_items)

    # 2) Top-k indices
    top_val, top_idx = torch.topk(pred_t, k, dim=1)    # shape=(batch_size, k)

    # 3) Gains = ground truth boolean in the top-k
    gt_bool = (gt_t > 0).float()                       # shape=(batch_size, n_items)
    gains = torch.gather(gt_bool, 1, top_idx)          # shape=(batch_size, k)

    # 4) Compute discount factor: 1/log2( rank+1 )
    #    rank = 1..k => indices=0..k-1, so we do range(2, k+2)
    discounts = 1.0 / torch.log2(torch.arange(2, k+2, device=device).float())
    # shape(discounts)=(k,). We can broadcast to (batch_size, k)

    dcg = (gains * discounts).sum(dim=1)               # shape=(batch_size,)

    # 5) Compute ideal DCG
    #    Sort each user's ground-truth descending, take top-k
    #    Then multiply by same discount factor
    gt_bool_sorted, _ = torch.sort(gt_bool, dim=1, descending=True)
    ideal_gains = gt_bool_sorted[:, :k]                # shape=(batch_size, k)
    ideal_dcg = (ideal_gains * discounts).sum(dim=1)
    # Avoid division by zero
    ideal_dcg = torch.where(ideal_dcg == 0, torch.tensor(1.0, device=device), ideal_dcg)

    ndcg = dcg / ideal_dcg                             # shape=(batch_size,)

    # 6) Move result to CPU and return as numpy array
    ndcg_cpu = ndcg.cpu().numpy() 
    return ndcg_cpu




def torch2scipy_csc(mat_torch: torch.Tensor) -> sp.csc_matrix:
    """
    Convert a PyTorch sparse_coo_tensor to a SciPy csc_matrix
    """
    coo = mat_torch.coalesce().cpu()  # Ensure it's coalesced
    values = coo.values().numpy()
    indices = coo.indices().numpy()
    shape = coo.shape
    sp_coo = sp.coo_matrix((values, (indices[0], indices[1])), shape=shape)
    return sp_coo.tocsc()

def to_torch_tensor(matrix_or_tensor):
    if isinstance(matrix_or_tensor, torch.Tensor):
        return matrix_or_tensor.float()
    elif sp.issparse(matrix_or_tensor):
        arr = matrix_or_tensor.toarray()
        return torch.from_numpy(arr).float()
    elif isinstance(matrix_or_tensor, np.ndarray):
        return torch.from_numpy(matrix_or_tensor).float()
    else:
        raise TypeError("to_torch_tensoro: unsupported type {}".format(type(matrix_or_tensor)))


import scipy.sparse as sp
def normalize_sparse_adjacency_matrix_sparse(R, alpha=0.5):
    """
    Normalize the sparse adjacency matrix R.

    Parameters:
    - R: scipy.sparse matrix (user-item interaction matrix)
    - alpha: float (normalization exponent)

    Returns:
    - R_norm: scipy.sparse matrix (normalized adjacency matrix)
    """
    # Compute the degree of each user and item
    user_degrees = np.array(R.sum(axis=1)).flatten()  # Sum over rows (users)
    item_degrees = np.array(R.sum(axis=0)).flatten() # Sum over columns (items)

    # Avoid division by zero
    user_degrees[user_degrees == 0] = 1e-12
    item_degrees[item_degrees == 0] = 1e-12

    # Compute D_u^{-alpha}
    D_u_inv = sp.diags(np.power(user_degrees, -alpha))

    # Compute D_i^{alpha - 1} (corrected exponent)
    D_i_inv = sp.diags(np.power(item_degrees, alpha - 1))

    # Normalize R
    R_norm = D_u_inv.dot(R).dot(D_i_inv)
    R_norm = R_norm.astype(np.float32)

    # Compute D_i^{alpha} (separate degree matrix for ideal LPF)
    D_i = sp.diags(np.power(item_degrees, alpha))

    return R_norm, D_i, D_i_inv

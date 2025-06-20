import random
import numpy as np
import torch
import scipy.sparse as sp
from utils import *
from tqdm import tqdm
import time
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import eigsh # For PGSP
from sklearn.mixture import GaussianMixture # For HIGSP

class GF_CF:
    def __init__(self, args, R_norm, D_i, D_i_inv, device, alpha=0.1, k=400):
        self.args   = args
        self.device = device
        self.alpha  = alpha
        self.k      = k

        self.R_norm = R_norm.to(device)
        if self.R_norm.is_sparse:
            self.R_norm_T = self.R_norm.transpose(0, 1).coalesce()
        else:
            self.R_norm_T = self.R_norm.transpose(0, 1)

        self.d_i     = self._extract_diag_to_1d(D_i)
        self.d_i_inv = self._extract_diag_to_1d(D_i_inv)

        print("Starts filter computation with torch.svd_lowrank()...\n")
        self.VSVT = self._compute_VSVT(self.R_norm, self.k)
        print("Filter computation done.\n")

    def _extract_diag_to_1d(self, mat):
        if isinstance(mat, np.ndarray):
            if mat.ndim == 1:
                diag_vals = mat
            elif mat.ndim == 2:
                diag_vals = np.diag(mat)
            else:
                raise ValueError("D_i or D_i_inv has unexpected shape.")
            return torch.from_numpy(diag_vals).float().to(self.device)

        elif sp.issparse(mat):
            if isinstance(mat, sp.dia.dia_matrix):
                diag_vals = mat.diagonal()  # (n_i,)
                return torch.from_numpy(diag_vals).float().to(self.device)
            else:
                diag_vals = mat.diagonal()
                return torch.from_numpy(diag_vals).float().to(self.device)

        elif isinstance(mat, torch.Tensor):
            if mat.ndim == 1:
                return mat.float().to(self.device)
            elif mat.ndim == 2:
                return torch.diag(mat).float().to(self.device)
            else:
                raise ValueError("D_i or D_i_inv must be 1D or 2D tensor.")
        else:
            raise TypeError("D_i / D_i_inv must be np.ndarray, scipy.sparse, or torch.Tensor.")

    def _compute_VSVT(self, R_norm: torch.Tensor, k: int) -> torch.Tensor:
        if R_norm.is_sparse:
            R_dense = R_norm.to_dense()
        else:
            R_dense = R_norm

        U, S, V = torch.svd_lowrank(R_dense, q=k)
        S_mat   = torch.diag(S)
        VS      = V @ S_mat
        VSVT    = VS @ V.t()  # (n_i x n_i)
        return VSVT.to(self.device)

    def predict_batch(self, r_u: torch.Tensor) -> torch.Tensor:
        if self.R_norm.is_sparse:
            temp = r_u.mm(self.R_norm_T)  # shape=(b, n_u)
            out  = temp.mm(self.R_norm)   # shape=(b, n_i)
        else:
            out  = r_u @ self.R_norm_T
            out  = out @ self.R_norm
        out_ideal_1 = r_u * self.d_i  # shape(b, n_i), broadcasting
        out_ideal_2 = out_ideal_1.mm(self.VSVT)
        out_ideal   = out_ideal_2 * self.d_i_inv

        out += self.alpha * out_ideal
        return out
    


class Turbo_CF:
    def __init__(self, args, R_norm, filter_order, device):
        self.args = args
        self.device= device
        self.R_norm = R_norm
        self.filter_order = filter_order 
        self.power = 1.0

        print("Starts filter computation .. \n")
        self.P = self.get_similarity_graph()
        # Dense P makes memory more efficient
        self.P = self.P.to_dense()
        # self.P.data **=  self.power

    def predict_batch(self, r_u):
        if self.filter_order == 1:
            return r_u @ self.P
        elif self.filter_order == 2:
            return r_u @ (2 * self.P - self.P@self.P)
        elif self.filter_order == 3:
            return r_u @ (self.P + 0.01*(-self.P@self.P@self.P +10*self.P@self.P - 29*self.P))
    
    def get_similarity_graph(self):
        return torch.sparse.mm(self.R_norm.transpose(0, 1),self.R_norm)
    
class Mem_GF:
    def __init__(self, args, R_norm, filter_order, a_k, device):
        self.args = args
        self.krylov_order = filter_order + 1
        self.device = device
        self.R_norm = R_norm
        self.poly_coeff = a_k
        # 3) Additional power step
        self.R_norm = self.R_norm ** args.power

    def predict_batch(self, r_u):
        # 5.2) Normalize row vectors (for Lanczos init)
        norm_r_u = torch.norm(r_u, dim=1, keepdim=True)
        zero_mask = (norm_r_u.squeeze() == 0)
        norm_r_u[zero_mask] = 1e-10
        
        q1 = (r_u / norm_r_u).to(self.device)

        # 5.3) Lanczos steps
        Q, alpha, beta = lanczos_algorithm(self.R_norm, q1, self.krylov_order, self.device)

        T_m = compute_T(alpha, beta, self.device)  # shape: [batch_users, K, K]
        f_T_m = polynomial_filter(T_m, self.poly_coeff, self.device)

        # 5.4) Multiply f(T_m) * e1 => shape=[batch, K,1] => then Q_m => shape=[batch, n_items]
        e1 = torch.zeros((len(r_u), self.krylov_order, 1), device=self.device)
        e1[:, 0, 0] = 1.0

        f_T_m_e1 = torch.bmm(f_T_m, e1)  # (batch_users, K, 1)
        Q_m = Q[:-1].permute(1, 0, 2)    # (K, batch_users, n_items) => (batch_users, K, n_items)

        s_u = torch.bmm(f_T_m_e1.transpose(1, 2), Q_m)  # => (batch_users, 1, n_items)
        s_u = s_u.squeeze(1)                            # (batch_users, n_items)

        return norm_r_u * s_u                            # multiply back by norm_r_u
   


class PGSP:
    def __init__(
        self, 
        args,
        R_norm: torch.Tensor,  # (n_users x n_items)
        R:      torch.Tensor,  # (n_users x n_items)
        device: torch.device,
        alpha:  float = 0.1,
        k:      int   = 30,
        beta:   float = 0.5    # 예: degree 가중 지수
    ):
        self.args = args
        self.device = device
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n_users, self.n_items = R.shape
        if R_norm.is_sparse:
            R_norm_dense = R_norm.to_dense()
        else:
            R_norm_dense = R_norm
        
        item_deg = torch.sum(R_norm_dense, dim=0)  # shape=(n_items,)
        # pow
        item_deg_beta = item_deg.pow(self.beta)    # (n_items,)
        # diag
        D_item_beta = torch.diag(item_deg_beta).to(device)
        # R_norm_modified
        R_norm_modified = R_norm_dense.to(device).mm(D_item_beta)

        R_norm_scipy = sp.csr_matrix(R_norm_modified.cpu().numpy())
        if R.is_sparse:
            R_scipy = torch_sparse_to_scipy_csr(R)
        else:
            R_scipy = sp.csr_matrix(R.cpu().numpy())

        # 증강 그래프
        zero_user_user = sp.csr_matrix((self.n_users, self.n_users), dtype=np.float32)
        zero_item_item = sp.csr_matrix((self.n_items, self.n_items), dtype=np.float32)
        top_row = sp.hstack([zero_user_user, R_scipy], format='csr')
        bottom_row = sp.hstack([R_scipy.transpose(), zero_item_item], format='csr')
        S_aug = sp.vstack([top_row, bottom_row], format='csr')

        deg = np.array(S_aug.sum(axis=1)).ravel()
        D_aug = sp.diags(deg, offsets=0, dtype=np.float32)
        L_aug = D_aug - S_aug

        print("Starts ideal lpf computation .. \n")
        vals, vecs = eigsh(L_aug, k=self.k, which='SM')
        print("Decomposition done .. \n")
        
        U_k = torch.from_numpy(vecs).float()  # (n_users+n_items, k)
        H_ideal = U_k @ U_k.t()

        # Linear LPF = I - L_aug 
        L_aug_dense = torch.from_numpy(L_aug.toarray()).float()
        num_nodes = self.n_users + self.n_items
        I_aug = torch.eye(num_nodes)
        H_linear = I_aug - L_aug_dense

        self.H_ideal = H_ideal.to(device)
        self.H_linear = H_linear.to(device)

        self.item_deg_beta = item_deg_beta  # (n_items,)

        print("Filter computation done .. \n")


    def predict_batch(self, p_u: torch.Tensor, r_u: torch.Tensor) -> torch.Tensor:

        R_personalized = torch.cat([p_u, r_u], dim=1)  # (batch_size, n_users + n_items)


        out_ideal = R_personalized.mm(self.H_ideal)
        out_linear = R_personalized.mm(self.H_linear)
        out_mixed = (1 - self.alpha) * out_ideal + self.alpha * out_linear


        # R_pred = R_filtered[:, n_users:]
        r_pred = out_mixed[:, self.n_users:]  # (batch_size, n_items)
        deg_inv_beta = self.item_deg_beta.pow(-1).to(self.device)  # (n_items,)
        D_item_inv_beta = torch.diag(deg_inv_beta)
        r_pred = r_pred.mm(D_item_inv_beta)

        return r_pred


class LGCN_IDE:
    def __init__(self, args, R_norm, filter_order, device):
        self.args = args
        self.device= device
        self.R_norm = R_norm
        self.filter_order = filter_order 
        self.P = self.get_similarity_graph()
        # Dense P makes memory more efficient
        self.P = self.P.to_dense()

    def predict_batch(self, r_u):
        if self.filter_order == 1:
            return r_u @ self.P
        elif self.filter_order == 2:
            return r_u @ (2 * self.P - self.P@self.P)
        
    def get_similarity_graph(self):
        return torch.sparse.mm(self.R_norm.transpose(0, 1),self.R_norm)
    


class EASE:
    def __init__(self, args, R_norm, device):
        self.args = args
        self.device = device
        self.R_norm = R_norm  # Normalized user-item interaction matrix
        self.lambda_reg = 0.3 # Regularization parameter

        print("Starts filter computation .. \n")
        self.P = self.get_similarity_graph()  # Precompute the similarity graph

    def predict_batch(self, r_u):
        return r_u @ self.P

    def get_similarity_graph(self):
        # Compute the Gram matrix (convert to dense for compatibility)
        G = torch.mm(self.R_norm.T, self.R_norm).to_dense()  # Ensure G is dense
        # Add regularization to the diagonal
        lambda_eye = torch.eye(G.size(0), device=self.device) * self.lambda_reg
        G += lambda_eye  
        # Compute the inverse of the regularized Gram matrix
        P = torch.linalg.inv(G)
        # Compute the item-item weight matrix with zero diagonal (B matrix in EASE)
        B = P / (-torch.diag(P).unsqueeze(1))  # Broadcast for element-wise division
        B.fill_diagonal_(0) 
        return B

        

class SVD_AE:
    def __init__(self, args, R_norm, R_tr,  device):
        self.args = args
        self.device= device
        self.R_norm = R_norm
        self.R = R_tr
        self.ut, self.s, self.vt = torch.svd_lowrank(self.R_norm, q=int(len(R_norm.T) * 0.04), niter=3, M=None)
        # s to diag mat and its element to inverse  
        self.s = torch.diag(1/self.s)

    def predict_batch(self, r_u):
        return  r_u @ self.vt @ self.s @ self.ut.T @ self.R

import torch
import numpy as np
import scipy.sparse as sp
from sklearn.mixture import GaussianMixture
from scipy.sparse.linalg import svds
from utils import torch2scipy_csc  # Assume you have a helper for R_norm->CSC

class HIGSP:
    """
    HiGSP example referencing GF_CF style, with minimal overhead from CSR->COO conversions.
    """
    def __init__(
        self,
        args,
        R_norm:  torch.Tensor,   # (n_users x n_items), user-item matrix (normalized if needed)
        R_tilde: torch.Tensor,   # (n_items x n_items), already normalized item-item graph
        device:  torch.device,
        alpha1:  float = 0.5,
        alpha2:  float = 0.5,
        order1:  int   = 2,
        order2:  int   = 2,
        n_clusters: int = 2,
        k_svd:    int   = 50
    ):
        self.device = device
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.order1 = order1
        self.order2 = order2
        self.n_clusters = n_clusters
        self.k_svd = k_svd

        # 1) Store user-item matrix in sparse format if possible
        if not R_norm.is_sparse:
            R_norm = R_norm.to_sparse()
        self.R_norm = R_norm.to(device)

        # 2) Store item-item graph in sparse if possible
        if not R_tilde.is_sparse:
            R_tilde = R_tilde.to_sparse()
        self.R_tilde = R_tilde.to(device)

        print("Starts building clusters and its filters .. \n")
        self.cluster_filters, self.cluster_labels = self.build_cluster_filters()

        print("Starts filter computation .. \n")
        print("Global filter computations ..")
        self.ideal_lpf = self.build_ideal_lpf()
        self.high_order_lpf = self.build_high_order_lpf()

    def build_cluster_filters(self):
        """
        1) Cluster users by GMM on dense R_norm
        2) For each cluster, build local filter Fc = I - (I - Gc)^order1 (sparse)
        """
        # -- GMM needs dense array
        R_dense = self.R_norm.to_dense().cpu().numpy()
        gmm = GaussianMixture(n_components=self.n_clusters, random_state=42)
        user_labels = gmm.fit_predict(R_dense)

        n_users, n_items = R_dense.shape
        filters = []

        for c_id in range(self.n_clusters):
            idx_c = np.where(user_labels == c_id)[0]
            if len(idx_c) == 0:
                # If cluster is empty
                eye_mat = self._eye_torch_sparse(n_items)
                filters.append(eye_mat)
                continue

            # R_c -> shape=(|idx_c| x n_items)
            R_c = R_dense[idx_c, :]
            R_c_sp = sp.csr_matrix(R_c)
            G_c_sp = R_c_sp.transpose() @ R_c_sp  # (n_items x n_items), CSR

            # Row-col normalize
            G_c_sp = self._rowcol_normalize_sp(G_c_sp)

            # L_c = I - G_c
            L_c = self._eye_sparse(n_items) - G_c_sp
            # L_c^order1
            L_c_k = self._matpow_sparse(L_c, self.order1)
            # F_c_sp = I - L_c^k
            F_c_sp = self._eye_sparse(n_items) - L_c_k

            # Finally convert CSR -> COO -> torch sparse
            F_c_coo = F_c_sp.tocoo()
            row = F_c_coo.row
            col = F_c_coo.col
            indices_np = np.vstack((row, col))  # shape = (2, nnz)
            i_idx = torch.LongTensor(indices_np)
            v_val = torch.FloatTensor(F_c_coo.data)
            shape = (n_items, n_items)
            F_c_torch = torch.sparse_coo_tensor(i_idx, v_val, shape).coalesce().to(self.device)
            filters.append(F_c_torch)

        return filters, user_labels

    def build_ideal_lpf(self):
        """
        Ideal LPF via top-K SVD of R_norm, similar to GF_CF.
        """
        R_csc = torch2scipy_csc(self.R_norm)  # (n_users x n_items) csc
        u, s, vt = svds(R_csc, k=self.k_svd)
        s_mat = np.diag(s)
        ideal_np = vt.T @ s_mat @ vt  # shape=(n_items, n_items)

        # Convert to sparse
        ideal_sp = sp.csr_matrix(ideal_np)
        ideal_coo = ideal_sp.tocoo()
        i_idx = torch.LongTensor([ideal_coo.row, ideal_coo.col])
        v_val = torch.FloatTensor(ideal_coo.data)
        shape = ideal_coo.shape
        return torch.sparse_coo_tensor(i_idx, v_val, shape).coalesce().to(self.device)

    def build_high_order_lpf(self):
        """
        High-order LPF: F_H = I - (I - R_tilde)^order2
        R_tilde is assumed to be (n_items x n_items) already normalized, torch.sparse
        """
        n_items = self.R_tilde.size(0)
        L = self._eye_torch_sparse(n_items) - self.R_tilde
        L_k = self._matpow_torch_sparse(L, self.order2)
        F_H = self._eye_torch_sparse(n_items) - L_k
        return F_H.coalesce()

    def predict_batch(self, r_u: torch.Tensor, user_indices: torch.Tensor) -> torch.Tensor:
        """
        P = r_u * F_c + alpha1*(r_u * ideal_lpf) + alpha2*(r_u * high_order_lpf)
        r_u: (batch_size, n_items)
        user_indices: (batch_size,)
        """
        batch_size, n_items = r_u.shape
        out = torch.zeros_like(r_u, device=self.device)

        # We assume r_u is dense
        if r_u.is_sparse:
            r_u = r_u.to_dense()

        for i in range(batch_size):
            u_id = user_indices[i].item()
            c_id = self.cluster_labels[u_id]

            # local filter
            F_c = self.cluster_filters[c_id]  # torch.sparse
            r_local = torch.sparse.mm(F_c, r_u[i].unsqueeze(1)).squeeze(1)

            # ideal LPF
            r_g1 = torch.sparse.mm(self.ideal_lpf, r_u[i].unsqueeze(1)).squeeze(1)

            # high-order LPF
            r_g2 = torch.sparse.mm(self.high_order_lpf, r_u[i].unsqueeze(1)).squeeze(1)

            out[i] = r_local + self.alpha1 * r_g1 + self.alpha2 * r_g2

        return out

    def _rowcol_normalize_sp(self, mat_sp: sp.csr_matrix) -> sp.csr_matrix:
        """
        Row-col symmetric normalization: D^{-1/2} * mat_sp * D^{-1/2}
        """
        row_sum = np.array(mat_sp.sum(axis=1)).flatten()
        d_inv_r = np.power(row_sum, -0.5, where=row_sum != 0)
        d_inv_r[np.isinf(d_inv_r)] = 0.
        D_inv_r = sp.diags(d_inv_r)
        temp = D_inv_r.dot(mat_sp)

        col_sum = np.array(temp.sum(axis=0)).flatten()
        d_inv_c = np.power(col_sum, -0.5, where=col_sum != 0)
        d_inv_c[np.isinf(d_inv_c)] = 0.
        D_inv_c = sp.diags(d_inv_c)

        return temp.dot(D_inv_c).tocsr()

    def _matpow_sparse(self, mat_sp: sp.csr_matrix, order: int) -> sp.csr_matrix:
        """
        CSR-based sparse matrix power. Returns CSR.
        """
        if order < 1:
            n = mat_sp.shape[0]
            return sp.eye(n, format='csr')
        out = mat_sp.copy()
        for _ in range(2, order + 1):
            out = out.dot(mat_sp)
        return out

    def _eye_sparse(self, n: int) -> sp.csr_matrix:
        return sp.eye(n, format='csr')

    def _eye_torch_sparse(self, n: int) -> torch.Tensor:
        idx = torch.arange(n, dtype=torch.long, device=self.device)
        i_idx = torch.stack([idx, idx], dim=0)
        v_val = torch.ones(n, device=self.device)
        return torch.sparse_coo_tensor(i_idx, v_val, (n, n)).coalesce()

    def _matpow_torch_sparse(self, mat_sp: torch.Tensor, order: int) -> torch.Tensor:
        """
        Torch sparse matrix power using repeated multiplication.
        """
        if order < 1:
            return self._eye_torch_sparse(mat_sp.size(0))
        out = mat_sp.clone()
        for _ in range(2, order + 1):
            out = torch.sparse.mm(out, mat_sp)
        return out.coalesce()

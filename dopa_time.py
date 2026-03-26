import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.optimize import minimize
import cvxpy as cp
from sparsemax import Sparsemax
import torch  # needed to call Sparsemax and convert result to numpy
from tqdm import trange
import time

############## ============================================================ ##############
#  Tsallis-FTRL probabilities (alpha = 1/2) 
############## ============================================================ ##############


def tsallis_probs(u, eta, q=0.5,eps=1e-8):
    u = np.asarray(u, dtype=float)
    K = u.shape[0]

    def F(s):
        base = s * (q - 1.0) / q + 1.0 / q 
        base = np.maximum(base, eps)
        return base ** (1.0 / (q - 1.0))

    def F_k(s):
        return np.clip(1.0 - F(-s / float(eta)), 0.0, 1.0)

    def F_inv(t):
        t = np.maximum(t, eps)
        return (q * (t ** (q - 1.0)) - 1.0) / (q - 1.0)

    Fk_inv_1_minus_1_over_K = -eta * F_inv(1.0 / K)

    offset = Fk_inv_1_minus_1_over_K
    tau_u = np.max(-u - offset)
    tau_l = np.min(-u - offset)

    if tau_u < tau_l:
        tau_u, tau_l = tau_l, tau_u

    # Dynamic n_iter calculation based on precision
    width = tau_u - tau_l
    tol = eps / 0.25 / np.sqrt(K)
    if width > tol:
        n_iter = int(np.ceil(np.log2(width / tol)))
    else:
        n_iter = 0

    tau = 0.5 * (tau_u + tau_l)
    for _ in range(n_iter):
        tau = 0.5 * (tau_u + tau_l)
        p_hat = 1.0 - F_k(-u - tau)
        if p_hat.sum() > 1.0:
            tau_u = tau
        else:
            tau_l = tau

    F_vals = F_k(-u - tau)
    S = F_vals.sum()
    p = (1.0 + S) / K - F_vals

    p = np.maximum(p, 0.0)
    s = p.sum()
    if s <= 0:
        p = np.ones_like(p) / K
    else:
        p /= s

    return p



############## ============================================================ ##############
#  Renyi-FTRL probabilities
############## ============================================================ ##############

def renyi_probs(u, eta, eps_=1e-8):

    u = np.asarray(u, dtype=float)
    K = u.shape[0]

    p = cp.Variable(K, nonneg=True)
    
    # Reformulation to avoid log(sum(sqrt(p))) which can be unstable/hit recursion limits for large K
    # Maximize u^T p + 2 * eta * log(sum(sqrt(p)))
    # Let q_i <= sqrt(p_i)  <=>  q_i^2 <= p_i  (Rotated SOC)
    # Let t <= log(sum(q))  <=>  exp(t) <= sum(q) (ExpCone)
    
    q = cp.Variable(K, nonneg=True)
    t = cp.Variable()

    constraints = [
        cp.sum(p) == 1,
        p <= 1
    ]

    # Vectorized SOC: || [2*q_i, p_i-1] ||_2 <= p_i + 1
    # Equivalent to q_i^2 <= p_i
    z_soc = cp.vstack([2 * q, p - 1])
    constraints.append(cp.SOC(p + 1, z_soc, axis=0))

    # ExpCone: t <= log(sum(q))
    constraints.append(cp.ExpCone(t, 1, cp.sum(q)))

    objective = cp.Minimize(-u @ p - 2 * eta * t)

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.MOSEK, eps=eps_)
    except:
        prob.solve(solver=cp.SCS, eps=eps_)
    
    p_val = p.value

    if p_val is None:
        return np.ones(K) / K

    s = p_val.sum() 
    if s <= 0:
        p_val = np.ones(K) / K
    elif s > 1.01:
        # print("Warning: renyi_probs sum > 1.01")
        # print(f"sum={s}, u={u}, eta={eta}, p={p_val}")
        p_val /= s 
    else:
        p_val /= s

    return p_val


def tsallis_probs_cvxpy(u, eta):
    u = np.asarray(u, dtype=float)
    K = u.shape[0]
    p = cp.Variable(K, nonneg=True)
    z = cp.Variable(K)

    constraints = [
        cp.sum(p) == 1,
        p <= 1
    ]
    
    X = cp.vstack([cp.reshape(2 * z, (1, K)), cp.reshape(1 - p, (1, K))])
    constraints.append(cp.SOC(1 + p, X, axis=0))

    objective = cp.Maximize(u @ p + 2 * eta * cp.sum(z - p))
    
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.MOSEK)
    except:
        prob.solve(solver=cp.SCS)
    return p.value

# ==============================
# Sparsemax Probs
# ==============================

def sparsemax_probs(u, eta=1.0, n_iter=60, eps=1e-8):
    u = np.asarray(u, dtype=float)
    K = u.shape[0]

    # F(s) = s/2 + 1/2 for s in [-1, 1]
    def F(s):
        return 0.5 * s + 0.5

    def F_k(s):
        # 1.0 - F(-s/eta)
        val = 0.5 * (1.0 + s / float(eta))
        return np.clip(val, 0.0, 1.0)

    def F_inv(t):
        return 2.0 * t - 1.0

    Fk_inv_1_minus_1_over_K = -eta * F_inv(1.0 / K)

    offset = Fk_inv_1_minus_1_over_K
    tau_u = np.max(-u - offset)
    tau_l = np.min(-u - offset)

    if tau_u < tau_l:
        tau_u, tau_l = tau_l, tau_u

    tau = 0.5 * (tau_u + tau_l)
    width = tau_u - tau_l
    tol = eps / 0.5 / np.sqrt(K)
    if width > tol:
        n_iter = int(np.ceil(np.log2(width / tol)))
    else:
        n_iter = 0
    for _ in range(n_iter):
        tau = 0.5 * (tau_u + tau_l)
        p_hat = 1.0 - F_k(-u - tau)
        if p_hat.sum() > 1.0:
            tau_u = tau
        else:
            tau_l = tau

    F_vals = F_k(-u - tau)
    S = F_vals.sum()
    p = (1.0 + S) / K - F_vals

    p = np.maximum(p, 0.0)
    s = p.sum()
    if s <= 0:
        p = np.ones_like(p) / K
    else:
        p /= s

    return p

def run_projection_benchmark(K_values, n_sims=10):
    print("Running projection benchmark...")
    times = {
        'DOPA (Uniform, bisection)': {'mean': [], 'min': [], 'max': []},
        'DOPA (Pareto, bisection)': {'mean': [], 'min': [], 'max': []}, # Tsallis
        'FTRL (Tsallis, MOSEK)': {'mean': [], 'min': [], 'max': []},
        'FTRL (Renyi, MOSEK)': {'mean': [], 'min': [], 'max': []}
    }
    errors = {
        'DOPA (Pareto, bisection)': {'mean': [], 'min': [], 'max': []}
    }
    
    for K in K_values:
        print(f"Benchmarking K={K}...")
        t_sparse = []
        t_tsallis = []
        t_tsallis_cvx = []
        t_renyi = []
        e_tsallis = []
        
        for _ in range(n_sims):
            u = np.random.rand(K)
            eta = 1.0
            
            # Sparsemax
            tic = time.perf_counter()
            sparsemax_probs(u, eta)
            t_sparse.append(time.perf_counter() - tic)
            
            # Tsallis (Fast)
            tic = time.perf_counter()
            p_ts = tsallis_probs(u, eta)
            t_tsallis.append(time.perf_counter() - tic)

            # Tsallis CVXPY (Ground Truth)
            tic = time.perf_counter()
            p_ts_cvx = tsallis_probs_cvxpy(u, eta)
            t_tsallis_cvx.append(time.perf_counter() - tic)
            
            if p_ts_cvx is not None:
                e_tsallis.append(np.sum(np.abs(p_ts - p_ts_cvx)))
            else:
                e_tsallis.append(0.0)
            
            # Renyi
            tic = time.perf_counter()
            renyi_probs(u, eta)
            t_renyi.append(time.perf_counter() - tic)
            
        # Store stats
        for name, data in zip(['DOPA (Uniform, bisection)', 'DOPA (Pareto, bisection)', 'FTRL (Tsallis, MOSEK)', 'FTRL (Renyi, MOSEK)'], 
                              [t_sparse, t_tsallis, t_tsallis_cvx, t_renyi]):
            times[name]['mean'].append(np.mean(data))
            times[name]['min'].append(np.min(data))
            times[name]['max'].append(np.max(data))
            
        errors['DOPA (Pareto, bisection)']['mean'].append(np.mean(e_tsallis))
        errors['DOPA (Pareto, bisection)']['min'].append(np.min(e_tsallis))
        errors['DOPA (Pareto, bisection)']['max'].append(np.max(e_tsallis))
            
    return times, errors

def plot_projection_benchmark(times, K_values):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'DOPA (Uniform, bisection)': 'gray', 'DOPA (Pareto, bisection)': 'purple', 'FTRL (Tsallis, MOSEK)': 'red', 'FTRL (Renyi, MOSEK)': 'orange'}
    
    for alg in ['DOPA (Uniform, bisection)', 'DOPA (Pareto, bisection)', 'FTRL (Tsallis, MOSEK)', 'FTRL (Renyi, MOSEK)']:
        means = np.array(times[alg]['mean'])
        mins = np.array(times[alg]['min'])
        maxs = np.array(times[alg]['max'])
        
        ax.plot(K_values, means, label=alg, color=colors[alg], linewidth=2)
        ax.fill_between(K_values, mins, maxs, color=colors[alg], alpha=0.2)
        
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('# of arms (K)', fontsize=14)
    ax.set_ylabel('Execution time (s)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", ls="-", alpha=0.3)
    
    fig.tight_layout()
    fig.savefig("dopa_time.pdf")
    fig.show()

def plot_quality_benchmark(errors, K_values):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'DOPA (Pareto)': 'purple'}
    
    alg = 'DOPA (Pareto)'
    means = np.array(errors[alg]['mean'])
    mins = np.array(errors[alg]['min'])
    maxs = np.array(errors[alg]['max'])
    
    ax.plot(K_values, means, label=f"{alg} vs CVXPY", color=colors[alg], linewidth=2, marker='o')
    ax.fill_between(K_values, mins, maxs, color=colors[alg], alpha=0.2)
        
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('# of arms (K)', fontsize=14)
    ax.set_ylabel('L1 Error', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", ls="-", alpha=0.3)
    
    fig.tight_layout()
    fig.savefig("dopa_quality.pdf")
    fig.show()


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    K_bench = np.unique(np.logspace(0, 4, num=50).astype(int))
    times_bench, errors_bench = run_projection_benchmark(K_values=K_bench, n_sims=10)
    plot_projection_benchmark(times_bench, K_bench)
    # plot_quality_benchmark(errors_bench, K_bench)

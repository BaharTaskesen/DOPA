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


def tsallis_probs(u, eta, q=0.5,n_iter=60,eps=1e-8):
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

import numpy as np


############## ============================================================ ##############
#  Hyperbolic-FTRL probabilities
############## ============================================================ ##############
# F(s) = sinh(s - k) where k = sqrt(2) - 1 - arcsinh(1)
#         k = np.sqrt(2) - 1.0 - np.arcsinh(1.0)
#         return np.sinh(s - k)
# L = sqrt(2)

def hyperbolic_probs(u, eta, n_iter=60, eps=1e-8):
    u = np.asarray(u, dtype=float)
    K = u.shape[0]

    # Constants for Hyperbolic
    k_val = np.sqrt(2) - 1.0 - np.arcsinh(1.0)

    def F(s):
        return np.sinh(s - k_val)

    def F_k(s):
        return np.clip(1.0 - F(-s / float(eta)), 0.0, 1.0)

    def F_inv(t):
        return np.arcsinh(t) + k_val

    Fk_inv_1_minus_1_over_K = -eta * F_inv(1.0 / K)

    offset = Fk_inv_1_minus_1_over_K
    tau_u = np.max(-u - offset)
    tau_l = np.min(-u - offset)

    if tau_u < tau_l:
        tau_u, tau_l = tau_l, tau_u

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


# ============================================================
#  Exponential FTRL: softmax
# ============================================================

def softmax_probs(u, eta):
    # Use scipy.special.softmax for numerical stability and simplicity.
    # It internally performs the shift z <- z - max(z) to prevent overflow (exp(large) -> inf).
    return softmax(np.asarray(u) / float(eta))

# ==============================
# Sparsemax Probs
# ==============================

def sparsemax_probs(u, eta=1.0):

    z = np.asarray(u, dtype=float) / float(eta)
    K = z.shape[0]

    # Sort in descending order
    z_sorted = np.sort(z)[::-1]
    z_cumsum = np.cumsum(z_sorted)

    k = np.arange(1, K + 1)
    # τ_k candidates
    tau_candidates = (z_cumsum - 1.0) / k

    # Find k* = max {k : z_(k) - τ_k > 0}
    cond = z_sorted - tau_candidates > 0
    if not np.any(cond):
        # fallback to uniform if something degenerates
        return np.ones(K) / K

    k_star = np.max(np.where(cond)[0])
    tau = tau_candidates[k_star]

    p = np.maximum(z - tau, 0.0)
    p_sum = p.sum()
    if p_sum <= 0:
        p = np.ones(K) / K
    else:
        p /= p_sum
    return p


# ==============================
# Data-Driven Probs
# ==============================

def data_driven_probs(u, eta, n_iter=60):
    u = np.asarray(u, dtype=float)
    K = u.shape[0]
    
    # Use empirical CDF of -u/eta as the "shape"
    vals = -u / float(eta)
    vals_sorted = np.sort(vals)
    y_vals = np.linspace(0, 1, K)
    
    z = -u / float(eta)
    
    # Bounds for tau
    tau_min = np.min(z) - np.max(vals_sorted) - 10.0
    tau_max = np.max(z) - np.min(vals_sorted) + 10.0
    
    for _ in range(n_iter):
        tau = 0.5 * (tau_min + tau_max)
        
        # Vectorized interpolation
        f_vals = np.interp(z - tau, vals_sorted, y_vals, left=0.0, right=1.0)
        s_val = np.sum(f_vals)
            
        if s_val > K - 1:
            tau_min = tau
        else:
            tau_max = tau
            
    p = 1.0 - np.interp(z - tau, vals_sorted, y_vals, left=0.0, right=1.0)
    p = np.maximum(p, 0.0)
    if p.sum() > 0:
        p /= p.sum()
    else:
        p = np.ones(K)/K
        
    return p

# ============================================================
#  Environments
# ============================================================

def make_stochastic_means(K, gap=0.1):
    base = 0.9
    means = base - gap * np.arange(K)
    means = np.clip(means, 0.05, 0.95)
    return means

def make_many_suboptimal_means(K, gap=0.1):
    # 1 optimal arm at 1.0 (Deterministic), K-1 arms at 1.0 - gap
    # This high SNR environment prevents sparse algorithms from accidentally killing the optimal arm.
    means = np.full(K, 1.0 - gap)
    means[0] = 0.99
    return means


def make_adversarial_rewards(T, K):
    # Alternating Adversary
    # Arm 0 is good for even rounds.
    # Arm 1 is good for odd rounds.
    R = np.zeros((T, K), dtype=float)
    for t in range(T):
        if t % 2 == 0:
            R[t, 0] = 1.0
        else:
            R[t, 1] = 1.0
    return R

# ============================================================
#  Simulation functions
# ============================================================

def run_generic(T, K, n_runs, get_reward_fn, get_regret_fn_factory, get_rng_fn,
                eta_exp, eta_uni, eta_hyp, eta_renyi, eta_data, alpha, desc, initial_u=None):
    
    algs = ['tsallis', 'hyp', 'renyi', 'exp', 'uni', 'data', 'ensemble']
    regrets = {alg: np.zeros((n_runs, T)) for alg in algs}
    times = {alg: np.zeros(n_runs) for alg in algs}
    
    # Pre-calculate Tsallis eta (using T)
    eta_tsa = np.sqrt(T * (1 - alpha) / (2 * alpha)) * K ** (alpha - 1/2)

    for run in trange(n_runs, desc=desc):
        rng = get_rng_fn(run)
        
        # Initialize u with prior knowledge if provided
        if initial_u is not None:
            us = {alg: initial_u.copy() for alg in algs}
        else:
            us = {alg: np.zeros(K) for alg in algs}
            
        # Initialize Ensemble Agent
        # Use eta_uni (Sparsemax) and eta_hyp (Hyperbolic) for the sub-experts

        cum_regrets = {alg: 0.0 for alg in algs}
        cum_rewards = {alg: 0.0 for alg in algs}
        run_times = {alg: 0.0 for alg in algs}
        
        calc_regret = get_regret_fn_factory(run)

        def step(alg, prob_func, eta, **kwargs):
            tic = time.perf_counter()
            
            p = prob_func(us[alg], eta=eta, **kwargs)
            a = rng.choice(K, p=p)
            r = get_reward_fn(t, a, rng)
            
            # Update
            r_bandit = r -1
            us[alg][a] += r_bandit / p[a]
            
            run_times[alg] += time.perf_counter() - tic
            return a, r

        for t in range(1, T + 1):
            # Execute steps for all algorithms
            
            # Tsallis
            a, r = step('tsallis', tsallis_probs, eta_tsa)
            calc_regret('tsallis', t, a, r, cum_regrets, cum_rewards, regrets, run)

            # Hyperbolic
            a, r = step('hyp', hyperbolic_probs, eta_hyp)
            calc_regret('hyp', t, a, r, cum_regrets, cum_rewards, regrets, run)

            # Renyi
            a, r = step('renyi', renyi_probs, eta_renyi)
            calc_regret('renyi', t, a, r, cum_regrets, cum_rewards, regrets, run)

            # Exponential
            a, r = step('exp', softmax_probs, eta_exp)
            calc_regret('exp', t, a, r, cum_regrets, cum_rewards, regrets, run)

            # Sparsemax
            a, r = step('uni', sparsemax_probs, eta_uni * 2)
            calc_regret('uni', t, a, r, cum_regrets, cum_rewards, regrets, run)

            # Data-Driven
            a, r = step('data', data_driven_probs, eta_data)
            calc_regret('data', t, a, r, cum_regrets, cum_rewards, regrets, run)
            

        for alg in algs:
            times[alg][run] = run_times[alg]

    results = {}
    for alg in algs:
        results[f"{alg}_mean"] = regrets[alg].mean(axis=0)
        results[f"{alg}_std"] = regrets[alg].std(axis=0)
    
    time_stats = {alg: (times[alg].mean(), times[alg].std()) for alg in algs}
    
    return results, time_stats

def run_stochastic(T=5000, K=5, n_runs=20, gap=0.1,
                   eta_tsallis=0.5, eta_exp=0.5, eta_uni=0.5,
                   alpha=0.1, eta_hyp=0.5, eta_renyi=0.5, eta_data=0.5, seed=0):

    means = make_stochastic_means(K, gap=0.1)
    mu_star = means.max()
    
    # Single RNG for all runs in stochastic (preserving original behavior)
    rng_shared = np.random.default_rng(seed)
    def get_rng(run_idx):
        return rng_shared

    def get_reward_fn(t, a, rng):
        return 1.0 if rng.random() < means[a] else 0.0
        
    def get_regret_fn_factory(run):
        def calc_regret(alg, t, a, r, cum_regrets, cum_rewards, regrets, run_idx):
            cum_regrets[alg] += mu_star - means[a]
            regrets[alg][run_idx, t - 1] = cum_regrets[alg]
        return calc_regret

    results, time_stats = run_generic(T, K, n_runs, get_reward_fn, get_regret_fn_factory, get_rng,
                                      eta_exp, eta_uni, eta_hyp, eta_renyi, eta_data, alpha, "Stochastic Runs")

    return (results['tsallis_mean'], results['tsallis_std'],
            results['hyp_mean'], results['hyp_std'],
            results['renyi_mean'], results['renyi_std'],
            results['exp_mean'], results['exp_std'],
            results['uni_mean'], results['uni_std'],
            results['data_mean'], results['data_std'],
            time_stats)

def run_many_suboptimal(T=5000, K=5, n_runs=20, gap=0.1,
                   eta_tsallis=0.5, eta_exp=0.5, eta_uni=0.5,
                   alpha=0.1, eta_hyp=0.5, eta_renyi=0.5, eta_data=0.5, seed=0):

    means = make_many_suboptimal_means(K, gap=gap)
    mu_star = means.max()
    
    # Single RNG for all runs in stochastic
    rng_shared = np.random.default_rng(seed)
    def get_rng(run_idx):
        return rng_shared

    def get_reward_fn(t, a, rng):
        return 1.0 if rng.random() < means[a] else 0.0
        
    def get_regret_fn_factory(run):
        def calc_regret(alg, t, a, r, cum_regrets, cum_rewards, regrets, run_idx):
            cum_regrets[alg] += mu_star - means[a]
            regrets[alg][run_idx, t - 1] = cum_regrets[alg]
        return calc_regret

    results, time_stats = run_generic(T, K, n_runs, get_reward_fn, get_regret_fn_factory, get_rng,
                                      eta_exp, eta_uni, eta_hyp, eta_renyi, eta_data, alpha, "Many Suboptimal Runs")

    return (results['tsallis_mean'], results['tsallis_std'],
            results['hyp_mean'], results['hyp_std'],
            results['renyi_mean'], results['renyi_std'],
            results['exp_mean'], results['exp_std'],
            results['uni_mean'], results['uni_std'],
            results['data_mean'], results['data_std'],
            results['ensemble_mean'], results['ensemble_std'],
            time_stats)

def run_adversarial(T=5000, K=5, n_runs=20,
                    eta_tsallis=0.5, eta_exp=0.5, alpha=0.1, eta_uni=0.5, eta_hyp=0.5, eta_renyi=0.5, eta_data=0.5, seed=0):
 
    R = make_adversarial_rewards(T, K)
    cum_per_arm = R.sum(axis=0)
    k_star = np.argmax(cum_per_arm)
    best_cum = np.cumsum(R[:, k_star])

    def get_rng(run_idx):
        return np.random.default_rng(seed + 1000 + run_idx)

    def get_reward_fn(t, a, rng):
        return R[t-1, a]
        
    def get_regret_fn_factory(run):
        def calc_regret(alg, t, a, r, cum_regrets, cum_rewards, regrets, run_idx):
            cum_rewards[alg] += r
            regrets[alg][run_idx, t - 1] = best_cum[t - 1] - cum_rewards[alg]
        return calc_regret

    results, time_stats = run_generic(T, K, n_runs, get_reward_fn, get_regret_fn_factory, get_rng,
                                      eta_exp, eta_uni, eta_hyp, eta_renyi, eta_data, alpha, "Adversarial Runs")

    return (results['tsallis_mean'], results['tsallis_std'],
            results['hyp_mean'], results['hyp_std'],
            results['renyi_mean'], results['renyi_std'],
            results['exp_mean'], results['exp_std'],
            results['uni_mean'], results['uni_std'],
            results['data_mean'], results['data_std'],
            results['ensemble_mean'], results['ensemble_std'],
            time_stats)

# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    T = 10000
    n_runs = 10
    
    eta_tsallis = 1.0 
    eta_exp = np.sqrt(T)
    eta_uni = np.sqrt(T)
    eta_hyp = np.sqrt(T)
    eta_renyi = np.sqrt(T)
    eta_data = 2000.0 
    
    alpha = 0.5

    t = np.arange(1, T + 1)
    z = 1.96
    
    # Helper to plot
    def plot_results(ax, title, uni_m, uni_s, exp_m, exp_s, ts_m, ts_s, hyp_m, hyp_s, renyi_m, renyi_s, ens_m, ens_s):
        # Uniform / sparsemax
        se_uni = uni_s / np.sqrt(n_runs)
        ax.plot(t, uni_m, label="Uniform", color="gray")
        ax.fill_between(t, uni_m - z * se_uni, uni_m + z * se_uni, alpha=0.2, color="gray")

        # Exponential
        se_exp = exp_s / np.sqrt(n_runs)
        ax.plot(t, exp_m, label="Exponential", linestyle="--", color="green")
        ax.fill_between(t, exp_m - z * se_exp, exp_m + z * se_exp, alpha=0.2, color="green")

        # Tsallis / Pareto
        se_ts = ts_s / np.sqrt(n_runs)
        ax.plot(t, ts_m, label="Pareto (q=1/2)", linestyle="-.", color="purple")
        ax.fill_between(t, ts_m - z * se_ts, ts_m + z * se_ts, alpha=0.2, color="purple")

        # Hyperbolic
        se_hyp = hyp_s / np.sqrt(n_runs)
        ax.plot(t, hyp_m, label="Hyperbolic", linestyle=":", color="blue")
        ax.fill_between(t, hyp_m - z * se_hyp, hyp_m + z * se_hyp, alpha=0.2, color="blue")

        # Renyi
        se_renyi = renyi_s / np.sqrt(n_runs)
        ax.plot(t, renyi_m, label="Renyi", linestyle="-", color="orange")
        ax.fill_between(t, renyi_m - z * se_renyi, renyi_m + z * se_renyi, alpha=0.2, color="orange")

        # Ensemble
        # se_ens = ens_s / np.sqrt(n_runs)
        # ax.plot(t, ens_m, label="Ensemble", linestyle="-", color="black", linewidth=2.5)
        # ax.fill_between(t, ens_m - z * se_ens, ens_m + z * se_ens, alpha=0.2, color="black")

        ax.set_title(title, fontsize=50)
        ax.set_xlabel("Round t",fontsize=18)
        ax.set_ylabel("Cumulative Regret",fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend(fontsize=18)
        ax.grid(True, alpha=0.2)

    # Experiment: Many Suboptimal with different K
    K_values = [2,5,10]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    handles, labels = [], []

    for idx, K_sub in enumerate(K_values):
        ax = axes[idx]
        print(f"Running Many Suboptimal Experiment (K={K_sub})...")
        
        (many_tsallis_mean, many_tsallis_std, many_hyp_mean, many_hyp_std, 
         many_renyi_mean, many_renyi_std, many_exp_mean, many_exp_std,
         many_uni_mean, many_uni_std, _, _, 
         many_ens_mean, many_ens_std,
         time_stats_many) = run_many_suboptimal(
            T=T, K=K_sub, n_runs=n_runs, gap=0.5, alpha=alpha,
            eta_tsallis=eta_tsallis, eta_exp=eta_exp,
            eta_uni=eta_uni, eta_hyp=eta_hyp, eta_renyi=eta_renyi, eta_data=eta_data, seed=123
        )

        plot_results(ax, f"K = {K_sub}", 
                     many_uni_mean, many_uni_std, many_exp_mean, many_exp_std,
                     many_tsallis_mean, many_tsallis_std, many_hyp_mean, many_hyp_std,
                     many_renyi_mean, many_renyi_std,
                     many_ens_mean, many_ens_std)
        
        # Capture handles/labels from the first plot for the global legend
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()

        # Remove individual legend from subplot
        if ax.get_legend():
            ax.get_legend().remove()

        print(f"\nFinal Cumulative Regret (K={K_sub}):")
        print(f"Sparsemax:   {many_uni_mean[-1]:.2f}")
        print(f"Exponential: {many_exp_mean[-1]:.2f}")
        print(f"Tsallis:     {many_tsallis_mean[-1]:.2f}")
        print(f"Hyperbolic:  {many_hyp_mean[-1]:.2f}")
        print(f"Renyi:       {many_renyi_mean[-1]:.2f}")
        # print(f"Ensemble:    {many_ens_mean[-1]:.2f}")

    # Add global horizontal legend at the bottom
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=18)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3) # Make space for the legend
    plt.savefig(f"many_suboptimal_regret_combined.pdf")

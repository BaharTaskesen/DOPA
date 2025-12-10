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


# ============================================================
#  Environments
# ============================================================

def make_stochastic_means(K, gap=0.1):
    base = 0.9
    means = base - gap * np.arange(K)
    means = np.clip(means, 0.05, 0.95)
    return means


def make_adversarial_rewards(T, K):
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
                eta_exp, eta_uni, eta_hyp, eta_renyi, alpha, desc):
    
    algs = ['tsallis', 'hyp', 'renyi', 'exp', 'uni']
    regrets = {alg: np.zeros((n_runs, T)) for alg in algs}
    times = {alg: np.zeros(n_runs) for alg in algs}
    
    # Pre-calculate Tsallis eta (using T)
    eta_tsa = np.sqrt(T * (1 - alpha) / (2 * alpha)) * K ** (alpha - 1/2)

    for run in trange(n_runs, desc=desc):
        rng = get_rng_fn(run)
        
        us = {alg: np.zeros(K) for alg in algs}
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
            r_bandit = r - 1.0
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
                   alpha=0.1, eta_hyp=0.5, eta_renyi=0.5, seed=0):

    means = make_stochastic_means(K, gap=gap)
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
                                      eta_exp, eta_uni, eta_hyp, eta_renyi, alpha, "Stochastic Runs")

    return (results['tsallis_mean'], results['tsallis_std'],
            results['hyp_mean'], results['hyp_std'],
            results['renyi_mean'], results['renyi_std'],
            results['exp_mean'], results['exp_std'],
            results['uni_mean'], results['uni_std'],
            time_stats)

def run_adversarial(T=5000, K=5, n_runs=20,
                    eta_tsallis=0.5, eta_exp=0.5, alpha=0.1, eta_uni=0.5, eta_hyp=0.5, eta_renyi=0.5, seed=0):
 
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
                                      eta_exp, eta_uni, eta_hyp, eta_renyi, alpha, "Adversarial Runs")

    return (results['tsallis_mean'], results['tsallis_std'],
            results['hyp_mean'], results['hyp_std'],
            results['renyi_mean'], results['renyi_std'],
            results['exp_mean'], results['exp_std'],
            results['uni_mean'], results['uni_std'],
            time_stats)


def run_projection_benchmark(K_values, n_sims=10):
    print("Running projection benchmark...")
    times = {
        'DOPA (Uniform)': {'mean': [], 'min': [], 'max': []},
        'DOPA (Pareto)': {'mean': [], 'min': [], 'max': []}, # Tsallis
        'FTRL (Tsallis)': {'mean': [], 'min': [], 'max': []},
        'FTRL (Renyi)': {'mean': [], 'min': [], 'max': []}
    }
    
    for K in K_values:
        print(f"Benchmarking K={K}...")
        t_sparse = []
        t_tsallis = []
        t_tsallis_cvx = []
        t_renyi = []
        
        for _ in range(n_sims):
            u = np.random.rand(K)
            eta = 1.0
            
            # Sparsemax
            tic = time.perf_counter()
            sparsemax_probs(u, eta)
            t_sparse.append(time.perf_counter() - tic)
            
            # Tsallis
            tic = time.perf_counter()
            tsallis_probs(u, eta)
            t_tsallis.append(time.perf_counter() - tic)

            # Tsallis CVXPY
            tic = time.perf_counter()
            tsallis_probs_cvxpy(u, eta)
            t_tsallis_cvx.append(time.perf_counter() - tic)
            
            # Renyi
            tic = time.perf_counter()
            renyi_probs(u, eta)
            t_renyi.append(time.perf_counter() - tic)
            
        # Store stats
        for name, data in zip(['DOPA (Uniform)', 'DOPA (Pareto)', 'FTRL (Tsallis)', 'FTRL (Renyi)'], 
                              [t_sparse, t_tsallis, t_tsallis_cvx, t_renyi]):
            times[name]['mean'].append(np.mean(data))
            times[name]['min'].append(np.min(data))
            times[name]['max'].append(np.max(data))
            
    return times

def plot_projection_benchmark(times, K_values):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'DOPA (Uniform)': 'gray', 'DOPA (Pareto)': 'purple', 'FTRL (Tsallis)': 'red', 'FTRL (Renyi)': 'orange'}
    
    for alg in ['DOPA (Uniform)', 'DOPA (Pareto)', 'FTRL (Tsallis)', 'FTRL (Renyi)']:
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
    fig.savefig("projection_benchmark.pdf")
    fig.show()


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    T = 10000
    n_runs = 10

    eta_tsallis = 1.0 # calculated inside loop
    eta_exp = np.sqrt(T)
    eta_uni = np.sqrt(T)
    eta_hyp = np.sqrt(T)
    eta_renyi = np.sqrt(T)

    # seed = 1
    alpha = 0.5

    # Define ranges for K and gap
    # K_values = [2, 5, 10]
    # gap_values = [0.05, 0.1,0.2]

    # # Prepare figures: 3 rows (K), 3 columns (gap)
    # fig_sto, axes_sto = plt.subplots(3, 3, figsize=(15, 12))
    # fig_adv, axes_adv = plt.subplots(3, 3, figsize=(15, 12))

    # # Prepare figures for time
    # fig_time_sto, axes_time_sto = plt.subplots(3, 3, figsize=(15, 12))
    # fig_time_adv, axes_time_adv = plt.subplots(3, 3, figsize=(15, 12))

    # t = np.arange(1, T + 1)
    # z = 1.96  # for 95 percent conf interval

    # for i, K in enumerate(K_values):
    #     for j, gap_sto in enumerate(gap_values):
    #         print(f"Running experiments: K={K}, gap={gap_sto}")

    #         # --- Stochastic experiment ---
    #         (sto_tsallis_mean, sto_tsallis_std, sto_hyp_mean, sto_hyp_std, sto_renyi_mean, sto_renyi_std, sto_exp_mean, sto_exp_std,sto_uni_mean, sto_uni_std, time_stats_sto) = run_stochastic(
    #             T=T, K=K, n_runs=n_runs, alpha=alpha, gap=gap_sto,
    #             eta_tsallis=eta_tsallis, eta_exp=eta_exp,
    #             eta_uni=eta_uni, eta_hyp=eta_hyp, eta_renyi=eta_renyi, seed=seed
    #         )

    #         # --- Adversarial experiment ---
    #         (adv_tsallis_mean, adv_tsallis_std, adv_hyp_mean, adv_hyp_std, adv_renyi_mean, adv_renyi_std, adv_exp_mean, adv_exp_std, adv_uni_mean, adv_uni_std, time_stats_adv) = run_adversarial(
    #             T=T, K=K, n_runs=n_runs, alpha=alpha,
    #             eta_tsallis=eta_tsallis, eta_exp=eta_exp,
    #             eta_uni=eta_uni, eta_hyp=eta_hyp, eta_renyi=eta_renyi
    #         )

    #         # =======================
    #         # Plot Stochastic
    #         # =======================
    #         ax = axes_sto[i, j]

    #         # Uniform / sparsemax
    #         se_uni = sto_uni_std / np.sqrt(n_runs)
    #         ax.plot(t, sto_uni_mean, label="Sparsemax", color="gray")
    #         ax.fill_between(t, sto_uni_mean - z * se_uni, sto_uni_mean + z * se_uni, alpha=0.2, color="gray")

    #         # Exponential
    #         se_exp = sto_exp_std / np.sqrt(n_runs)
    #         ax.plot(t, sto_exp_mean, label="Exponential", linestyle="--", color="green")
    #         ax.fill_between(t, sto_exp_mean - z * se_exp, sto_exp_mean + z * se_exp, alpha=0.2, color="green")

    #         # Tsallis / Pareto
    #         se_ts = sto_tsallis_std / np.sqrt(n_runs)
    #         ax.plot(t, sto_tsallis_mean, label="Pareto (q=1/2)", linestyle="-.", color="purple")
    #         ax.fill_between(t, sto_tsallis_mean - z * se_ts, sto_tsallis_mean + z * se_ts, alpha=0.2, color="purple")

    #         # Hyperbolic
    #         se_hyp = sto_hyp_std / np.sqrt(n_runs)
    #         ax.plot(t, sto_hyp_mean, label="Hyperbolic", linestyle=":", color="blue")
    #         ax.fill_between(t, sto_hyp_mean - z * se_hyp, sto_hyp_mean + z * se_hyp, alpha=0.2, color="blue")

    #         # Renyi
    #         se_renyi = sto_renyi_std / np.sqrt(n_runs)
    #         ax.plot(t, sto_renyi_mean, label="Renyi", linestyle="-", color="orange")
    #         ax.fill_between(t, sto_renyi_mean - z * se_renyi, sto_renyi_mean + z * se_renyi, alpha=0.2, color="orange")

    #         ax.set_title(f"K={K}, gap={gap_sto}")
    #         if i == 2: ax.set_xlabel("Round t")
    #         if j == 0: ax.set_ylabel("Cumulative Regret")
    #         ax.grid(True, alpha=0.3)
    #         if i == 0 and j == 0: ax.legend()

    #         # =======================
    #         # Plot Stochastic Time
    #         # =======================
    #         ax_t = axes_time_sto[i, j]
    #         algs = ['Sparsemax', 'Exponential', 'Pareto', 'Hyperbolic', 'Renyi']
    #         means = [time_stats_sto['uni'][0], time_stats_sto['exp'][0], time_stats_sto['tsallis'][0], time_stats_sto['hyp'][0], time_stats_sto['renyi'][0]]
    #         stds = [time_stats_sto['uni'][1], time_stats_sto['exp'][1], time_stats_sto['tsallis'][1], time_stats_sto['hyp'][1], time_stats_sto['renyi'][1]]
            
    #         ax_t.bar(algs, means, yerr=stds, capsize=5, color=['gray', 'green', 'purple', 'blue', 'orange'], alpha=0.7)
    #         ax_t.set_title(f"K={K}, gap={gap_sto}")
    #         if j == 0: ax_t.set_ylabel("Time (s)")
    #         ax_t.tick_params(axis='x', rotation=45)
    #         ax_t.grid(True, axis='y', alpha=0.3)

    #         # =======================
    #         # Plot Adversarial
    #         # =======================
    #         ax = axes_adv[i, j]

    #         # Uniform / sparsemax
    #         se_uni = adv_uni_std / np.sqrt(n_runs)
    #         ax.plot(t, adv_uni_mean, label="Sparsemax", color="gray")
    #         ax.fill_between(t, adv_uni_mean - z * se_uni, adv_uni_mean + z * se_uni, alpha=0.2, color="gray")

    #         # Exponential
    #         se_exp = adv_exp_std / np.sqrt(n_runs)
    #         ax.plot(t, adv_exp_mean, label="Exponential", linestyle="--", color="green")
    #         ax.fill_between(t, adv_exp_mean - z * se_exp, adv_exp_mean + z * se_exp, alpha=0.2, color="green")

    #         # Tsallis / Pareto
    #         se_ts = adv_tsallis_std / np.sqrt(n_runs)
    #         ax.plot(t, adv_tsallis_mean, label="Pareto (q=1/2)", linestyle="-.", color="purple")
    #         ax.fill_between(t, adv_tsallis_mean - z * se_ts, adv_tsallis_mean + z * se_ts, alpha=0.2, color="purple")
            
    #         # Hyperbolic
    #         se_hyp = adv_hyp_std / np.sqrt(n_runs)
    #         ax.plot(t, adv_hyp_mean, label="Hyperbolic", linestyle=":", color="blue")
    #         ax.fill_between(t, adv_hyp_mean - z * se_hyp, adv_hyp_mean + z * se_hyp, alpha=0.2, color="blue")

    #         # Renyi
    #         se_renyi = adv_renyi_std / np.sqrt(n_runs)
    #         ax.plot(t, adv_renyi_mean, label="Renyi", linestyle="-", color="orange")
    #         ax.fill_between(t, adv_renyi_mean - z * se_renyi, adv_renyi_mean + z * se_renyi, alpha=0.2, color="orange")

    #         ax.set_title(f"K={K}")
    #         if i == 2: ax.set_xlabel("Round t")
    #         if j == 0: ax.set_ylabel("Cumulative Regret")
    #         ax.grid(True, alpha=0.3)
    #         if i == 0 and j == 0: ax.legend()

    #         # =======================
    #         # Plot Adversarial Time
    #         # =======================
    #         ax_t = axes_time_adv[i, j]
    #         algs = ['Sparsemax', 'Exponential', 'Pareto', 'Hyperbolic', 'Renyi']
    #         means = [time_stats_adv['uni'][0], time_stats_adv['exp'][0], time_stats_adv['tsallis'][0], time_stats_adv['hyp'][0], time_stats_adv['renyi'][0]]
    #         stds = [time_stats_adv['uni'][1], time_stats_adv['exp'][1], time_stats_adv['tsallis'][1], time_stats_adv['hyp'][1], time_stats_adv['renyi'][1]]
            
    #         ax_t.bar(algs, means, yerr=stds, capsize=5, color=['gray', 'green', 'purple', 'blue', 'orange'], alpha=0.7)
    #         ax_t.set_title(f"K={K}")
    #         if j == 0: ax_t.set_ylabel("Time (s)")
    #         ax_t.tick_params(axis='x', rotation=45)
    #         ax_t.grid(True, axis='y', alpha=0.3)

    # # Save Stochastic Figure
    # fig_sto.suptitle("Stochastic Environment Regret", fontsize=16)
    # fig_sto.tight_layout(rect=[0, 0.03, 1, 0.95])
    # fig_sto.savefig("stochastic_regrets.pdf")
    # fig_sto.show()

    # # Save Adversarial Figure
    # fig_adv.suptitle("Adversarial Environment Regret", fontsize=16)
    # fig_adv.tight_layout(rect=[0, 0.03, 1, 0.95])
    # fig_adv.savefig("adversarial_regrets.pdf")
    # fig_adv.show()

    # --- Projection Benchmark ---
    K_bench = np.unique(np.logspace(0, 4, num=10).astype(int))
    times_bench = run_projection_benchmark(K_values=K_bench, n_sims=10)
    plot_projection_benchmark(times_bench, K_bench)

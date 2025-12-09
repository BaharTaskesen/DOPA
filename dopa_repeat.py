import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.optimize import minimize
import cvxpy as cp
from sparsemax import Sparsemax
import torch  # needed to call Sparsemax and convert result to numpy
from tqdm import trange

############## ============================================================ ##############
#  Tsallis-FTRL probabilities (alpha = 1/2) 
############## ============================================================ ##############


def tsallis_probs(u, eta, q=0.5,n_iter=60,eps=1e-12):
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

def hyperbolic_probs(u, eta, n_iter=60, eps=1e-12):
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
    t = cp.Variable()
    # print(np.shape(term_inner))
    constraints = [cp.sum(p) == 1]
    objective = cp.Minimize(-u @ p - eta*2 * cp.log(sum(p**0.5)))

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, eps=eps_)
    p = p.value

    s = p.sum() 
    if s <= 0:
        p = np.ones(K) / K
    elif s > 1.01:
        print("Warning: renyi_probs sum > 1.01")
        print(f"sum={s}, u={u}, eta={eta}, p={p}")
        p /= s 
    else:
        p /= s

    return p


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

def run_stochastic(T=5000, K=5, n_runs=20, gap=0.1,
                   eta_tsallis=0.5, eta_exp=0.5, eta_uni=0.5,
                   alpha=0.1, eta_hyp=0.5, eta_renyi=0.5, seed=0):

    means = make_stochastic_means(K, gap=gap)
    mu_star = means.max()
    rng = np.random.default_rng(seed)

    # Store regret trajectories per run
    regrets_tsallis_runs = np.zeros((n_runs, T))
    regrets_hyp_runs = np.zeros((n_runs, T))
    regrets_renyi_runs = np.zeros((n_runs, T))
    regrets_exp_runs = np.zeros((n_runs, T))
    regrets_uni_runs = np.zeros((n_runs, T))


    for run in trange(n_runs, desc="Stochastic Runs"):
        u_tsallis = np.zeros(K)
        u_exp = np.zeros(K)
        u_uni = np.zeros(K)
        u_hyp = np.zeros(K)
        u_renyi = np.zeros(K)

        cum_reg_tsallis = 0.0
        cum_reg_exp = 0.0
        cum_reg_uni = 0.0
        cum_reg_hyp = 0.0
        cum_reg_renyi = 0.0

        for t in range(1, T + 1):
            # Tsallis
            alpha = 0.5
            eta_tsa_t = np.sqrt((T + 1) * (1 - alpha) / (2 * alpha)) * K ** (alpha - 1/2)
            p_t = tsallis_probs(u_tsallis, eta=eta_tsa_t)
            a_t = rng.choice(K, p=p_t)
            r_t = 1.0 if rng.random() < means[a_t] else 0.0
            cum_reg_tsallis += mu_star - means[a_t]
            
            # Use shifted rewards (r - 1) for stability
            r_t_bandit = r_t - 1.0
            u_tsallis[a_t] += r_t_bandit / p_t[a_t]
            regrets_tsallis_runs[run, t - 1] = cum_reg_tsallis

            # Hyperbolic
            p_h = hyperbolic_probs(u_hyp, eta=eta_hyp)
            a_h = rng.choice(K, p=p_h)
            r_h = 1.0 if rng.random() < means[a_h] else 0.0
            cum_reg_hyp += mu_star - means[a_h]
            
            # Use shifted rewards (r - 1)
            r_h_bandit = r_h - 1.0
            u_hyp[a_h] += r_h_bandit / p_h[a_h]
            regrets_hyp_runs[run, t - 1] = cum_reg_hyp

            # Renyi
            p_r = renyi_probs(u_renyi, eta=eta_renyi)
            a_r = rng.choice(K, p=p_r)
            r_r = 1.0 if rng.random() < means[a_r] else 0.0
            cum_reg_renyi += mu_star - means[a_r]
            
            # Use shifted rewards (r - 1)
            r_r_bandit = r_r - 1.0
            u_renyi[a_r] += r_r_bandit / p_r[a_r]
            regrets_renyi_runs[run, t - 1] = cum_reg_renyi

            # Exponential
            p_e = softmax_probs(u_exp, eta=eta_exp)
            p_comp = softmax(u_exp / eta_exp)
            if (p_e - p_comp).max() > 1e-8:
                print("softmax probs mismatch!")
                print(f"t={t}, run={run}, p_e={p_e}, p_comp={p_comp}")
            a_e = rng.choice(K, p=p_e)
            r_e = 1.0 if rng.random() < means[a_e] else 0.0
            cum_reg_exp += mu_star - means[a_e]
            
            # Use shifted rewards (r - 1)
            r_e_bandit = r_e - 1.0
            u_exp[a_e] += r_e_bandit / p_e[a_e]
            regrets_exp_runs[run, t - 1] = cum_reg_exp
        
            # Sparsemax
            p_u = sparsemax_probs(u_uni, eta=eta_uni * 2)
            # torch_input = torch.from_numpy((u_uni / eta_uni)).float().unsqueeze(0)
            # p_comp_sparse = Sparsemax(dim=-1)(torch_input).squeeze(0).detach().cpu().numpy()
            # if np.max(np.abs(p_u - p_comp_sparse)) > 1e-6:
            #     print("sparsemax probs mismatch!")
            #     print(f"t={t}, run={run}, p_u={p_u}, p_comp_sparse={p_comp_sparse}")
            a_u = rng.choice(K, p=p_u)
            r_u = 1.0 if rng.random() < means[a_u] else 0.0
            cum_reg_uni += mu_star - means[a_u]
            
            # Use shifted rewards (r - 1)
            r_u_bandit = r_u - 1.0
            u_uni[a_u] += r_u_bandit / p_u[a_u]
            regrets_uni_runs[run, t - 1] = cum_reg_uni

            

    # Means and stds across runs
    sto_tsallis_mean = regrets_tsallis_runs.mean(axis=0)
    sto_tsallis_std  = regrets_tsallis_runs.std(axis=0)

    sto_hyp_mean    = regrets_hyp_runs.mean(axis=0)
    sto_hyp_std     = regrets_hyp_runs.std(axis=0)

    sto_renyi_mean  = regrets_renyi_runs.mean(axis=0)
    sto_renyi_std   = regrets_renyi_runs.std(axis=0)

    sto_exp_mean     = regrets_exp_runs.mean(axis=0)
    sto_exp_std      = regrets_exp_runs.std(axis=0)

    sto_uni_mean     = regrets_uni_runs.mean(axis=0)
    sto_uni_std      = regrets_uni_runs.std(axis=0)

    return (sto_tsallis_mean, sto_tsallis_std,
            sto_hyp_mean,    sto_hyp_std,
            sto_renyi_mean,  sto_renyi_std,
            sto_exp_mean,    sto_exp_std,
            sto_uni_mean,    sto_uni_std)

def run_adversarial(T=5000, K=5, n_runs=20,
                    eta_tsallis=0.5, eta_exp=0.5, alpha=0.1, eta_uni=0.5, eta_hyp=0.5, eta_renyi=0.5, seed=0):
 
    R = make_adversarial_rewards(T, K)
    cum_per_arm = R.sum(axis=0)
    k_star = np.argmax(cum_per_arm)
    best_cum = np.cumsum(R[:, k_star])

    regrets_tsallis_runs = np.zeros((n_runs, T))
    regrets_exp_runs = np.zeros((n_runs, T))
    regrets_uni_runs = np.zeros((n_runs, T))
    regrets_hyp_runs = np.zeros((n_runs, T))
    regrets_renyi_runs = np.zeros((n_runs, T))

    # rng = np.random.default_rng(seed + 1)

    for run in trange(n_runs, desc="Adversarial Runs"):
        rng = np.random.default_rng(seed + 1000 + run)
        u_tsallis = np.zeros(K)
        u_exp = np.zeros(K)
        u_uni = np.zeros(K)
        u_hyp = np.zeros(K)
        u_renyi = np.zeros(K)

        cum_alg_tsallis = 0.0
        cum_alg_exp = 0.0
        cum_alg_uni = 0.0
        cum_alg_hyp = 0.0
        cum_alg_renyi = 0.0

        for t in range(1, T + 1):
            # Tsallis
            alpha = 0.5
            eta_tsa_t = np.sqrt((T) * (1 - alpha) / (2 * alpha)) * K ** (alpha - 1/2)
            p_t = tsallis_probs(u_tsallis, eta=eta_tsa_t)
            a_t = rng.choice(K, p=p_t)
            r_t = R[t - 1, a_t]
            cum_alg_tsallis += r_t
            
            # Use shifted rewards (r - 1)
            r_t_bandit = r_t - 1.0
            u_tsallis[a_t] += r_t_bandit / p_t[a_t]
            regrets_tsallis_runs[run, t - 1] = best_cum[t - 1] - cum_alg_tsallis

            # Softmax
            p_e = softmax_probs(u_exp, eta=eta_exp)
            a_e = rng.choice(K, p=p_e)
            r_e = R[t - 1, a_e]
            cum_alg_exp += r_e
            
            # Use shifted rewards (r - 1)
            r_e_bandit = r_e - 1.0
            u_exp[a_e] += r_e_bandit / p_e[a_e]
            regrets_exp_runs[run, t - 1] = best_cum[t - 1] - cum_alg_exp

            # Sparsemax
            # Match Tsallis q=2 scaling
            p_u = sparsemax_probs(u_uni, eta=eta_uni * 2)
            a_u = rng.choice(K, p=p_u)
            r_u = R[t - 1, a_u]
            cum_alg_uni += r_u
            
            # Use shifted rewards (r - 1)
            r_u_bandit = r_u - 1.0
            u_uni[a_u] += r_u_bandit / p_u[a_u]
            regrets_uni_runs[run, t - 1] = best_cum[t - 1] - cum_alg_uni

            # Hyperbolic
            p_h = hyperbolic_probs(u_hyp, eta=eta_hyp)
            a_h = rng.choice(K, p=p_h)
            r_h = R[t - 1, a_h]
            cum_alg_hyp += r_h
            
            # Use shifted rewards (r - 1)
            r_h_bandit = r_h - 1.0
            u_hyp[a_h] += r_h_bandit / p_h[a_h]
            regrets_hyp_runs[run, t - 1] = best_cum[t - 1] - cum_alg_hyp

            # Renyi
            p_r = renyi_probs(u_renyi, eta=eta_renyi)
            a_r = rng.choice(K, p=p_r)
            r_r = R[t - 1, a_r]
            cum_alg_renyi += r_r
            
            # Use shifted rewards (r - 1)
            r_r_bandit = r_r - 1.0
            u_renyi[a_r] += r_r_bandit / p_r[a_r]
            regrets_renyi_runs[run, t - 1] = best_cum[t - 1] - cum_alg_renyi

    adv_tsallis_mean = regrets_tsallis_runs.mean(axis=0)
    adv_tsallis_std  = regrets_tsallis_runs.std(axis=0)

    adv_hyp_mean    = regrets_hyp_runs.mean(axis=0)
    adv_hyp_std     = regrets_hyp_runs.std(axis=0)

    adv_exp_mean     = regrets_exp_runs.mean(axis=0)
    adv_exp_std      = regrets_exp_runs.std(axis=0)

    adv_uni_mean     = regrets_uni_runs.mean(axis=0)
    adv_uni_std      = regrets_uni_runs.std(axis=0)

    adv_renyi_mean   = regrets_renyi_runs.mean(axis=0)
    adv_renyi_std    = regrets_renyi_runs.std(axis=0)

    return (adv_tsallis_mean, adv_tsallis_std,
            adv_hyp_mean,    adv_hyp_std,
            adv_renyi_mean,  adv_renyi_std,
            adv_exp_mean,    adv_exp_std,
            adv_uni_mean,    adv_uni_std)


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

    seed = 1234
    alpha = 0.5

    # Define ranges for K and gap
    K_values = [2, 5, 10]
    gap_values = [0.05, 0.1,0.2]

    # Prepare figures: 3 rows (K), 3 columns (gap)
    fig_sto, axes_sto = plt.subplots(3, 3, figsize=(15, 12))
    fig_adv, axes_adv = plt.subplots(3, 3, figsize=(15, 12))

    t = np.arange(1, T + 1)
    z = 1.96  # for 95 percent conf interval

    for i, K in enumerate(K_values):
        for j, gap_sto in enumerate(gap_values):
            print(f"Running experiments: K={K}, gap={gap_sto}")

            # --- Stochastic experiment ---
            (sto_tsallis_mean, sto_tsallis_std, sto_hyp_mean, sto_hyp_std, sto_renyi_mean, sto_renyi_std, sto_exp_mean, sto_exp_std,sto_uni_mean, sto_uni_std) = run_stochastic(
                T=T, K=K, n_runs=n_runs, alpha=alpha, gap=gap_sto,
                eta_tsallis=eta_tsallis, eta_exp=eta_exp,
                eta_uni=eta_uni, eta_hyp=eta_hyp, eta_renyi=eta_renyi, seed=seed
            )

            # --- Adversarial experiment ---
            (adv_tsallis_mean, adv_tsallis_std, adv_hyp_mean, adv_hyp_std, adv_renyi_mean, adv_renyi_std, adv_exp_mean, adv_exp_std, adv_uni_mean, adv_uni_std) = run_adversarial(
                T=T, K=K, n_runs=n_runs, alpha=alpha,
                eta_tsallis=eta_tsallis, eta_exp=eta_exp,
                eta_uni=eta_uni, eta_hyp=eta_hyp, eta_renyi=eta_renyi
            )

            # =======================
            # Plot Stochastic
            # =======================
            ax = axes_sto[i, j]

            # Uniform / sparsemax
            se_uni = sto_uni_std / np.sqrt(n_runs)
            ax.plot(t, sto_uni_mean, label="Sparsemax", color="gray")
            ax.fill_between(t, sto_uni_mean - z * se_uni, sto_uni_mean + z * se_uni, alpha=0.2, color="gray")

            # Exponential
            se_exp = sto_exp_std / np.sqrt(n_runs)
            ax.plot(t, sto_exp_mean, label="Exponential", linestyle="--", color="green")
            ax.fill_between(t, sto_exp_mean - z * se_exp, sto_exp_mean + z * se_exp, alpha=0.2, color="green")

            # Tsallis / Pareto
            se_ts = sto_tsallis_std / np.sqrt(n_runs)
            ax.plot(t, sto_tsallis_mean, label="Pareto (q=1/2)", linestyle="-.", color="purple")
            ax.fill_between(t, sto_tsallis_mean - z * se_ts, sto_tsallis_mean + z * se_ts, alpha=0.2, color="purple")

            # Hyperbolic
            se_hyp = sto_hyp_std / np.sqrt(n_runs)
            ax.plot(t, sto_hyp_mean, label="Hyperbolic", linestyle=":", color="blue")
            ax.fill_between(t, sto_hyp_mean - z * se_hyp, sto_hyp_mean + z * se_hyp, alpha=0.2, color="blue")

            # Renyi
            se_renyi = sto_renyi_std / np.sqrt(n_runs)
            ax.plot(t, sto_renyi_mean, label="Renyi", linestyle="-", color="orange")
            ax.fill_between(t, sto_renyi_mean - z * se_renyi, sto_renyi_mean + z * se_renyi, alpha=0.2, color="orange")

            ax.set_title(f"K={K}, gap={gap_sto}")
            if i == 2: ax.set_xlabel("Round t")
            if j == 0: ax.set_ylabel("Cumulative Regret")
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0: ax.legend()

            # =======================
            # Plot Adversarial
            # =======================
            ax = axes_adv[i, j]

            # Uniform / sparsemax
            se_uni = adv_uni_std / np.sqrt(n_runs)
            ax.plot(t, adv_uni_mean, label="Sparsemax", color="gray")
            ax.fill_between(t, adv_uni_mean - z * se_uni, adv_uni_mean + z * se_uni, alpha=0.2, color="gray")

            # Exponential
            se_exp = adv_exp_std / np.sqrt(n_runs)
            ax.plot(t, adv_exp_mean, label="Exponential", linestyle="--", color="green")
            ax.fill_between(t, adv_exp_mean - z * se_exp, adv_exp_mean + z * se_exp, alpha=0.2, color="green")

            # Tsallis / Pareto
            se_ts = adv_tsallis_std / np.sqrt(n_runs)
            ax.plot(t, adv_tsallis_mean, label="Pareto (q=1/2)", linestyle="-.", color="purple")
            ax.fill_between(t, adv_tsallis_mean - z * se_ts, adv_tsallis_mean + z * se_ts, alpha=0.2, color="purple")
            
            # Hyperbolic
            se_hyp = adv_hyp_std / np.sqrt(n_runs)
            ax.plot(t, adv_hyp_mean, label="Hyperbolic", linestyle=":", color="blue")
            ax.fill_between(t, adv_hyp_mean - z * se_hyp, adv_hyp_mean + z * se_hyp, alpha=0.2, color="blue")

            # Renyi
            se_renyi = adv_renyi_std / np.sqrt(n_runs)
            ax.plot(t, adv_renyi_mean, label="Renyi", linestyle="-", color="orange")
            ax.fill_between(t, adv_renyi_mean - z * se_renyi, adv_renyi_mean + z * se_renyi, alpha=0.2, color="orange")

            ax.set_title(f"K={K}")
            if i == 2: ax.set_xlabel("Round t")
            if j == 0: ax.set_ylabel("Cumulative Regret")
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0: ax.legend()

    # Save Stochastic Figure
    fig_sto.suptitle("Stochastic Environment Regret", fontsize=16)
    fig_sto.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_sto.savefig("stochastic_regrets.pdf")
    fig_sto.show()

    # Save Adversarial Figure
    fig_adv.suptitle("Adversarial Environment Regret", fontsize=16)
    fig_adv.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_adv.savefig("adversarial_regrets.pdf")
    fig_adv.show()

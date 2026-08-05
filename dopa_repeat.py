import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{newtxtext,newtxmath}',
    'legend.fontsize': 17,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})
from matplotlib.legend_handler import HandlerTuple
from scipy.special import softmax
from scipy.optimize import minimize
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
                   alpha=0.1, seed=0):

    means = make_stochastic_means(K, gap=gap)
    mu_star = means.max()
    rng = np.random.default_rng(seed)

    # Store regret trajectories per run
    regrets_tsallis_runs = np.zeros((n_runs, T))
    regrets_exp_runs = np.zeros((n_runs, T))
    regrets_uni_runs = np.zeros((n_runs, T))


    for run in trange(n_runs, desc="Stochastic Runs"):
        u_tsallis = np.zeros(K)
        u_exp = np.zeros(K)
        u_uni = np.zeros(K)

        cum_reg_tsallis = 0.0
        cum_reg_exp = 0.0
        cum_reg_uni = 0.0

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

    sto_exp_mean     = regrets_exp_runs.mean(axis=0)
    sto_exp_std      = regrets_exp_runs.std(axis=0)

    sto_uni_mean     = regrets_uni_runs.mean(axis=0)
    sto_uni_std      = regrets_uni_runs.std(axis=0)

    return (sto_tsallis_mean, sto_tsallis_std,
            sto_exp_mean,    sto_exp_std,
            sto_uni_mean,    sto_uni_std)

def run_adversarial(T=5000, K=5, n_runs=20,
                    eta_tsallis=0.5, eta_exp=0.5, alpha=0.1, eta_uni=0.5, seed=0):
 
    R = make_adversarial_rewards(T, K)
    cum_per_arm = R.sum(axis=0)
    k_star = np.argmax(cum_per_arm)
    best_cum = np.cumsum(R[:, k_star])

    regrets_tsallis_runs = np.zeros((n_runs, T))
    regrets_exp_runs = np.zeros((n_runs, T))
    regrets_uni_runs = np.zeros((n_runs, T))

    # rng = np.random.default_rng(seed + 1)

    for run in trange(n_runs, desc="Adversarial Runs"):
        rng = np.random.default_rng(seed + 1000 + run)
        u_tsallis = np.zeros(K)
        u_exp = np.zeros(K)
        u_uni = np.zeros(K)

        cum_alg_tsallis = 0.0
        cum_alg_exp = 0.0
        cum_alg_uni = 0.0

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

    adv_tsallis_mean = regrets_tsallis_runs.mean(axis=0)
    adv_tsallis_std  = regrets_tsallis_runs.std(axis=0)

    adv_exp_mean     = regrets_exp_runs.mean(axis=0)
    adv_exp_std      = regrets_exp_runs.std(axis=0)

    adv_uni_mean     = regrets_uni_runs.mean(axis=0)
    adv_uni_std      = regrets_uni_runs.std(axis=0)

    return (adv_tsallis_mean, adv_tsallis_std,
            adv_exp_mean,    adv_exp_std,
            adv_uni_mean,    adv_uni_std)


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    T = 10000
    n_runs = 100
    eta_tsallis = 1.0 # calculated inside loop
    eta_exp = np.sqrt(T)
    eta_uni = np.sqrt(T)
    seed = 1234
    alpha = 0.5

    # Define ranges for K and gap
    K_values = [2, 5, 10]
    gap_values = [0.05, 0.1, 0.2]

    t = np.arange(1, T + 1)
    z = 1.96  # for 95 percent conf interval

    def plot_stoch_panel(ax, res, title, show_ylabel):
        """Plot one stochastic subplot (band+line for each method); returns legend handles/labels."""
        uni_m, uni_s, exp_m, exp_s, ts_m, ts_s = res
        legend_handles = []
        legend_labels = []

        def plot_with_band(mean, std, label, color, linestyle="-"):
            se = std / np.sqrt(n_runs)
            band = ax.fill_between(t, mean - z * se, mean + z * se, alpha=0.2, color=color)
            line, = ax.plot(t, mean, color=color, linestyle=linestyle)
            legend_handles.append((band, line))
            legend_labels.append(label)

        plot_with_band(uni_m, uni_s, "Uniform", "gray")
        plot_with_band(exp_m, exp_s, "Exponential", "green", "--")
        plot_with_band(ts_m, ts_s, r"Pareto ($\alpha=1/2$)", "purple", "-.")

        ax.set_title(title, fontsize=25)
        ax.set_xlabel(r"Round $t$", fontsize=20)
        if show_ylabel:
            ax.set_ylabel("Regret", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.grid(True, alpha=0.2)
        return legend_handles, legend_labels

    # =======================
    # Stochastic: combined 1x3 figure (K=2,5,10) per gap, screenshot style.
    # Also cache K=5 results to build a per-gap (delta) figure afterwards.
    # =======================
    k5_by_gap = {}
    for gap_sto in gap_values:
        fig_sto, axes = plt.subplots(1, 3, figsize=(18, 5))
        handles, labels = [], []

        for idx, K in enumerate(K_values):
            print(f"Running stochastic experiment: K={K}, gap={gap_sto}")

            (sto_tsallis_mean, sto_tsallis_std, sto_exp_mean, sto_exp_std,sto_uni_mean, sto_uni_std) = run_stochastic(
                T=T, K=K, n_runs=n_runs, alpha=alpha, gap=gap_sto,
                eta_tsallis=eta_tsallis, eta_exp=eta_exp,
                eta_uni=eta_uni, seed=seed
            )

            res = (sto_uni_mean, sto_uni_std, sto_exp_mean, sto_exp_std, sto_tsallis_mean, sto_tsallis_std)
            if K == 5:
                k5_by_gap[gap_sto] = res

            lh, ll = plot_stoch_panel(axes[idx], res, f"$K = {K}$", show_ylabel=(idx == 0))
            if idx == 0:
                handles, labels = lh, ll

        fig_sto.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0),
                       ncol=3, fontsize=25, handler_map={tuple: HandlerTuple(ndivide=1)})
        fig_sto.tight_layout()
        fig_sto.subplots_adjust(bottom=0.28)  # room for the shared legend
        fig_sto.savefig(f"stoch_gap{gap_sto}.pdf")
        plt.close(fig_sto)

    # =======================
    # Stochastic (K=5): combined 1x3 figure over gaps (delta = 0.05, 0.1, 0.2)
    # =======================
    fig_k5, axes = plt.subplots(1, 3, figsize=(18, 5))
    handles, labels = [], []
    for idx, gap_sto in enumerate(gap_values):
        lh, ll = plot_stoch_panel(axes[idx], k5_by_gap[gap_sto],
                                  rf"$h = {gap_sto}$", show_ylabel=(idx == 0))
        if idx == 0:
            handles, labels = lh, ll

    fig_k5.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0),
                  ncol=3, fontsize=25, handler_map={tuple: HandlerTuple(ndivide=1)})
    fig_k5.tight_layout()
    fig_k5.subplots_adjust(bottom=0.28)
    fig_k5.savefig("stoch_K5_delta.pdf")
    plt.close(fig_k5)

    for K in K_values:
        # =======================
        # Adversarial: one PDF per K (gap-independent), no title
        # =======================
        print(f"Running adversarial experiment: K={K}")
        (adv_tsallis_mean, adv_tsallis_std, adv_exp_mean, adv_exp_std, adv_uni_mean, adv_uni_std) = run_adversarial(
            T=T, K=K, n_runs=n_runs, alpha=alpha,
            eta_tsallis=eta_tsallis, eta_exp=eta_exp,
            eta_uni=eta_uni
        )

        fig_adv, ax = plt.subplots(figsize=(7, 5))

        # Uniform / sparsemax
        se_uni = adv_uni_std / np.sqrt(n_runs)
        ax.plot(t, adv_uni_mean, label="Uniform", color="gray")
        ax.fill_between(t, adv_uni_mean - z * se_uni, adv_uni_mean + z * se_uni, alpha=0.2, color="gray")

        # Exponential
        se_exp = adv_exp_std / np.sqrt(n_runs)
        ax.plot(t, adv_exp_mean, label="Exponential", linestyle="--", color="green")
        ax.fill_between(t, adv_exp_mean - z * se_exp, adv_exp_mean + z * se_exp, alpha=0.2, color="green")

        # Tsallis / Pareto
        se_ts = adv_tsallis_std / np.sqrt(n_runs)
        ax.plot(t, adv_tsallis_mean, label=r"Pareto ($\alpha=1/2$)", linestyle="-.", color="purple")
        ax.fill_between(t, adv_tsallis_mean - z * se_ts, adv_tsallis_mean + z * se_ts, alpha=0.2, color="purple")

        ax.set_xlabel(r"Round $t$")
        ax.set_ylabel("Regret", fontsize=20)
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig_adv.tight_layout()
        fig_adv.savefig(f"adv_K{K}.pdf")
        plt.close(fig_adv)

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{newtxtext,newtxmath}',
    'legend.fontsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
})
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
    if K == 1:
        return np.ones(1)

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



def tsallis_probs_cvxpy(u, eta):
    u = np.asarray(u, dtype=float)
    K = u.shape[0]
    if K == 1:
        return np.ones(1)
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


def tsallis_probs_cvxpy_alpha(u, eta, alpha):
    """General Tsallis-FTRL via MOSEK interior point solver (power cone).

    Maximizes u @ p + 2 * eta * sum(t - p) subject to t_i <= p_i^alpha,
    modeled with the 3D power cone. alpha = 1/2 recovers tsallis_probs_cvxpy.
    """
    u = np.asarray(u, dtype=float)
    K = u.shape[0]
    if K == 1:
        return np.ones(1)
    p = cp.Variable(K, nonneg=True)
    t = cp.Variable(K, nonneg=True)

    constraints = [
        cp.sum(p) == 1,
        p <= 1,
        # t_i <= p_i^alpha * 1^(1-alpha)
        cp.PowCone3D(p, np.ones(K), t, alpha * np.ones(K)),
    ]

    objective = cp.Maximize(u @ p + 2 * eta * cp.sum(t - p))

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.MOSEK)
    except:
        prob.solve(solver=cp.SCS)
    return p.value

def quadratic_probs_cvxpy(u, eta):
    """Quadratic (Euclidean) FTRL via MOSEK interior point solver.

    Maximizes u @ p - (1/(2*eta)) ||p||^2 over the simplex, i.e. the
    Euclidean projection underlying sparsemax.
    """
    u = np.asarray(u, dtype=float)
    K = u.shape[0]
    p = cp.Variable(K, nonneg=True)

    constraints = [
        cp.sum(p) == 1,
        p <= 1,
    ]

    objective = cp.Maximize(u @ p - (1.0 / (2.0 * eta)) * cp.sum_squares(p))

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.MOSEK)
    except:
        prob.solve(solver=cp.SCS)
    return p.value


def hybrid_probs_cvxpy(u, eta):
    """FTRL with the hybrid regularizer via MOSEK.

    Hybrid regularizer (per round t, with eta playing the role of sqrt(t)):

        R(p) = sum_k -eta * ( 2 (sqrt(p_k) - p_k) + (p_k - 1) log(1 - p_k) )

    so the FTRL step is

        max_{p in simplex}  <u,p> + eta * sum_k ( 2 (sqrt(p_k) - p_k) + (p_k - 1) log(1 - p_k) ).

    Note (p_k - 1) log(1 - p_k) = -(1 - p_k) log(1 - p_k) = entr(1 - p_k), so the
    second term is a Shannon entropy of the complement 1 - p_k (exp cone), while
    sqrt(p_k) - p_k is the 1/2-Tsallis term (second-order cone), normalized as in
    tsallis_probs_cvxpy so that its stationarity condition gives the generator
    inverse G2^{-1}(rho) = 2 - 1/sqrt(rho). Same objective as
    hybrid_probs_bisection, so the two agree.
    """
    u = np.asarray(u, dtype=float)
    K = u.shape[0]
    if K == 1:
        return np.ones(1)
    p = cp.Variable(K, nonneg=True)
    z = cp.Variable(K)  # z_i <= sqrt(p_i)

    constraints = [
        cp.sum(p) == 1,
        p <= 1,
    ]

    # z_i <= sqrt(p_i):  || [2 z_i, 1 - p_i] ||_2 <= 1 + p_i
    X = cp.vstack([cp.reshape(2 * z, (1, K)), cp.reshape(1 - p, (1, K))])
    constraints.append(cp.SOC(1 + p, X, axis=0))

    objective = cp.Maximize(u @ p + 2 * eta * cp.sum(z - p) + eta * cp.sum(cp.entr(1 - p)))

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
        return 0.5 * s

    def F_k(s):
        # 1.0 - F(-s/eta)
        val = 0.5 * (2.0 + s / float(eta))
        return np.clip(val, 0.0, 1.0)

    def F_inv(t):
        return 2.0 * t

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

# ==============================
# Hybrid (Tsallis 1/2 + Shannon) via inexact bisection (Algorithms 4 & 5)
# ==============================

def hybrid_probs_bisection(u, eta, eps=1e-8, rho_eps=1e-15):
    """FTRL with the hybrid regularizer, via the inexact-bisection oracle of
    Algorithms 4 (outer) and 5 (inner), faithful to the updated pseudocode.

    Frechet marginal generator of Corollary 3 (Definition 3 with eta_k = 1,
    gamma_1 = gamma_2 = gamma_t = sqrt(t) = eta here):

        G1(s) = 1 - exp(-(s+1))    =>  G1^{-1}(rho) = -1 - log(1 - rho),
        G2(s) = (2 - s)^{-2} (s<2) =>  G2^{-1}(rho) = 2 - 1 / sqrt(rho),
        M(rho) = gamma_1 G1^{-1}(rho) + gamma_2 G2^{-1}(rho).

    Algorithm 5(s, N): N bisection steps on rho in [0, 1] to invert M at the
    evaluation point s (M is increasing), returning the marginal bracket
    [ell_k, ell_bar_k] = [max{0, 1 - rho_bar}, max{0, 1 - rho_under}].

    Algorithm 4: bisection on the simplex multiplier tau over the explicit
    bracket [tau_under, tau_bar] with tau_(bar/under) = (min/max)_k{-u_k} + M(1/K),
    calling Algorithm 5 at s_k = u_k + tau, and returning the DOPA projection
    p_k = 1 - ell_bar_k + (1/K)(1 - sum_j (1 - ell_bar_j)).

    delta(.) is the accuracy modulus; delta(x) = x is used here, so the outer
    loop runs ceil(log2((tau_bar - tau_under)/(eps/2))) steps.
    """
    u = np.asarray(u, dtype=float)
    K = u.shape[0]
    if K == 1:
        return np.ones(1)
    g1 = g2 = eta   # gamma_1 = gamma_2 = gamma_t = eta

    def M_sum(rho):
        rho = np.clip(rho, rho_eps, 1.0 - rho_eps)
        return g1 * (-1.0 - np.log1p(-rho)) + g2 * (2.0 - 1.0 / np.sqrt(rho))

    # Algorithm 5: N = ceil(log2(8K/eps)).
    N = max(1, int(np.ceil(np.log2(8.0 * K / eps))))

    def alg5(s):
        # Vectorized over k: bisect rho in [0, 1] to solve M(rho) = s.
        rho_bar = np.ones(K)
        rho_und = np.zeros(K)
        for _ in range(N):
            rho = 0.5 * (rho_bar + rho_und)
            big = M_sum(rho) > s          # M increasing => root below rho
            rho_bar = np.where(big, rho, rho_bar)
            rho_und = np.where(big, rho_und, rho)
        # [ell_k, ell_bar_k] = [max{0, 1 - rho_bar}, max{0, 1 - rho_under}]
        return np.maximum(0.0, 1.0 - rho_bar), np.maximum(0.0, 1.0 - rho_und)

    def project(ell_bar):
        # p_k = 1 - ell_bar_k + (1/K)(1 - sum_j (1 - ell_bar_j))
        m = 1.0 - ell_bar
        p = m + (1.0 - m.sum()) / K
        p = np.maximum(p, 0.0)
        s = p.sum()
        return p / s if s > 0 else np.ones(K) / K

    # Algorithm 4: explicit tau bracket.
    M_1K = M_sum(1.0 / K)
    tau_bar = np.max(-u) + M_1K
    tau_und = np.min(-u) + M_1K

    # Outer iteration count: ceil(log2((tau_bar - tau_und)/delta(eps/2))), delta(x)=x.
    width = tau_bar - tau_und
    n_outer = max(1, int(np.ceil(np.log2(width / (0.5 * eps))))) if width > 0 else 1

    ell, ell_bar = alg5(u + tau_und)
    for _ in range(n_outer):
        tau = 0.5 * (tau_bar + tau_und)
        ell, ell_bar = alg5(u + tau)                 # [ell_k, ell_bar_k]
        if np.sum(1.0 - ell_bar) > 1.0:              # sum_k (1 - ell_bar_k) > 1
            tau_bar = tau
        elif np.sum(1.0 - ell) < 1.0:                # sum_k (1 - ell_k) < 1
            tau_und = tau
        else:
            return project(ell_bar)
    # Fallback: evaluate at tau_under (Algorithm 5(u_k + tau_under, N)).
    _, ell_bar = alg5(u + tau_und)
    return project(ell_bar)

def run_projection_benchmark(K_values, n_sims=10):
    print("Running projection benchmark...")
    names = [
        'DOPA (Pareto 1/2)',
        'FTRL (Tsallis 1/2)',
        'DOPA (Pareto 1/4)',
        'FTRL (Tsallis 1/4)',
        'DOPA (Pareto 3/4)',
        'FTRL (Tsallis 3/4)',
        'DOPA (Hybrid)',
        'FTRL (Hybrid)',
    ]
    times = {name: {'mean': [], 'min': [], 'max': []} for name in names}
    errors = {
        'DOPA (Pareto 1/2)': {'mean': [], 'min': [], 'max': []}
    }

    for K in K_values:
        print(f"Benchmarking K={K}...")
        t_tsallis_12 = []
        t_tsallis_cvx_12 = []
        t_tsallis_14 = []
        t_tsallis_cvx_14 = []
        t_tsallis_34 = []
        t_tsallis_cvx_34 = []
        t_hybrid_bis = []
        t_hybrid_cvx = []
        e_tsallis = []

        for _ in range(n_sims):
            u = np.random.rand(K)
            eta = 1.0

            # Pair 1: DOPA (Pareto 1/2, bisection) vs FTRL (Tsallis 1/2, MOSEK)
            tic = time.perf_counter()
            p_ts = tsallis_probs(u, eta, q=0.5)
            t_tsallis_12.append(time.perf_counter() - tic)

            tic = time.perf_counter()
            p_ts_cvx = tsallis_probs_cvxpy(u, eta)
            t_tsallis_cvx_12.append(time.perf_counter() - tic)

            if p_ts_cvx is not None:
                e_tsallis.append(np.sum(np.abs(p_ts - p_ts_cvx)))
            else:
                e_tsallis.append(0.0)

            # Pair 2: DOPA (Pareto 1/4, bisection) vs FTRL (Tsallis 1/4, MOSEK)
            tic = time.perf_counter()
            tsallis_probs(u, eta, q=0.25)
            t_tsallis_14.append(time.perf_counter() - tic)

            tic = time.perf_counter()
            tsallis_probs_cvxpy_alpha(u, eta, 0.25)
            t_tsallis_cvx_14.append(time.perf_counter() - tic)

            # Pair 3: DOPA (Pareto 3/4, bisection) vs FTRL (Tsallis 3/4, MOSEK)
            tic = time.perf_counter()
            tsallis_probs(u, eta, q=0.75)
            t_tsallis_34.append(time.perf_counter() - tic)

            tic = time.perf_counter()
            tsallis_probs_cvxpy_alpha(u, eta, 0.75)
            t_tsallis_cvx_34.append(time.perf_counter() - tic)

            # Pair 4: DOPA (Hybrid, inexact bisection) vs FTRL (Hybrid, MOSEK)
            tic = time.perf_counter()
            hybrid_probs_bisection(u, eta)
            t_hybrid_bis.append(time.perf_counter() - tic)

            tic = time.perf_counter()
            hybrid_probs_cvxpy(u, eta)
            t_hybrid_cvx.append(time.perf_counter() - tic)

        # Store stats
        for name, data in zip(names,
                              [t_tsallis_12, t_tsallis_cvx_12,
                               t_tsallis_14, t_tsallis_cvx_14, t_tsallis_34, t_tsallis_cvx_34,
                               t_hybrid_bis, t_hybrid_cvx]):
            times[name]['mean'].append(np.mean(data))
            times[name]['min'].append(np.min(data))
            times[name]['max'].append(np.max(data))

        errors['DOPA (Pareto 1/2)']['mean'].append(np.mean(e_tsallis))
        errors['DOPA (Pareto 1/2)']['min'].append(np.min(e_tsallis))
        errors['DOPA (Pareto 1/2)']['max'].append(np.max(e_tsallis))

    return times, errors

def plot_projection_benchmark(times, K_values):
    # Each DOPA bisection method vs its FTRL MOSEK counterpart.
    pairs = [
        ('DOPA (Pareto 1/2)', 'FTRL (Tsallis 1/2)'),
        ('DOPA (Pareto 1/4)', 'FTRL (Tsallis 1/4)'),
        ('DOPA (Pareto 3/4)', 'FTRL (Tsallis 3/4)'),
        ('DOPA (Hybrid)', 'FTRL (Hybrid)'),
    ]

    ncols = 2
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    axes = axes.ravel()

    # Shared y-axis limits across all subplots.
    all_min = min(np.min(times[alg]['min']) for pair in pairs for alg in pair if np.min(times[alg]['min']) > 0)
    all_max = max(np.max(times[alg]['max']) for pair in pairs for alg in pair)
    ylim = (all_min * 0.5, all_max * 2.0)

    for ax, (dopa, ftrl) in zip(axes, pairs):
        for alg, color in [(dopa, '#C0498A'), (ftrl, '#7A7A7A')]:
            means = np.array(times[alg]['mean'])
            mins = np.array(times[alg]['min'])
            maxs = np.array(times[alg]['max'])
            ax.plot(K_values, means, label=alg, color=color, linewidth=2)
            ax.fill_between(K_values, mins, maxs, color=color, alpha=0.2)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(ylim)
        ax.set_xlabel(r'\# of arms ($K$)', fontsize=20)
        ax.set_ylabel('Execution time (s)', fontsize=20)
        ax.legend(fontsize=16)
        ax.grid(True, which="both", ls="-", alpha=0.3)

    # Hide any unused axes (e.g. the 6th slot when there are 5 pairs).
    for ax in axes[len(pairs):]:
        ax.axis('off')

    fig.tight_layout()
    fig.savefig("dopa_time.pdf")
    fig.show()

def plot_quality_benchmark(errors, K_values):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'DOPA (Pareto 1/2)': 'purple'}

    alg = 'DOPA (Pareto 1/2)'
    means = np.array(errors[alg]['mean'])
    mins = np.array(errors[alg]['min'])
    maxs = np.array(errors[alg]['max'])
    
    ax.plot(K_values, means, label=f"{alg} vs CVXPY", color=colors[alg], linewidth=2, marker='o')
    ax.fill_between(K_values, mins, maxs, color=colors[alg], alpha=0.2)
        
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'\# of arms ($K$)', fontsize=20)
    ax.set_ylabel('L1 Error', fontsize=20)
    ax.legend(fontsize=16)
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

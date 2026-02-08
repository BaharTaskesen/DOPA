import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. Environment & Utilities
# ==========================================

class LinearBanditEnv:
    def __init__(self, mode='hard', n_arms=3, dim=2, noise_std=0.5, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
            
        self.dim = dim
        self.n_arms = n_arms
        self.noise_std = noise_std
        
        if mode == 'random':
            arms = np.random.randn(n_arms, dim)
            self.arms = arms / np.linalg.norm(arms, axis=1, keepdims=True)
            self.theta = np.zeros(dim)
            self.theta[0] = 1.0 
            
        elif mode == 'hard':
            # "Soare-like" Instance
            # High dimension, info aligned on axes
            self.dim = 3
            self.n_arms = 4
            d = self.dim
            
            eps = 0.1
            
            # 1. Best Arm (aligned with theta)
            a1 = np.array([1.0, 0.0, 0.0])
            
            # 2. Competitor (close to best, distinct in dim 2)
            # Gap is small (~eps/2), but requires resolving dim 2
            a2 = np.array([1.0 - eps, 0.5, 0.0])
            
            # 3. Informative for Dim 2 (High penalty but resolves distinction between 1 and 2)
            # Regret is large, but norm in dim 2 is huge.
            a3 = np.array([0.0, 10.0, 0.0])
            
            # 4. Decoy (Orthogonal, useless)
            a4 = np.array([0.2, 0.0, 1.0])

            self.arms = np.array([a1, a2, a3, a4])
            self.theta = np.array([1.0, 0.0, 0.0]) 

        elif mode == 'paper':
            """
            ### Linear Bandits Setup (Mode: "Paper")

            We consider a stochastic linear bandit problem with dimension $d=5$ and a set of $K=10$ arms. The configuration ensures that the expected rewards for all arms lie strictly within the interval $[0.1, 0.9]$. The environment is constructed as follows:

            **1. Latent Parameter Generation:**
            First, a latent vector $\hat{\theta} \in \mathbb{R}^{d-1}$ (where $d-1=4$) is drawn from a standard multivariate normal distribution:
            $$ \hat{\theta} \sim \mathcal{N}(0, I_{d-1}) $$
            The true unknown parameter vector $\theta_* \in \mathbb{R}^d$ is constructed by concatenating $\hat{\theta}$ with a fixed bias component:
            $$ \theta_* = \\begin{bmatrix} \hat{\theta} \\\\ 1 \\end{bmatrix} $$

            **2. Arm Feature Generation:**
            For each arm $k \in \{1, \dots, K\}$, a raw feature vector $\hat{x}_k \in \mathbb{R}^{d-1}$ is drawn independently from a standard normal distribution:
            $$ \hat{x}_k \sim \mathcal{N}(0, I_{d-1}) $$

            **3. Reward Scaling (Normalization):**
            To ensure the expected rewards $\mu_k = \langle x_k, \theta_* \rangle$ fall within $[0.1, 0.9]$, we perform an affine transformation. Let $z_k = \langle \hat{x}_k, \hat{\theta} \rangle$ be the raw dot product for arm $k$. We define the range of these raw values as:
            $$ z_{\min} = \min_{k} z_k, \quad z_{\max} = \max_{k} z_k $$
            We compute scaling scalars $a, b \in \mathbb{R}$ to map the interval $[z_{\min}, z_{\max}]$ to $[0.1, 0.9]$:
            $$ a = \\frac{0.8}{z_{\max} - z_{\min}}, \quad b = 0.9 - a \cdot z_{\max} $$

            **4. Final Arm Construction:**
            The final feature vector $x_k \in \mathbb{R}^d$ for each arm $k$ is constructed as:
            $$ x_k = \\begin{bmatrix} a \cdot \hat{x}_k \\\\ b \\end{bmatrix} $$

            **5. Reward Process:**
            At time $t$, if the agent pulls arm $k$, the observed reward is:
            $$ r_t = \langle x_k, \theta_* \rangle + \eta_t $$
            where $\eta_t \sim \mathcal{N}(0, \sigma^2)$ is Gaussian noise with $\sigma=0.5$.

            **Verification of Means:**
            By construction, the true mean of arm $k$ is:
            $$ \mu_k = \langle x_k, \theta_* \rangle = (a \hat{x}_k)^\\top \hat{\theta} + b \cdot 1 = a (\hat{x}_k^\\top \hat{\theta}) + b = a z_k + b $$
            This guarantees that $\min_k \mu_k = 0.1$ and $\max_k \mu_k = 0.9$.
            """
            # Setup from Section 7.2.1
            self.dim = 5
            self.n_arms = 10
            
            # 1. Generate theta = [theta_hat, 1]
            theta_hat = np.random.randn(4)
            self.theta = np.concatenate([theta_hat, [1.0]])
            
            # 2. Generate x_hat for each arm
            x_hat = np.random.randn(self.n_arms, 4)
            
            # 3. Calculate dot products to determine scaling
            dots = x_hat @ theta_hat
            max_dot = np.max(dots)
            min_dot = np.min(dots)
            
            # 4. Solves a, b for scaling mean to [0.1, 0.9]
            if np.isclose(max_dot, min_dot):
                 a = 0.0
                 b = 0.5
            else:
                 a = 0.8 / (max_dot - min_dot)
                 b = 0.9 - a * max_dot
                 
            # 5. Construct arms c_x = [a * x_hat, b]
            # Shape (10, 5)
            # Last column is just b
            self.arms = np.hstack([x_hat * a, np.full((self.n_arms, 1), b)])
            
        self.true_means = self.arms @ self.theta
        self.best_arm_idx = np.argmax(self.true_means)
        self.max_reward = self.true_means[self.best_arm_idx]

    def get_reward(self, arm_idx):
        return self.true_means[arm_idx] + np.random.normal(0, self.noise_std)

    def get_regret(self, arm_idx):
        return self.max_reward - self.true_means[arm_idx]

# ==========================================
# 2. Algorithms
# ==========================================

def run_linucb(env, horizon, alpha=1.0):
    d = env.dim
    # Initialize sufficient stats
    A = np.eye(d)
    b = np.zeros(d)
    regrets = []
    cum_regret = 0
    
    for t in range(horizon):
        # 1. Estimate Theta
        A_inv = np.linalg.inv(A)
        theta_hat = A_inv @ b
        alpha = np.sqrt(d * np.log((t + 1) * (d + 1) / 0.1))  # Confidence width
        # 2. UCB Calculation
        # fast vectorized calculation: x^T A^-1 x
        # (n_arms, d) @ (d, d) @ (d, n_arms) -> diagonals
        variances = np.sum((env.arms @ A_inv) * env.arms, axis=1)
        stds = np.sqrt(variances)
        ucbs = (env.arms @ theta_hat) + alpha * stds
        
        # 3. Action
        action = np.argmax(ucbs)
        
        # 4. Observation
        reward = env.get_reward(action)
        r_step = env.get_regret(action)
        cum_regret += r_step
        regrets.append(cum_regret)
        
        # 5. Update
        x = env.arms[action]
        A += np.outer(x, x)
        b += reward * x
        
    return np.array(regrets)

def solve_op_cvxpy(hat_gaps, arms, block_len, beta_scale, d):
    n = len(hat_gaps)
    p = cp.Variable(n, nonneg=True)
    objective = cp.Minimize(p @ hat_gaps)
    
    S_p = cp.sum([p[i] * np.outer(arms[i], arms[i]) for i in range(n)])
    S_p_reg = S_p + 1e-4 * np.eye(d)
    
    constraints = [cp.sum(p) == 1]
    
    # Robust constraints scaling
    safe_gaps = np.maximum(hat_gaps, 1e-3) 
    
    for i in range(n):
        term = (block_len * (safe_gaps[i]**2)) / (beta_scale + 1e-6)
        rhs = term + 2.0 * d 
        constraints.append(cp.matrix_frac(arms[i], S_p_reg) <= rhs)
        
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.MOSEK, verbose=False, max_iters=1000)
    except:
        pass

    if p.value is None or np.isnan(p.value).any():
        dist = np.ones(n) / n
    else:
        dist = np.maximum(p.value, 0)
        if np.sum(dist) <= 1e-9: dist = np.ones(n)/n
        dist /= np.sum(dist)
    return dist

def solve_op_pd(hat_gaps, arms, block_len, beta_scale, d, n_iters=1000):
   # Primal-Dual implementation for:
   # min p^T gaps
   # s.t. ||a_i||_{S_p^-1}^2 <= RHS_i 
   
   n, dim = arms.shape
   
   # Calculate RHS constants
   safe_gaps = np.maximum(hat_gaps, 1e-3)
   rhs_vec = (block_len * (safe_gaps**2)) / (beta_scale + 1e-6) + 2.0 * d
   
   # Init vars
   p = np.ones(n) / n
   q = np.ones(n) / n # Dual weights (probability simplex trick)
   U = np.zeros((n, dim)) 
   A_mats = np.array([np.outer(a, a) for a in arms])
   
   alpha = 1.0 # Global constraint multiplier
   
   lr_u_base = 0.1
#    0.1
   # lr_alpha_base = 20.0

   for t in range(1, n_iters + 1):
       gamma_t = 2.0 / (t + 2.0)
       
       # 1. Update U (Inverse estimator)
       S = np.tensordot(p, A_mats, axes=([0],[0])) + 1e-6*np.eye(dim)
       # M_t = ceil(ln(t)) + 1
       M_t = int(np.log(t)) + 2 if t > 1 else 10
       
       for _ in range(M_t):
           grad_U = 2 * arms - 2 * (U @ S)
           U += lr_u_base * grad_U
           
       # 2. constraint violations
       # V_i = x_i^T S^-1 x_i - rhs_i
       # Approx by 2<U_i, x_i> - <U_i, S U_i> based on Fenchel
       term1 = np.sum(U * arms, axis=1)
       term2 = np.sum(U * (U @ S), axis=1)
       violations = (2 * term1 - term2) - rhs_vec
       # 3. Update Alpha
       # Scaled by gamma_t (Mirror Descent step size)
       # eta_alpha = 1.0 roughly, can tune
       eta_alpha = 10.0
       expected_viol = np.dot(q, violations)
       alpha = alpha * np.exp(eta_alpha * gamma_t * expected_viol)
       alpha = max(0.01, min(alpha, 1e6)) # Projection
       
       # 4. Update q (Adversary on constraints)
       best_q = np.argmax(violations)
       q *= (1 - gamma_t)
       q[best_q] += gamma_t
       
       # 5. Update p (Agent)
       # Efficient computation of interaction terms
       M = U @ arms.T
       gains = np.dot(q, M**2)
       
       grad_p = hat_gaps - alpha * gains
       
       best_p = np.argmin(grad_p)
       p *= (1 - gamma_t)
       p[best_p] += gamma_t
       
   return p
   
def run_op_bandit(env, horizon, beta_scale=1.0, solver='cvxpy'):
    """
    Anytime Instance-optimal Linear DOPA (Algorithm 1).
    Runs in doubling blocks. Optimization solves for exploration distribution p_m
    based on gaps estimated from the previous block using Ridge Regression.
    solver: 'cvxpy' or 'pd'
    """
    n, d = env.n_arms, env.dim
    
    regrets = []
    cum_regret = 0
    t = 1
    m = 0
    
    # Global history for Least Squares
    X_hist = []
    y_hist = []
    
    # Initial Estimates
    prev_R_hat = np.zeros(n)
    
    while t <= horizon:
        # 1. Define Block B_m length: 2^m
        block_len = 2**m
        
        # 2. Estimated Gaps (Delta_m)
        if m == 0:
            hat_gaps = np.zeros(n)
        else:
            best_val = np.max(prev_R_hat)
            hat_gaps = np.maximum(best_val - prev_R_hat, 0)
        
        # 3. Solve Optimization Problem OP
        if solver == 'cvxpy':
            dist = solve_op_cvxpy(hat_gaps, env.arms, block_len, beta_scale, d)
        elif solver == 'pd':
            dist = solve_op_pd(hat_gaps, env.arms, block_len, beta_scale, d, n_iters=4000)
            
        # 4. Run Block Execution
        end_t = min(t + block_len, horizon + 1)
        
        while t < end_t:
            action = np.random.choice(n, p=dist)
            reward = env.get_reward(action)
            
            cum_regret += env.get_regret(action)
            regrets.append(cum_regret)
            
            X_hist.append(env.arms[action])
            y_hist.append(reward)
            t += 1
            
        # 5. Update Estimates (Ridge Regression)
        # Using all history up to now is more stable than block-only IPS
        X_mat = np.array(X_hist)
        y_vec = np.array(y_hist)
        
        # Regularization lambda
        lam = 1.0 
        theta_hat = np.linalg.solve(X_mat.T @ X_mat + lam * np.eye(d), X_mat.T @ y_vec)
        prev_R_hat = env.arms @ theta_hat
            
        m += 1
        
    return np.array(regrets)

# ==========================================
# 5. Algorithm: DuSA (Dual Stabilized Algorithm)
# ==========================================

def solve_dusa_allocation(actions, gaps, d, best_idx):
    """
    Solves the DuSA efficiency optimization problem.
    Minimize Sum eta_k * gap_k
    s.t. ||a_star - a_k||_{V(eta)^-1}^2 <= gap_k^2 / 2  (for all k != star)
    
    This finds the minimal exploration rates eta needed to distinguish the best arm
    from all others with correct confidence.
    """
    K = len(actions)
    eta = cp.Variable(K, nonneg=True)
    
    # Objective: Minimize regret rate
    obj = cp.Minimize(eta @ gaps)
    
    # Construction of V(eta)
    V_sum = cp.sum([eta[i] * np.outer(actions[i], actions[i]) for i in range(K)])
    
    # Regularization for numerical stability (minimal)
    V_reg = V_sum + 1e-6 * np.eye(d)
    
    constraints = []
    a_star = actions[best_idx]
    
    for k in range(K):
        # We only constrain suboptimal arms that are somewhat close
        if k == best_idx or gaps[k] < 1e-5:
            continue
            
        # Direction to discriminate
        dir_vec = a_star - actions[k]
        
        # Constraint: Power of test >= 1 (Asymptotic lower bound condition)
        # ||a_star - a_k||^2_{V^-1} <= gap_k^2 / 2
        bound = (gaps[k]**2) / 2.0
        
        # We perform the check: matrix_frac(x, V) <= bound
        constraints.append(cp.matrix_frac(dir_vec, V_reg) <= bound)

    prob = cp.Problem(obj, constraints)
    
    try:
        prob.solve(solver=cp.MOSEK, verbose=False, max_iters=2000)
    except:
        pass
        
    if eta.value is None:
        return np.ones(K)
        
    return np.maximum(eta.value, 0)

def run_dusa(env, horizon):
    """
    DuSA: Dual Stabilized Algorithm for Linear Bandits.
    Faithful implementation matching DuSA.jl:
    - Re-evaluates stopping rules at every step (no latching)
    - Includes Estimation Phase for under-sampled arms
    - Solves optimal allocation for exploration
    """
    n, d = env.n_arms, env.dim
    
    # Initialization
    regrets = []
    cum_regret = 0
    
    # Track statistics
    V_t = np.zeros((d, d))
    b_t = np.zeros(d)
    N_pulls = np.zeros(n)
    
    # Force initial exploration 
    for i in range(n):
        r = env.get_reward(i)
        
        x = env.arms[i]
        V_t += np.outer(x, x)
        b_t += r * x
        
        N_pulls[i] += 1
        cum_regret += env.get_regret(i)
        regrets.append(cum_regret)
        
    # State tracking
    s_t = 0 # Counter for exploration rounds (Julia: s)
    
    def glrt_threshold(t):
        if t <= 1: return 100.0
        # Threshold similar to log(t) + ...
        # Julia uses normalized checks against ~1.0
        # Here we use the unnormalized equivalent.
        val = np.log(t) + 3 * np.log(np.log(t) + 1e-2)
        return max(val, 0) + 2.0 # Little buffer

    for t in range(n + 1, horizon + 1):
        # 1. MLE Update
        V_inv = np.linalg.inv(V_t + 1e-5 * np.eye(d))
        theta_hat = V_inv @ b_t
        
        est_means = env.arms @ theta_hat
        best_idx = np.argmax(est_means)
        a_star = env.arms[best_idx]
        
        # Estimated gaps
        est_gaps = np.maximum(est_means[best_idx] - est_means, 0)
        
        # 2. GLRT Stopping Test
        # Z = min_{k != *} (Delta_k^2) / (2 * ||a_* - a_k||_{V_t^-1}^2)
        min_Z = float('inf')
        
        for k in range(n):
            if k == best_idx: continue
            
            diff = a_star - env.arms[k]
            norm_sq = diff @ V_inv @ diff
            
            if norm_sq < 1e-9:
                z_k = float('inf')
            else:
                z_k = (est_gaps[k]**2) / (2 * norm_sq)
                
            if z_k < min_Z:
                min_Z = z_k
        
        # 3. Decision Logic (DuSA.jl style)
        beta_t = glrt_threshold(t)
        
        if min_Z >= beta_t:
            # --- Exploitation Phase ---
            # We have sufficient confidence
            action = best_idx
        else:
            # --- Exploration/Estimation Phase ---
            s_t += 1
            
            # A) Estimation Check (Avoid under-sampling)
            # Julia: minimum(N_t) <= 0.01 * s_t / log(s_t + 1)
            est_thresh = 0.01 * s_t / (np.log(s_t + 1) + 1e-6)
            
            if np.min(N_pulls) <= est_thresh:
                action = np.argmin(N_pulls)
            else:
                # B) Allocation Phase
                opt_eta = solve_dusa_allocation(env.arms, est_gaps, d, best_idx)
                
                # C-Tracking
                # Target counts matches the GLRT requirement: N ~ eta * log(t)
                targets = opt_eta * beta_t
                
                diff = targets - N_pulls
                # We pull the arm that is furthest behind its target
                action = np.argmax(diff)
                
                # If all constraints satisfied, default to best_idx
                if diff[action] <= 0:
                    action = best_idx
                
        # 4. Observe
        reward = env.get_reward(action)
        cum_regret += env.get_regret(action)
        regrets.append(cum_regret)
        
        N_pulls[action] += 1
        x_act = env.arms[action]
        V_t += np.outer(x_act, x_act)
        b_t += reward * x_act
        
    return np.array(regrets)

# ==========================================
# 3. Simulation Runner
# ==========================================

def run_experiment(n_trials=5, horizon=200, mode='paper'):
    print(f"Starting Experiment: {n_trials} trials, T={horizon}, Mode={mode}")
    
    results = {
        # 'OP (CVXPY)': [],
        'LDOPA': [],
        'LinUCB': [],
        # 'FTRL-Shannon': [],
        'DuSA': []
    }
    
    execution_times = {
        'LDOPA': 0.0,
        'DuSA': 0.0
    }
    
    for i in range(n_trials):
        print(f"  Trial {i+1}/{n_trials}...")
        # Use the selected instance (generates new random params every time for 'paper' mode)
        env = LinearBanditEnv(mode=mode, noise_std=0.5)
        
        # Run LinUCB
        # alpha=0.5 often tuned better for small scale problems
        r_ucb = run_linucb(env, horizon, alpha=0.5) 
        results['LinUCB'].append(r_ucb)
        
        # Run FTRL
        # r_ftrl = run_ftrl_shannon(env, horizon, lr=5.0)
        # results['FTRL-Shannon'].append(r_ftrl)
        
        # Run OP (CVXPY)
        # r_op_cvx = run_op_bandit(env, horizon, beta_scale=1.0, solver='cvxpy')
        # results['OP (CVXPY)'].append(r_op_cvx)

        # Run OP (PD)
        tic = time.time()
        r_op_pd = run_op_bandit(env, horizon, beta_scale=1.0, solver='pd')
        toc = time.time()
        results['LDOPA'].append(r_op_pd)
        execution_times['LDOPA'] += (toc - tic)
        print(f"    LDOPA Time: {toc - tic:.4f}s")
        print(f"    LDOPA Regret: {r_op_pd[-1]:.2f}"  )
        # Run DuSA
        tic = time.time()
        r_dusa = run_dusa(env, horizon)
        toc = time.time()
        results['DuSA'].append(r_dusa)
        execution_times['DuSA'] += (toc - tic)
        print(f"    DuSA Time: {toc - tic:.4f}s")
        print(f"    DuSA Regret: {r_dusa[-1]:.2f}"  )
        
    print("\nTotal Execution Times (over all trials):")
    print(f"  LDOPA: {execution_times['LDOPA']:.4f}s")
    print(f"  DuSA:    {execution_times['DuSA']:.4f}s")
    print(f"  Ratio (LDOPA/DuSA): {execution_times['LDOPA']/execution_times['DuSA']:.2f}x\n")
    return results

# ==========================================
# 4. Plotting
# ==========================================

def plot_results(results, horizon):
    plt.figure(figsize=(8,6), dpi=120)
    
    colors = {
        'LinUCB': 'green',
        # 'FTRL-Shannon': 'purple',
        # 'OP (CVXPY)': 'cyan',
        'LDOPA': 'blue',
        'DuSA': 'red'
    }
    styles = {
        'LinUCB': '--',
        # 'FTRL-Shannon': '-.',
        # 'OP (CVXPY)': '-',
        'LDOPA': '-',
        'DuSA': ':'
    }
    
    x = np.arange(horizon)
    
    for name, data in results.items():
        if len(data) == 0: continue
        data = np.array(data)
        # Pad with 0 for the warm-start steps if lengths differ
        # (Our code handles lengths consistently, but just in case)
        if data.shape[1] != horizon:
            print(f"Warning: {name} data length mismatch")
            continue
            
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        plt.plot(x, mean, label=name, color=colors[name], linestyle=styles[name], linewidth=2)
        plt.fill_between(x, mean - 0.5*std, mean + 0.5*std, color=colors[name], alpha=0.2)
        
    plt.title("Stochastic Linear Bandit Comparison", fontsize=14)
    plt.xlabel("Round t", fontsize=14)
    plt.ylabel("Cumulative Regret", fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"lin_bandit_comparison_ldopa_opt.png")
    plt.show()

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # Parameters tailored for comparison
    T = 10000
    trials = 100 # As per Section 7.2.1
    
    data = run_experiment(n_trials=trials, horizon=T, mode='paper')
    plot_results(data, T)



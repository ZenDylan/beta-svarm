"""
β-SVARM: Beta-weighted Semivalue Data Valuation
via Stratified Non-Marginal Decomposition.

Core idea:
  1. SVARM decomposition: φᵢ = φᵢ⁺ − φᵢ⁻ (no marginal contributions needed)
  2. Each coalition evaluation updates ALL n players simultaneously
  3. Stratum estimates are reweighted with Beta(α,β) for semivalue computation
  4. Adaptive Neyman allocation steers budget to high-weight, high-variance strata
"""

import math
import numpy as np
from scipy.special import beta as beta_func, comb
from sklearn.linear_model import LogisticRegression
import time


def _generate_paper_distribution(n):
    """
    Probability distribution over coalition sizes (0..n) according to the
    paper's theoretical optimal allocation for stratified sampling.

    For even n:
      P(s) ∝ 1/s for s=2..n/2-1 and s=n/2+1..n-2 (symmetric pairs)
      P(n/2) ∝ 1/(n log n)
    For odd n:
      P(s) ∝ 1/s for s=2..(n-1)/2 and s=(n+1)/2..n-2 (symmetric pairs)
    """
    dist = [0.0 for _ in range(n + 1)]

    if n % 2 == 0:
        nlogn = n * math.log(n)
        H = sum(1.0 / s for s in range(1, n // 2))
        frac = (nlogn - 1) / (2 * nlogn * (H - 1))
        for s in range(2, n // 2):
            dist[s] = frac / s
            dist[n - s] = frac / s
        dist[n // 2] = 1.0 / nlogn
    else:
        H = sum(1.0 / s for s in range(1, (n - 1) // 2 + 1))
        frac = 1.0 / (2 * (H - 1))
        for s in range(2, (n - 1) // 2 + 1):
            dist[s] = frac / s
            dist[n - s] = frac / s

    return dist


class BetaSVARM:
    """
    β-SVARM data valuator.

    Parameters
    ----------
    alpha : float, default=16
        Alpha parameter of Beta(α,β) weighting. α=1, β=1 recovers Shapley.
    beta_param : float, default=1
        Beta parameter of Beta(α,β) weighting.
    adaptive : bool, default=True
        Whether to use Neyman allocation across strata.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(self, alpha=16, beta_param=1, adaptive=True, random_state=42):
        self.alpha = alpha
        self.beta_param = beta_param
        self.adaptive = adaptive
        self.rng = np.random.RandomState(random_state)

    def _evaluate_utility(self, indices, X_train, y_train, X_val, y_val, model_class):
        """Train model on subset and return validation accuracy."""
        if len(indices) == 0:
            if len(y_val) == 0:
                return 0.5
            most_common = np.bincount(y_val.astype(int)).argmax()
            return np.mean(y_val == most_common)
        X_sub, y_sub = X_train[indices], y_train[indices]
        if len(np.unique(y_sub)) < 2:
            return np.mean(y_val == y_sub[0])
        try:
            model = model_class(max_iter=500, random_state=42)
            model.fit(X_sub, y_sub)
            return model.score(X_val, y_val)
        except Exception:
            return 0.5

    def compute_beta_weights(self, n, alpha=None, beta_param=None):
        """
        Compute normalized Beta(α,β) semivalue weights for coalition sizes k=0..n-1.

        Weight formula: w(k) = B(k+β, n-1-k+α) / B(α, β)
        Then normalized so Σ_k C(n-1,k) * w(k) = 1.
        """
        if alpha is None:
            alpha = self.alpha
        if beta_param is None:
            beta_param = self.beta_param

        weights = np.zeros(n)
        for k in range(n):
            a = k + beta_param
            b = n - 1 - k + alpha
            if a > 0 and b > 0:
                weights[k] = beta_func(a, b) / beta_func(alpha, beta_param)
        # Normalize
        total = sum(comb(n - 1, k, exact=True) * weights[k] for k in range(n))
        if total > 0:
            weights /= total
        return weights

    def fit(self, X_train, y_train, X_val, y_val,
            model_class=LogisticRegression, budget=5000):
        """
        Run β-SVARM to compute data values.

        Parameters
        ----------
        X_train, y_train : training data
        X_val, y_val : validation data
        model_class : sklearn classifier class
        budget : int, total number of utility evaluations allowed

        Returns
        -------
        values : np.array of shape (n,), the data value of each training point
        meta : dict with stratum-level estimates (for multi-semivalue reweighting)
        """
        n = len(X_train)

        # Storage: phi_plus[i, k] = running mean of v(A) where i∈A and |A\{i}|=k
        #          phi_minus[i, k] = running mean of v(A) where i∉A and |A|=k
        phi_plus = np.zeros((n, n))
        phi_minus = np.zeros((n, n))
        count_plus = np.zeros((n, n), dtype=int)
        count_minus = np.zeros((n, n), dtype=int)

        # Online variance tracking per stratum (Welford's algorithm)
        stratum_mean = np.zeros(n)
        stratum_M2 = np.zeros(n)
        stratum_count = np.zeros(n, dtype=int)
        eval_count = 0

        # ---- Phase 1: Exact boundary strata ----
        # v(∅)
        v_empty = self._evaluate_utility(
            np.array([], dtype=int), X_train, y_train, X_val, y_val, model_class)
        eval_count += 1
        for i in range(n):
            phi_minus[i, 0] = v_empty
            count_minus[i, 0] = 1

        # Singletons: v({i}) for each i
        for i in range(n):
            if eval_count >= budget:
                break
            v_i = self._evaluate_utility(
                np.array([i]), X_train, y_train, X_val, y_val, model_class)
            eval_count += 1
            # i ∈ A={i}, |A|=1, so stratum k=0 for phi_plus
            phi_plus[i, 0] = v_i
            count_plus[i, 0] = 1
            # For all j ≠ i: A={i} has size 1, j ∉ A, so stratum k=1 for phi_minus
            for j in range(n):
                if j != i:
                    old_c = count_minus[j, 1]
                    phi_minus[j, 1] = (phi_minus[j, 1] * old_c + v_i) / (old_c + 1)
                    count_minus[j, 1] += 1

        # v(N) — grand coalition
        all_idx = np.arange(n)
        v_full = self._evaluate_utility(
            all_idx, X_train, y_train, X_val, y_val, model_class)
        eval_count += 1
        for i in range(n):
            # i ∈ N, |N\{i}| = n-1, so stratum k=n-1 for phi_plus
            phi_plus[i, n - 1] = v_full
            count_plus[i, n - 1] = 1

        # Leave-one-out: v(N\{i}) for each i
        for i in range(n):
            if eval_count >= budget:
                break
            others = np.array([j for j in range(n) if j != i])
            v_loo = self._evaluate_utility(
                others, X_train, y_train, X_val, y_val, model_class)
            eval_count += 1
            # i ∉ A=N\{i}, |A|=n-1, so stratum k=n-1 for phi_minus
            phi_minus[i, n - 1] = v_loo
            count_minus[i, n - 1] = 1
            # For all j ≠ i: j ∈ A=N\{i}, |A\{j}|=n-2, so stratum k=n-2 for phi_plus
            for j in range(n):
                if j != i:
                    old_c = count_plus[j, n - 2]
                    phi_plus[j, n - 2] = (phi_plus[j, n - 2] * old_c + v_loo) / (old_c + 1)
                    count_plus[j, n - 2] += 1

        # ---- Phase 2: Warm-up (one sample per stratum s=2..n-2) ----
        for s in range(2, n - 1):
            if eval_count >= budget:
                break
            A = self.rng.choice(n, size=s, replace=False)
            val = self._evaluate_utility(
                A, X_train, y_train, X_val, y_val, model_class)
            eval_count += 1
            self._swarm_update(A, val, n, phi_plus, phi_minus,
                               count_plus, count_minus)
            self._update_variance(s, val, stratum_mean, stratum_M2, stratum_count)

        # ---- Phase 3: Main sampling loop ----
        beta_w = self.compute_beta_weights(n)

        while eval_count < budget:
            s = self._sample_stratum(n, beta_w, stratum_M2, stratum_count)
            A = self.rng.choice(n, size=s, replace=False)
            val = self._evaluate_utility(
                A, X_train, y_train, X_val, y_val, model_class)
            eval_count += 1
            self._swarm_update(A, val, n, phi_plus, phi_minus,
                               count_plus, count_minus)
            if 2 <= s <= n - 2:
                self._update_variance(s, val, stratum_mean, stratum_M2, stratum_count)

        # ---- Phase 4: Aggregate with Beta weights ----
        values = self._aggregate(n, phi_plus, phi_minus, count_plus, count_minus, beta_w)

        meta = {
            'phi_plus': phi_plus.copy(),
            'phi_minus': phi_minus.copy(),
            'count_plus': count_plus.copy(),
            'count_minus': count_minus.copy(),
            'eval_count': eval_count,
        }
        return values, meta

    def _sample_stratum(self, n, beta_w, stratum_M2, stratum_count):
        """Sample coalition size using adaptive Neyman or uniform distribution.

        Only samples from s=2..n-2 because:
          - s=0 (empty set) and s=n (grand coalition) are singletons
          - s=1 (singletons) and s=n-1 (leave-one-outs) are computed
            exactly in Phase 1
        """
        lo, hi = 2, n - 2  # valid range for random sampling
        size = hi - lo + 1
        if hi < lo:
            # n <= 4: no interior strata, fall back to s=1..n-1
            return self.rng.randint(1, n)
        if self.adaptive and np.sum(stratum_count[lo:hi + 1]) > 0:
            probs = np.zeros(size)
            for idx, s in enumerate(range(lo, hi + 1)):
                # β weight for stratum k=s-1 (phi_plus) and k=s (phi_minus)
                # Use the average of the two relevant weights
                w = (beta_w[s - 1] + beta_w[s]) / 2.0 if s < n else beta_w[s - 1]
                # Estimated std dev
                if stratum_count[s] > 1:
                    var_s = stratum_M2[s] / (stratum_count[s] - 1)
                else:
                    var_s = 1.0
                probs[idx] = w * np.sqrt(max(var_s, 1e-12))
            p_sum = np.sum(probs)
            if p_sum > 0:
                probs /= p_sum
                return lo + self.rng.choice(size, p=probs)
        # Fallback: paper distribution over interior strata s=2..n-2
        full_dist = _generate_paper_distribution(n)
        interior_probs = np.array([full_dist[s] for s in range(lo, hi + 1)])
        p_sum = np.sum(interior_probs)
        if p_sum > 0:
            interior_probs /= p_sum
            return lo + self.rng.choice(size, p=interior_probs)
        return self.rng.randint(lo, hi + 1)

    def _swarm_update(self, A, val, n, phi_plus, phi_minus,
                      count_plus, count_minus):
        """
        SVARM core: one coalition evaluation updates ALL n players.

        If coalition A has size s:
          - For i ∈ A:  update phi_plus[i, s-1]  (stratum k = s-1)
          - For i ∉ A:  update phi_minus[i, s]   (stratum k = s)
        """
        s = len(A)
        in_A = np.zeros(n, dtype=bool)
        in_A[A] = True

        for i in range(n):
            if in_A[i]:
                k = s - 1
                if 0 <= k < n:
                    c = count_plus[i, k]
                    phi_plus[i, k] = (phi_plus[i, k] * c + val) / (c + 1)
                    count_plus[i, k] += 1
            else:
                k = s
                if 0 <= k < n:
                    c = count_minus[i, k]
                    phi_minus[i, k] = (phi_minus[i, k] * c + val) / (c + 1)
                    count_minus[i, k] += 1

    def _update_variance(self, s, val, mean, M2, count):
        """Welford's online variance for stratum s."""
        count[s] += 1
        delta = val - mean[s]
        mean[s] += delta / count[s]
        delta2 = val - mean[s]
        M2[s] += delta * delta2

    def _aggregate(self, n, phi_plus, phi_minus, count_plus, count_minus, weights):
        """Aggregate stratum estimates with given semivalue weights.

        weights[k] is the per-coalition weight (normalized so that
        Σ_k C(n-1,k)*w(k) = 1).  The per-stratum contribution must
        therefore be multiplied by C(n-1,k).
        """
        values = np.zeros(n)
        for i in range(n):
            for k in range(n):
                mu_p = phi_plus[i, k] if count_plus[i, k] > 0 else 0.0
                mu_m = phi_minus[i, k] if count_minus[i, k] > 0 else 0.0
                values[i] += comb(n - 1, k, exact=True) * weights[k] * (mu_p - mu_m)
        return values

    def reweight(self, meta, n, alpha_new, beta_new):
        """
        Re-aggregate existing stratum estimates with NEW semivalue weights.

        This is the unique feature of β-SVARM: one run, many semivalues.
        The stratum estimates {phi_plus, phi_minus} are semivalue-agnostic;
        only the final aggregation weights depend on (α, β).
        """
        new_weights = self.compute_beta_weights(n, alpha_new, beta_new)
        return self._aggregate(
            n, meta['phi_plus'], meta['phi_minus'],
            meta['count_plus'], meta['count_minus'], new_weights)

    def banzhaf_reweight(self, meta, n):
        """
        Re-aggregate with uniform Banzhaf weights.

        Banzhaf: every coalition size gets equal weight,
        so we just average all phi_plus and phi_minus across strata.
        """
        values = np.zeros(n)
        for i in range(n):
            total_p, cnt_p = 0.0, 0
            total_m, cnt_m = 0.0, 0
            for k in range(n):
                if meta['count_plus'][i, k] > 0:
                    total_p += meta['phi_plus'][i, k] * meta['count_plus'][i, k]
                    cnt_p += meta['count_plus'][i, k]
                if meta['count_minus'][i, k] > 0:
                    total_m += meta['phi_minus'][i, k] * meta['count_minus'][i, k]
                    cnt_m += meta['count_minus'][i, k]
            values[i] = (total_p / max(cnt_p, 1)) - (total_m / max(cnt_m, 1))
        return values

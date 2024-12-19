import numpy as np
import math

class FISTA:
    def __init__(self, subalgorithm, L0=1.1, eta=1.1, epsilon=1e-5, max_iter=100):
        self.subalgorithm = subalgorithm
        self.L0 = L0
        self.eta = eta
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.MC_history = []

    def optimize(self, p0):
        p_bar = p0
        t_k = 1.0
        L_k = self.L0
        p_k_prev = p0

        for k in range(self.max_iter):
            # Compute MC and gradient at p_bar
            MC_pbar, grad_pbar = self.subalgorithm.run(p_bar)
            grad_norm = np.linalg.norm(grad_pbar)
            grad_norm_sq = grad_norm*grad_norm

            # Line search
            while True:
                # Candidate step
                p_new = p_bar + (grad_pbar / L_k)
                MC_f, _ = self.subalgorithm.run(p_new)

                # Quadratic upper bound check
                F = MC_f
                Q = MC_pbar + grad_norm_sq/(2*L_k)
                if F >= Q:
                    break
                L_k *= self.eta

            self.MC_history.append(MC_pbar)

            # Update p_k
            p_k = p_bar + (grad_pbar / L_k)

            # Convergence check
            if grad_norm < self.epsilon:
                print(f"Converged in {k+1} iterations.")
                return p_k

            # Restarting condition
            if np.sum(grad_pbar * (p_k - p_k_prev)) < 0.0:
                t_k = 1.0

            # FISTA update for t_k
            t_k_next = (1.0 + math.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0

            # Update p_bar
            p_bar = p_k + ((t_k - 1.0)/t_k_next)*(p_k - p_k_prev)
            t_k = t_k_next
            p_k_prev = p_k

        print("Reached maximum iterations without convergence.")
        return p_k

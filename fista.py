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
        p_bar = p0.copy()
        t_k = 1
        L_k = self.L0
        p_k_prev = p0.copy()

        for k in range(self.max_iter):
            # Compute MC and gradient at p_bar
            MC_pbar, grad_pbar = self.subalgorithm.run(p_bar)
            grad_norm = np.linalg.norm(grad_pbar)
            
            while True:
                p_new = p_bar + (1.0/L_k)*grad_pbar
                MC_f, _ = self.subalgorithm.run(p_new)
                F = MC_f
                Q = MC_pbar + (grad_norm*grad_norm)/(2*L_k)
                if F >= Q:
                    break
                else:
                    L_k *= self.eta
            self.MC_history.append(MC_pbar)
            p_k = p_bar + (1/L_k)*grad_pbar

            # Check for convergence
            if grad_norm < self.epsilon:
                print(f"Converged in {k+1} iterations.")
                break

            # Use element-wise multiplication and sum for multi-dimensional arrays
            if (grad_pbar * (p_k - p_k_prev)).sum() < 0:
                t_k = 1

            t_k_next = (1 + math.sqrt(1 + 4 * t_k * t_k)) / 2
            p_bar = p_k + ((t_k - 1)/t_k_next)*(p_k - p_k_prev)
            t_k = t_k_next
            p_k_prev = p_k.copy()

            if k == self.max_iter - 1:
                print("Reached maximum iterations without convergence.")

        return p_k

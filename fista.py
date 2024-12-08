import numpy as np
import math
from collections import defaultdict
import heapq

# fista.py

class FISTA:
    def __init__(self, subalgorithm, L0=1.0, eta=1.5, epsilon=1e-5, max_iter=100):
        self.subalgorithm = subalgorithm
        self.L0 = L0
        self.eta = eta
        self.epsilon = epsilon
        self.max_iter = max_iter

    def optimize(self, v0):
        v_k = v0.copy()
        y_k = v0.copy()
        t_k = 1
        L_k = self.L0

        for k in range(self.max_iter):
            # Compute MC and gradient at y_k
            MC_yk, grad_yk = self.subalgorithm.run(y_k)
            print("MC",MC_yk)
            print("grad_MC", grad_yk)

            # Line search to find appropriate step size
            while True:
                # Gradient step (ensure operations are NumPy compatible)
                v_new = y_k - (1 / L_k) * grad_yk
                MC_vk, _ = self.subalgorithm.run(v_new)

                # Quadratic approximation
                Q = (MC_yk +
                    np.dot(grad_yk, v_new - y_k) +
                    (L_k / 2) * np.sum((v_new - y_k) ** 2))
                if MC_vk <= Q:
                    break
                else:
                    L_k *= self.eta

            t_new = (1 + np.sqrt(1 + 4 * t_k ** 2)) / 2
            y_k = v_new + ((t_k - 1) / t_new) * (v_new - v_k)
            v_k = v_new.copy()
            t_k = t_new

            # Check for convergence
            grad_norm = np.linalg.norm(grad_yk)
            if grad_norm < self.epsilon:
                print(f"Converged in {k+1} iterations.")
                break

            if k == self.max_iter - 1:
                print("Reached maximum iterations without convergence.")

        return v_k


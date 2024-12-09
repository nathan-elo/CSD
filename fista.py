import numpy as np
import math

class FISTA:
    def __init__(self, subalgorithm, L0=1.1, eta=1.1, epsilon=1e-5, max_iter=100):
        """
        Initialize the FISTA optimizer.
        
        Parameters:
        - subalgorithm: An object with a `run` method that computes MC and grad_MC.
        - L0: Initial step size (L0 > 0).
        - eta: Scaling factor for L_k (η > 1).
        - epsilon: Convergence tolerance for the gradient norm.
        - max_iter: Maximum number of iterations.
        """
        self.subalgorithm = subalgorithm
        self.L0 = L0
        self.eta = eta
        self.epsilon = epsilon
        self.max_iter = max_iter

    def optimize(self, p0):
        """
        Perform the optimization using the FISTA algorithm.
        
        Parameters:
        - p0: Initial guess for the parameter vector.

        Returns:
        - p_k: Optimized parameter vector.
        """
        # Initialize variables
        p_bar = p0.copy()
        t_k = 1
        L_k = self.L0
        p_k_prev = p0.copy()
        eta=self.eta

        for k in range(self.max_iter):
            # Perform line search
            i=0
            while True:
                
                # Compute MC and gradient at p_bar
                MC_pbar, grad_pbar = self.subalgorithm.run(p_bar)
                
                step_size = 1 / (np.power(eta, i) * L_k)
                p_new = p_bar + step_size * grad_pbar
                grad_norm = np.linalg.norm(grad_pbar)
                # Compute MC and gradient at p_new
                MC_f, _ = self.subalgorithm.run(p_new)
                
                # Compute F and Q
                F = MC_f
                Q = MC_pbar + (1 / (2* np.power(eta,i)* L_k)) * grad_norm*grad_norm
                i +=1
                if F >= Q:
                    break  # Exit the inner loop
                else:
                    L_k *= self.eta  # Increase L_k
            print("MC",MC_pbar)
            print("grad",grad_pbar)
            print("L",L_k)
            # Update p_k
            p_k = p_bar + (1 / L_k) * grad_pbar
           
            # Check for convergence
            if grad_norm < self.epsilon:
                print(f"Converged in {k+1} iterations.")
                return p_k
            
            # Adjust t_k if necessary
            if np.dot(grad_pbar, p_k - p_k_prev) < 0:
                t_k = 1
            
            # Update t_k and p_bar
            t_k_next = (1 + np.sqrt(1 + 4 * t_k ** 2)) / 2
            p_bar = p_k + ((t_k - 1) / t_k_next) * (p_k - p_k_prev)
            t_k = t_k_next
            p_k_prev = p_k.copy()  # Save p_k for the next iteration
            

            if k == self.max_iter - 1:
                print("Reached maximum iterations without convergence.")

        return p_k



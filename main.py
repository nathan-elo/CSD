import numpy as np
from fista import FISTA
from MCA_func import calculate_MC_and_grad_MC

def main():

    seed = 1  # Choose any integer
    rng = np.random.default_rng(seed)
    # Define network
    o_dict = {
        "1": {"2": 1, "3": 1, "6": 10, "7": 10},
        "8": {"2": 1, "3": 1, "6": 10, "7": 10}
    }

    r_dict = {
        "2": {"4": 1},
        "3": {"5": 2}
    }

    s_dict = {
        "4": {"2": 2, "3": 1, "6": 10, "7": 5},
        "5": {"2": 1, "3": 1, "6": 8, "7": 5}
    }

    d_dict = {
        "6": {},
        "7": {}
    }

    # Parameters
    n_size = 5  # number of deliveries to be made
    rs_size = 2  # number of delivery spots
    K = 2
    theta_p = 1  # negative sign included
    theta_c = 1

    z_rs_bar = np.zeros(rs_size)
    z_rs_bar[0] = 2
    z_rs_bar[1] = 3  # Total deliveries to match number of deliveries to be made

    y_od_bar = 3  # Drivers per OD pair

    c_rs = np.zeros((rs_size, K))
    for r_s in range(rs_size):
        c_rs[r_s, 1] = 5
        c_rs[r_s, 0] = c_rs[r_s, 1] + 2

    # Define subalgorithm wrapper
    class Subalgorithm:
        def __init__(self, o_dict, r_dict, s_dict, d_dict, n_size, rs_size, K, theta_p, theta_c, z_rs_bar, y_od_bar, c_rs):
            self.o_dict = o_dict
            self.r_dict = r_dict
            self.s_dict = s_dict
            self.d_dict = d_dict
            self.n_size = n_size
            self.rs_size = rs_size
            self.K = K
            self.theta_p = theta_p
            self.theta_c = theta_c
            self.z_rs_bar = z_rs_bar
            self.y_od_bar = y_od_bar
            self.c_rs = c_rs

        def run(self, v):
            return calculate_MC_and_grad_MC(
                self.o_dict, self.r_dict, self.s_dict, self.d_dict,
                self.n_size, self.rs_size, self.K, self.theta_p, self.theta_c,
                self.z_rs_bar, self.y_od_bar, self.c_rs
            )

    subalgorithm = Subalgorithm(o_dict, r_dict, s_dict, d_dict, n_size, rs_size, K, theta_p, theta_c, z_rs_bar, y_od_bar, c_rs)

    # Initial guess for v
    v0 = np.zeros(rs_size)

    # FISTA optimization
    fista = FISTA(subalgorithm, L0=1.0, eta=1.5, epsilon=1e-5, max_iter=100)
    optimal_v = fista.optimize(v0)

    print("Optimal v:", optimal_v)

if __name__ == "__main__":
    main()
import numpy as np
from fista import FISTA
from fista_copy import FISTA_neg
from fista2 import optimization_algorithm
from MCA_func import calculate_MC_and_grad_MC

def main():

    seed = 42  # Choose any integer
    # Define network
    '''
    o_dict = {
        "1": {"2": 1, "3": 1,"7" : 2, "6": 10}
    }

    # Connected to a single S node
    r_dict = {
        "2": {"4": 1},
        "3": {"5": 2},
        "7": {"8" : 2}
    }

    # Connected to every R node and D nodes
    s_dict = {
        "4": {"2": 2, "3": 1,"7":1, "6": 10},
        "5": {"2": 1, "3": 1,"7":1, "6": 8},
        "8": {"2": 1, "3": 1,"7":1, "6": 8}
    }

    d_dict = {
        "6": {},
    } 
    '''
    o_dict = {
        "1": {"2": 1, "3": 1, "6": 10, "7": 10},
        "8": {"2": 1, "3": 1, "6": 10, "7": 10}
    }

    r_dict = {
        "2": {"4": 1},
        "3": {"5": 1}
    }

    s_dict = {
        "4": {"2": 1, "3": 1, "6": 1, "7": 1},
        "5": {"2": 1, "3": 1, "6": 1, "7": 1}
    }

    d_dict = {
        "6": {},
        "7": {}
    }
    ''' 
    o_dict = {
    "1": {"2": 1, "3": 1, "6": 10}
    }
    # Connected to a single S node
    r_dict = {
        "2": {"4": 1},
        "3": {"5": 2}
    }

    # Connected to every R node and D nodes
    s_dict = {
        "4": {"2": 2, "3": 1, "6": 10},
        "5": {"2": 1, "3": 1, "6": 8}
    }

    d_dict = {
        "6": {},
    }
    '''
    # Parameters
    n_size = 3  # number of deliveries to be made
    rs_size = 2  # number of delivery spots
    K = 2
    theta_p = 1  
    theta_c = 1

    z_rs_bar = np.zeros(rs_size)
    z_rs_bar[0] = 1
    z_rs_bar[1] = 1  # Total deliveries to match number of deliveries to be made
    
    y_od_bar = 1  # Drivers per OD pair
    

    c_rs = np.zeros((rs_size, K))
    for r_s in range(rs_size):
        c_rs[r_s, 1] = 5
        c_rs[r_s, 0] = c_rs[r_s, 1] + 2

    # Define subalgorithm wrapper
    class Subalgorithm:
        def __init__(self, o_dict, r_dict, s_dict, d_dict, n_size, rs_size, K, theta_p, theta_c, z_rs_bar, y_od_bar, c_rs, seed):
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
            self.seed = seed

        def run(self, v):
            # Pass v as an argument!
            return calculate_MC_and_grad_MC(
                self.o_dict, self.r_dict, self.s_dict, self.d_dict,
                self.n_size, self.rs_size, self.K, self.theta_p, self.theta_c,
                self.z_rs_bar, self.y_od_bar, self.c_rs, self.seed, v
            )


    subalgorithm = Subalgorithm(o_dict, r_dict, s_dict, d_dict, n_size, rs_size, K, theta_p, theta_c, z_rs_bar, y_od_bar, c_rs,seed)

    # Initial guess for v
    v0 = np.zeros(rs_size)
    v0 +=1
    #no matter the inital value, it converges to the same solution, this is nice !
    # but the solution is negative.

    # FISTA optimization
    fista = FISTA(subalgorithm, L0=1.0, eta=1.1, epsilon=1e-7, max_iter=1000)
    optimal_v = fista.optimize(v0)

    # FISTA optimization with minimization
    #fista = FISTA_neg(subalgorithm, L0=1.0, eta=1.1, epsilon=1e-5, max_iter=10)
    #optimal_v = fista.optimize_neg(v0) #causes overflow....

    print("Optimal v:", optimal_v)

if __name__ == "__main__":
    main()

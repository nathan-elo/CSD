import numpy as np
from fista import FISTA
from MCA_modified import compute_MC_and_grad_MC
from MCA_modified import compute_MC_and_grad_MC_return_intermediate
from MCA_modified import compute_y_distrib
from MCA_modified import compute_z_rs_1
from MCA_modified import compute_z_rs_0
import matplotlib.pyplot as plt
import time

def main():
    seed = 42
    # Each r connected to each s node
    o_dict = {
        "1": {"2": 2, "3": 2,"9" : 1, "6": 5,"7" : 5},
        "8": {"2": 1, "3": 1,"9" : 1, "6": 5,"7" : 5}
    }
    r_dict = {
        "2": {"4": 1, "5":1,"10":1},
        "3": {"4": 1, "5":1,"10":1},
        "9": {"4": 1, "5":1,"10":1}
    }
    s_dict = {
        "4": {"2": 1, "3": 3,"9" : 1, "6": 6,"7" : 6},
        "5": {"2": 2, "3": 1,"9" : 1, "6": 6,"7" : 6},
        "10": {"2": 2, "3": 1,"9" : 1, "6": 6,"7" : 6}
    }
    d_dict = {
        "6": {},
        "7" : {}
    }

    n_size = 9  # number of deliveries to be made
    r_size=3
    s_size=3
    o_size=2
    d_size=2
    K = 2
    theta_p = 1
    theta_c = 1

    
    z_rs_bar = np.zeros((r_size,s_size))
    z_rs_bar +=1
    y_od_bar = np.zeros((o_size,d_size))
    y_od_bar +=2

    # Set c_rs and v 
    c_rs = np.zeros((r_size, s_size, K))
    for rr in range(r_size):
        for ss in range(s_size):
            c_rs[rr, ss, 1] = 5
            c_rs[rr, ss, 0] = c_rs[rr, ss, 1] + 1 # cost difference

    class Subalgorithm:
        def __init__(self, o_dict, r_dict, s_dict, d_dict, n_size, K, theta_p, theta_c, z_rs_bar, y_od_bar, c_rs, seed):
            self.o_dict = o_dict
            self.r_dict = r_dict
            self.s_dict = s_dict
            self.d_dict = d_dict
            self.n_size = n_size
            self.K = K
            self.theta_p = theta_p
            self.theta_c = theta_c
            self.z_rs_bar = z_rs_bar
            self.y_od_bar = y_od_bar
            self.c_rs = c_rs
            self.seed = seed

        def run(self, v):
            return compute_MC_and_grad_MC(self.o_dict, self.r_dict, self.s_dict, self.d_dict,
                                          self.n_size, self.K, self.theta_p, self.theta_c,
                                          self.z_rs_bar, self.y_od_bar, self.c_rs, self.seed, v)
        
        def run_fr_sol(self, v):
            return compute_MC_and_grad_MC_return_intermediate(self.o_dict, self.r_dict, self.s_dict, self.d_dict,
                                                              self.n_size, self.K, self.theta_p, self.theta_c,
                                                              self.z_rs_bar, self.y_od_bar, self.c_rs, self.seed, v)

    subalgorithm = Subalgorithm(o_dict, r_dict, s_dict, d_dict, n_size, K, theta_p, theta_c, z_rs_bar, y_od_bar, c_rs, seed)

    # Start from zero to let algorithm find direction
    v0 = np.zeros((r_size,s_size))
    v0 +=1

    start_time = time.time()
    # tighter epsilon, more iterations for better convergence
    fista = FISTA(subalgorithm, L0=1.0, eta=1.1, epsilon=1e-4, max_iter=100)
    optimal_v = fista.optimize(v0)
   
    MC_fin, _, C_rs, sum_exp, theta_c_val, P_od_rs, P_od_no_rs, P_sd_rs, node_to_index, d_nodes, o_nodes, r_nodes, s_nodes = subalgorithm.run_fr_sol(optimal_v)

    z_rs_1_values = compute_z_rs_1(z_rs_bar, C_rs, sum_exp, optimal_v, theta_c_val)
    z_rs_0_values = compute_z_rs_0(z_rs_bar, C_rs, sum_exp, optimal_v, theta_c_val)

    y_od_rs, y_rs = compute_y_distrib(
        r_size=r_size,
        n_size=n_size,
        s_size=s_size,
        o_size=len(o_nodes),
        d_size=len(d_nodes),
        y_od_bar=y_od_bar*np.ones((len(o_nodes), len(d_nodes))),
        P_od_rs=P_od_rs,
        P_od_no_rs=P_od_no_rs,
        P_sd_rs=P_sd_rs,
        node_to_index=node_to_index,
        d_nodes=d_nodes,
        o_nodes=o_nodes,
        r_nodes=r_nodes,
        s_nodes=s_nodes
    )
    end_time = time.time()
    print("MC*: ",MC_fin)
    print("v*:", np.round(optimal_v,4))
    print("Time:", np.round((end_time - start_time)*1000,3), "ms")
    print("y_rs:")
    print(np.round(y_rs,4))
    print("z_rs_1:")
    print(np.round(z_rs_1_values,4))

    plt.figure(figsize=(6, 3))
    plt.plot(fista.MC_history, marker='o', linestyle='-')
    plt.xlabel('Iterations')
    plt.ylabel('MC Value')
    plt.title('MC Value through Iterations')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

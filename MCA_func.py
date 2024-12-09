import numpy as np

def calculate_MC_and_grad_MC(o_dict, r_dict, s_dict, d_dict, n_size, rs_size, K, theta_p, theta_c, z_rs_bar, y_od_bar, c_rs, seed, v):
    # Initialize random generator
    rng = np.random.default_rng(seed)
    loc = 0  # Location parameter (mean) for Gumbel distribution
    scale = theta_c  # Scale parameter

    # DO NOT redefine v here:
    # v = np.zeros((rs_size))  # Remove this line!

    d_nodes = list(d_dict.keys())
    o_nodes = list(o_dict.keys())
    r_nodes = list(r_dict.keys())
    s_nodes = list(s_dict.keys())

    # Order of nodes: D O R S
    node_order = d_nodes + o_nodes + r_nodes + s_nodes
    d_size = len(d_nodes)
    o_size = len(o_nodes)
    r_size = len(r_nodes)
    s_size = len(s_nodes)
    t_size = len(node_order)

    t = np.zeros((t_size, t_size))  # square matrix of size (D O R S)
    node_to_index = {node: i for i, node in enumerate(node_order)}

    for o_node, links in o_dict.items():
        for dest_node, value in links.items():
            t[node_to_index[o_node], node_to_index[dest_node]] = value

    for r_node, links in r_dict.items():
        for s_node, value in links.items():
            t[node_to_index[r_node], node_to_index[s_node]] = value

    for s_node, links in s_dict.items():
        for dest_node, value in links.items():
            t[node_to_index[s_node], node_to_index[dest_node]] = value

    r_to_s_mapping = {}
    for r_node, s_links in r_dict.items():
        # Assuming each R node is connected to a single S node
        s_node = list(s_links.keys())[0]
        r_to_s_mapping[node_to_index[r_node]] = node_to_index[s_node]

    Z_sd = np.zeros((max(2, n_size - 1), s_size, d_size))
    mu_sd = np.zeros((max(2, n_size - 1), rs_size, d_size))
    P_sd_rs = np.zeros((max(2, n_size - 1), rs_size, s_size, d_size))
    d_mu_sd = np.zeros((max(2, n_size - 1), rs_size, s_size, d_size))

    bigV_rs = np.zeros((rs_size))
    sum_exp= np.zeros((rs_size))
    z_rs_1 = np.zeros((rs_size))

    Z_od = np.zeros((o_size, d_size))
    mu_od = np.zeros((o_size, d_size))
    P_od_rs = np.zeros((rs_size, o_size, d_size))
    d_mu_od = np.zeros((rs_size, o_size, d_size))

    y_od_bar = np.full((o_size, d_size), y_od_bar)
    grad_MC = np.zeros((rs_size))

    i=0
    for s in range(d_size+o_size+r_size,t_size):
        mu_sd[0,i,:]=t[s,0:d_size]
        i +=1 #TESTED, WORKS

    for n in range(1, max(2, n_size - 1)):
        for s_idx, s_node in enumerate(s_nodes):
            s_index = node_to_index[s_node]
            for d_idx, d_node in enumerate(d_nodes):
                d_index = node_to_index[d_node]
                sum_tot = 0
                for rs_idx, r_node in enumerate(r_nodes):
                    r_index = node_to_index[r_node]
                    s_corresponding = r_to_s_mapping[r_index]
                    sum_elem = np.exp(-theta_p * (
                        t[s_index, r_index] +
                        t[r_index, s_corresponding] - v[rs_idx] +
                        mu_sd[n - 1, rs_idx, d_idx]
                    ))
                    sum_tot += sum_elem
                Z_sd[n, s_idx, d_idx] = np.exp(-theta_p * t[s_index, d_index]) + sum_tot
                mu_sd[n, s_idx, d_idx] = (-1 / theta_p) * np.log(Z_sd[n, s_idx, d_idx])

                for rs_idx, r_node in enumerate(r_nodes):
                    r_index = node_to_index[r_node]
                    s_corresponding = r_to_s_mapping[r_index]
                    P_sd_rs[n, rs_idx, s_idx, d_idx] = (
                        np.exp(-theta_p * (t[s_index, r_index] + t[r_index, s_corresponding] - v[rs_idx] + mu_sd[n - 1, rs_idx, d_idx])) 
                        / Z_sd[n, s_idx, d_idx]
                    )

                for rs_part in range(rs_size):
                    sum_tot2 = 0
                    for rs_in in range(rs_size):
                        sum_elem2 = P_sd_rs[n, rs_in, s_idx, d_idx] * d_mu_sd[n - 1, rs_part, rs_in, d_idx]
                        sum_tot2 += sum_elem2
                    d_mu_sd[n, rs_part, s_idx, d_idx] = -P_sd_rs[n, rs_part, s_idx, d_idx] + sum_tot2

    epsilon = rng.gumbel(loc, scale, size=(rs_size, K))*0.1
    C_rs = c_rs + epsilon

    for rs_idx in range(rs_size):
    
        for k in range(0,2):
            sum_exp[rs_idx] +=np.exp(-theta_c*(C_rs[rs_idx,k]+k*v[rs_idx]))
            
        bigV_rs[rs_idx]=-1/theta_c*np.log(sum_exp[rs_idx])


    z_rs_1 = z_rs_bar * (np.exp(-theta_c * (C_rs[:, 1] + v))) / sum_exp

    for o_idx, o_node in enumerate(o_nodes):
        o_index = node_to_index[o_node]
        for d_idx, d_node in enumerate(d_nodes):
            d_index = node_to_index[d_node]
            sum_tot = 0
            for rs_idx, r_node in enumerate(r_nodes):
                r_index = node_to_index[r_node]
                s_corresponding = r_to_s_mapping[r_index]
                sum_elem = np.exp(-theta_p * (
                    t[o_index, r_index] +
                    t[r_index, s_corresponding] - v[rs_idx] +
                    mu_sd[n_size - 2, rs_idx, d_idx]
                ))
                sum_tot += sum_elem
            Z_od[o_idx, d_idx] = np.exp(-theta_p * t[o_index, d_index]) + sum_tot
            mu_od[o_idx, d_idx] = (-1 / theta_p)*np.log(Z_od[o_idx, d_idx])

            for rs_idx, r_node in enumerate(r_nodes):
                r_index = node_to_index[r_node]
                s_corresponding = r_to_s_mapping[r_index]
                P_od_rs[rs_idx, o_idx, d_idx] = (
                    np.exp(-theta_p * (t[o_index, r_index] + t[r_index, s_corresponding] - v[rs_idx] + mu_sd[n_size - 2, rs_idx, d_idx]))
                    / Z_od[o_idx, d_idx]
                )

            for rs_part in range(rs_size):
                sum_tot2 = 0
                for rs_in in range(rs_size):
                    sum_elem2 = P_od_rs[rs_in, o_idx, d_idx] * d_mu_sd[n_size - 2, rs_part, rs_in, d_idx]
                    sum_tot2 += sum_elem2
                d_mu_od[rs_part, o_idx, d_idx] = -P_od_rs[rs_part, o_idx, d_idx] + sum_tot2

    sum_rs = np.sum(z_rs_bar * bigV_rs)
    sum_od = np.sum(y_od_bar * mu_od)
    MC = sum_od + sum_rs

    grad_MC = np.zeros((rs_size))
    for rs_idx in range(rs_size):
        sum_part_od = np.sum(y_od_bar * d_mu_od[rs_idx, :, :])
        grad_MC[rs_idx] = sum_part_od + z_rs_1[rs_idx]

    return MC, grad_MC


